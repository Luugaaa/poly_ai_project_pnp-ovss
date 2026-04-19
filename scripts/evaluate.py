"""
Batch evaluation — PnP-OVSS on a segmentation dataset
=======================================================
Runs the full pipeline on every (image, class) sample in the dataset,
computes binary IoU against ground-truth masks, and writes results
progressively to a CSV.

Usage
-----
    python3 scripts/evaluate.py                        # VOC val, 200 images
    python3 scripts/evaluate.py --config config.yaml
    python3 scripts/evaluate.py --max_images 50        # quick smoke-test
    python3 scripts/evaluate.py --dataset folder --root data/myset
    ./run.sh eval                                      # via launcher

Output
------
    experiments/eval_{run_slug}/
      results.csv      — per-sample rows, written progressively
      summary.txt      — mIoU per class + overall, written at the end
      masks/           — overlay PNGs (one per sample, optional)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.blip_wrapper import BLIPWrapper
from utils.config import load_config, make_run_slug
from utils.data_loader import format_prompt, get_class_token_indices
from utils.dataset import build_dataset, EvalSample, VOC_CLASSES
from utils.postprocess import postprocess_multiclass, postprocess, save_mask_overlay
from utils.live_board import LiveBoard
from core.patch_strategy import build_strategy
from core.salience_dropout import salience_dropout
from utils.visualize import save_patch_overview


# Prompt aliases to match original PnP-OVSS vocabulary.
PROMPT_CLASS_ALIAS = {
    "diningtable": "table",
}


def _select_clip_topk_classes(image, candidates, k):
    """Select top-k image-level classes with CLIP logits over candidate prompts."""
    from transformers import CLIPModel, CLIPProcessor
    import torch

    if not hasattr(_select_clip_topk_classes, "_cache"):
        model_name = "openai/clip-vit-large-patch14-336"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name).to(device)
        model.eval()
        _select_clip_topk_classes._cache = (processor, model, device)

    processor, model, device = _select_clip_topk_classes._cache

    texts = [f"a photo of a {c}" for c in candidates]
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    inputs = {kk: vv.to(device) for kk, vv in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits_per_image[0]
        probs = torch.softmax(logits, dim=0)
        topk = min(k, len(candidates))
        idx = torch.topk(probs, k=topk).indices.cpu().tolist()
    return [candidates[i] for i in idx]


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PnP-OVSS batch evaluation with mIoU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",      default="config.yaml")
    p.add_argument("--dataset",     default=None,
                   choices=["voc", "folder"],
                   help="Override dataset type from config.")
    p.add_argument("--root",        default=None,
                   help="Override dataset root from config.")
    p.add_argument("--max_images",  type=int, default=None,
                   help="Override max images from config.")
    p.add_argument("--split",       default=None,
                   choices=["train", "val", "trainval"],
                   help="VOC split (overrides config).")
    p.add_argument("--save_masks",  action="store_true",
                   help="Save overlay PNG for each sample (slow, lots of disk).")
    p.add_argument("--no_live_viz", action="store_true",
                   help="Disable progressive visualisation board.")
    p.add_argument("--no_crf",      action="store_true",
                   help="Skip Dense CRF.")
    p.add_argument("--verbose",     action="store_true",
                   help="Print per-pass dropout logs and error tracebacks.")
    return p.parse_args()


# ── IoU ───────────────────────────────────────────────────────────────────────

def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Binary IoU between predicted and ground-truth boolean masks.

    Parameters
    ----------
    pred_mask : ndarray [H, W]  float32 ∈ [0, 1]  (continuous confidence)
    gt_mask   : ndarray [H, W]  bool

    The prediction is binarised at 0.5 (post-CRF output is already in {0,1};
    Gaussian-only output is binarised here).
    """
    pred_bin = pred_mask > 0.5
    inter    = np.logical_and(pred_bin, gt_mask).sum()
    union    = np.logical_or(pred_bin, gt_mask).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Binary Dice score between predicted and ground-truth boolean masks."""
    pred_bin = pred_mask > 0.5
    inter    = np.logical_and(pred_bin, gt_mask).sum()
    pred_sum = pred_bin.sum()
    gt_sum   = gt_mask.sum()
    denom    = pred_sum + gt_sum
    if denom == 0:
        return 1.0
    return float(2 * inter) / float(denom)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Config ────────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    if args.no_crf:
        cfg["postprocess"]["use_dense_crf"] = False
    if args.max_images is not None:
        cfg["dataset"]["max_images"] = args.max_images
    if args.dataset is not None:
        cfg["dataset"]["name"] = args.dataset
    if args.root is not None:
        cfg["dataset"]["root"] = args.root
    if args.split is not None:
        cfg["dataset"]["split"] = args.split

    ds_cfg   = cfg["dataset"]
    pipe_cfg = cfg["pipeline"]
    pp_cfg   = cfg["postprocess"]
    pat_cfg  = cfg["patching"]

    # ── Output dirs ───────────────────────────────────────────────────────
    patch_type = pat_cfg.get("type", "regular")
    patch_tag = "sup" if patch_type == "superpixel" else ("regf" if patch_type == "regular_free" else "reg")
    run_slug = (
        f"{ds_cfg['name']}_{ds_cfg.get('split', 'val')}"
        f"_L{pipe_cfg['layer']}_H{pipe_cfg['head']}"
        f"_{patch_tag}"
        f"_dr{pipe_cfg['dropout_rounds']}"
        f"{'_filt' if not pipe_cfg.get('use_full_ensemble', True) else ''}"
        f"{'_blur' if pp_cfg.get('use_blur', False) else ''}"
        f"{'_crf' if pp_cfg['use_dense_crf'] else ''}"
    )
    eval_dir  = ROOT / cfg["tuning"]["out_dir"] / f"eval_{run_slug}"
    masks_dir = eval_dir / "masks"
    csv_path  = eval_dir / "results.csv"
    eval_dir.mkdir(parents=True, exist_ok=True)
    if args.save_masks:
        masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output : {eval_dir.relative_to(ROOT)}/")

    # ── Model ─────────────────────────────────────────────────────────────
    import torch
    model_cfg = cfg["model"]
    device    = None if model_cfg["device"] == "auto" else torch.device(model_cfg["device"])
    wrapper   = BLIPWrapper(
        model_name=model_cfg["name"],
        device=device,
        input_size=model_cfg.get("image_size"),
    )

    print(
        f"Model  : {wrapper.num_text_layers} layers × {wrapper.num_heads} heads | "
        f"L{pipe_cfg['layer']}/H{pipe_cfg['head']}"
    )

    # ── Dataset ───────────────────────────────────────────────────────────
    print(f"Dataset: {ds_cfg['name']}  split={ds_cfg.get('split','val')}  "
          f"max_images={ds_cfg.get('max_images','all')}")
    dataset = build_dataset(ds_cfg)
    if hasattr(dataset, "num_images"):
        print(f"  → {dataset.num_images()} images loaded")

    # ── CSV ───────────────────────────────────────────────────────────────
    fieldnames = [
        "image_id", "class_name",
        "iou", "dice", "pred_coverage_pct", "gt_coverage_pct",
        "elapsed_s", "status",
    ]
    csv_file = open(csv_path, "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()

    # ── Progressive visualisation ────────────────────────────────────────
    board = None if args.no_live_viz else LiveBoard(eval_dir)
    if board is not None:
        print(f"Live viz: open {eval_dir.relative_to(ROOT)}/progress_board.png "
              f"in an auto-reloading viewer")

    # ── Per-class accumulators ────────────────────────────────────────────
    class_ious: dict[str, list[float]] = {}
    class_dices: dict[str, list[float]] = {}
    global_hist = np.zeros((len(VOC_CLASSES), len(VOC_CLASSES)), dtype=np.float64)
    done = ok = 0
    _overview_saved = False

    print("=" * 70)

    # ── Grouping samples by image ──────────────────────────────────────────
    from collections import defaultdict
    grouped_samples = defaultdict(list)
    for sample in dataset:
        grouped_samples[id(sample.image)].append(sample)

    for image_id_group, samples in grouped_samples.items():
        done += len(samples)
        
        # Use the first sample's image
        base_sample = samples[0]
        image = base_sample.image
        
        classes = [s.class_name for s in samples]

        filt_cfg = pipe_cfg.get("class_filtering", {})
        filt_mode = filt_cfg.get("mode", "gt_present")
        if filt_mode == "clip_topk":
            top_k = int(filt_cfg.get("top_k", 5))
            candidate_classes = VOC_CLASSES[1:]
            selected_classes = _select_clip_topk_classes(image, candidate_classes, top_k)
        else:
            selected_classes = sorted(set(classes))

        prompt_classes = [PROMPT_CLASS_ALIAS.get(c, c) for c in selected_classes]
        
        t0  = time.perf_counter()

        try:
            use_full_ensemble = pipe_cfg.get("use_full_ensemble", True)
            acc_full = None

            # Optional pass 1: Full Ensemble (All 20 VOC classes)
            if use_full_ensemble:
                all_classes = VOC_CLASSES[1:]  # skip 'background'
                all_prompt_classes = [PROMPT_CLASS_ALIAS.get(c, c) for c in all_classes]
                prompt_full = format_prompt(all_prompt_classes)
                inputs_full = wrapper.preprocess(image, prompt_full)
                indices_full = get_class_token_indices(
                    wrapper.processor, prompt_full, all_prompt_classes
                )
            
            strategy = build_strategy(pat_cfg, wrapper, image)
            if not _overview_saved:
                save_patch_overview(
                    image, strategy, eval_dir / "patch_overview.png",
                    title=f"{ds_cfg['name']} — {type(strategy).__name__}"
                )
                _overview_saved = True

            if use_full_ensemble:
                acc_full = salience_dropout(
                    wrapper=wrapper, pixel_values=inputs_full["pixel_values"],
                    input_ids=inputs_full["input_ids"], attention_mask=inputs_full["attention_mask"],
                    layer_idx=pipe_cfg["layer"], head_idx=pipe_cfg["head"],
                    class_token_indices=indices_full, strategy=strategy,
                    dropout_rounds=pipe_cfg["dropout_rounds"], patches_per_drop=pipe_cfg["patches_per_drop"],
                    verbose=False
                )
            
            # Pass 2: Filtered Ensemble (class-filtered subset)
            prompt_filt = format_prompt(prompt_classes)
            inputs_filt = wrapper.preprocess(image, prompt_filt)
            indices_filt = get_class_token_indices(
                wrapper.processor, prompt_filt, prompt_classes
            )
            
            acc_filt = salience_dropout(
                wrapper=wrapper, pixel_values=inputs_filt["pixel_values"],
                input_ids=inputs_filt["input_ids"], attention_mask=inputs_filt["attention_mask"],
                layer_idx=pipe_cfg["layer"], head_idx=pipe_cfg["head"],
                class_token_indices=indices_filt, strategy=strategy,
                dropout_rounds=pipe_cfg["dropout_rounds"], patches_per_drop=pipe_cfg["patches_per_drop"],
                verbose=False
            )

            # Build final spatial maps (filtered-only or mean of full+filtered)
            spatial_maps_dict = {}
            for class_name in selected_classes:
                prompt_cls = PROMPT_CLASS_ALIAS.get(class_name, class_name)
                filt_spatial = strategy.to_spatial(acc_filt[prompt_cls])
                if use_full_ensemble and acc_full is not None:
                    full_spatial = strategy.to_spatial(acc_full[prompt_cls])
                    spatial_maps_dict[class_name] = (full_spatial + filt_spatial) / 2.0
                else:
                    spatial_maps_dict[class_name] = filt_spatial
                
            final_masks_dict = postprocess_multiclass(
                spatial_maps = spatial_maps_dict,
                original_image = image,
                threshold = pp_cfg.get("threshold", 0.15),
                gaussian_sigma = pp_cfg["gaussian_sigma"],
                use_blur = pp_cfg.get("use_blur", False),
                use_dense_crf  = pp_cfg["use_dense_crf"],
            )

            for sample in samples:
                row = {"image_id": sample.image_id, "class_name": sample.class_name}
                cls_name = sample.class_name

                if cls_name in final_masks_dict:
                    mask = final_masks_dict[cls_name]
                else:
                    mask = np.zeros(sample.gt_mask.shape, dtype=np.float32)
                
                elapsed        = time.perf_counter() - t0
                pred_coverage  = float((mask > 0.5).mean() * 100)
                gt_coverage    = float(sample.gt_mask.mean() * 100) if sample.gt_mask is not None else -1

                iou = compute_iou(mask, sample.gt_mask) if sample.gt_mask is not None else -1
                dice = compute_dice(mask, sample.gt_mask) if sample.gt_mask is not None else -1

                row.update({
                    "iou":              round(iou, 4) if iou >= 0 else "n/a",
                    "dice":             round(dice, 4) if dice >= 0 else "n/a",
                    "pred_coverage_pct": round(pred_coverage, 2),
                    "gt_coverage_pct":  round(gt_coverage, 2),
                    "elapsed_s":        round(elapsed, 1),
                    "status":           "ok",
                })

                # Accumulate for summary
                if iou >= 0:
                    class_ious.setdefault(sample.class_name, []).append(iou)
                if dice >= 0:
                    class_dices.setdefault(sample.class_name, []).append(dice)

                # Optional mask save
                if args.save_masks:
                    out_png = masks_dir / f"{sample.image_id}.png"
                    save_mask_overlay(mask, sample.image, str(out_png))

                ok += 1
                all_ious     = [v for vs in class_ious.values() for v in vs]
                all_dices    = [v for vs in class_dices.values() for v in vs]
                running_miou = np.mean(all_ious) if all_ious else 0.0
                running_mdice = np.mean(all_dices) if all_dices else 0.0

                # Progressive board update
                if board is not None and iou >= 0:
                    board.update(sample, mask, iou, running_miou, done)

                print(
                    f"[{done:4d}]  {sample.image_id:<40}  "
                    f"IoU={iou:.3f}  Dice={dice:.3f}  cov={pred_coverage:5.1f}%  "
                    f"mIoU={running_miou:.3f}  mDice={running_mdice:.3f}  ({elapsed:.1f}s)"
                )
                
                writer.writerow({k: row.get(k, "") for k in fieldnames})
                csv_file.flush()

            # Original-style multiclass confusion matrix update for VOC
            if base_sample.gt_label_map is not None:
                gt = base_sample.gt_label_map.astype(np.int64)
                pred = np.zeros_like(gt, dtype=np.int64)
                for cls_name, mask in final_masks_dict.items():
                    cls_idx = VOC_CLASSES.index(cls_name)
                    pred[mask > 0.5] = cls_idx

                valid = (gt >= 0) & (gt < len(VOC_CLASSES))
                hist = np.bincount(
                    len(VOC_CLASSES) * gt[valid] + pred[valid],
                    minlength=len(VOC_CLASSES) ** 2,
                ).reshape(len(VOC_CLASSES), len(VOC_CLASSES))
                global_hist += hist

        except KeyboardInterrupt:
            print("\nInterrupted.")
            for sample in samples:
                row = {"image_id": sample.image_id, "class_name": sample.class_name, "status": "interrupted"}
                writer.writerow({k: row.get(k, "") for k in fieldnames})
            csv_file.flush()
            break
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"[{done:4d}]  {base_sample.image_id}  ERROR ({elapsed:.1f}s): {exc}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            for sample in samples:
                row = {"image_id": sample.image_id, "class_name": sample.class_name}
                row.update({"elapsed_s": round(elapsed, 1), "status": f"error:{exc}"})
                writer.writerow({k: row.get(k, "") for k in fieldnames})
            csv_file.flush()
    csv_file.close()

    # ── Finalise visualisation ────────────────────────────────────────────
    if board is not None:
        board.finalize(class_ious)
        print(f"\nVisualisation files:")
        print(f"  progress_board.png  — rolling grid of all results")
        print(f"  miou_curve.png      — running mIoU over time")
        print(f"  summary_chart.png   — per-class mIoU bar chart")

    # ── Summary ───────────────────────────────────────────────────────────
    lines = _build_summary(class_ious, class_dices, done, ok, run_slug, global_hist)
    summary_path = eval_dir / "summary.txt"
    summary_path.write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\nCSV     : {csv_path.relative_to(ROOT)}")
    print(f"Summary : {summary_path.relative_to(ROOT)}")


def _build_summary(
    class_ious: dict[str, list[float]],
    class_dices: dict[str, list[float]],
    done: int,
    ok: int,
    run_slug: str,
    global_hist: np.ndarray,
) -> list[str]:
    lines = [
        "=" * 60,
        f"  PnP-OVSS Evaluation Summary",
        f"  Run: {run_slug}",
        "=" * 60,
        f"  Samples processed : {done}   succeeded : {ok}",
        "",
        f"  {'Class':<20}  {'N':>5}  {'mIoU':>8}  {'mDice':>8}",
        "  " + "-" * 49,
    ]
    all_ious: list[float] = []
    all_dices: list[float] = []
    for cls_name in sorted(class_ious):
        ious    = class_ious[cls_name]
        dices   = class_dices.get(cls_name, [])
        mean_iou = np.mean(ious)
        mean_dice = np.mean(dices) if dices else float("nan")
        all_ious.extend(ious)
        all_dices.extend(dices)
        lines.append(f"  {cls_name:<20}  {len(ious):>5}  {mean_iou:>8.4f}  {mean_dice:>8.4f}")

    if all_ious:
        mean_iou = float(np.mean(all_ious))
        mean_dice = float(np.mean(all_dices)) if all_dices else float("nan")
        lines += [
            "  " + "=" * 49,
            f"  {'Overall Mean IoU':<20}  {len(all_ious):>5}  {mean_iou:>8.4f}",
            f"  {'Overall Mean Dice':<20}  {len(all_dices):>5}  {mean_dice:>8.4f}",
            "=" * 60,
        ]
    else:
        lines += [
            "  No IoU/Dice computed",
            "=" * 60,
        ]

    if global_hist.sum() > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            iu = np.diag(global_hist) / (
                global_hist.sum(axis=1) + global_hist.sum(axis=0) - np.diag(global_hist)
            )
            dice = (2 * np.diag(global_hist)) / (
                global_hist.sum(axis=1) + global_hist.sum(axis=0)
            )
        valid = global_hist.sum(axis=1) > 0
        global_miou = float(np.nanmean(iu[valid]))
        global_mdice = float(np.nanmean(dice[valid]))
        lines += [
            "",
            "Original-style Global Metrics (multiclass confusion matrix)",
            f"  Global Mean IoU (incl. background): {global_miou:.4f}",
            f"  Background IoU: {float(iu[0]):.4f}",
            f"  Object-only Mean IoU: {float(np.nanmean(iu[1:])):.4f}",
            f"  Global Mean Dice (incl. background): {global_mdice:.4f}",
            f"  Background Dice: {float(dice[0]):.4f}",
            f"  Object-only Mean Dice: {float(np.nanmean(dice[1:])):.4f}",
        ]
    return lines


if __name__ == "__main__":
    main()
