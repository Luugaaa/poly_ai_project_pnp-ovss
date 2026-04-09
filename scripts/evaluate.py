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
from utils.dataset import build_dataset, EvalSample
from utils.postprocess import postprocess, save_mask_overlay
from utils.live_board import LiveBoard
from core.patch_strategy import build_strategy
from core.salience_dropout import salience_dropout
from utils.visualize import save_patch_overview


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
    run_slug = (
        f"{ds_cfg['name']}_{ds_cfg.get('split', 'val')}"
        f"_L{pipe_cfg['layer']}_H{pipe_cfg['head']}"
        f"_{'sup' if pat_cfg['type']=='superpixel' else 'reg'}"
        f"_dr{pipe_cfg['dropout_rounds']}"
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
    wrapper   = BLIPWrapper(model_name=model_cfg["name"], device=device)

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
        "iou", "pred_coverage_pct", "gt_coverage_pct",
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
    done = ok = 0
    _overview_saved = False

    print("=" * 70)

    for sample in dataset:
        done += 1
        row = {"image_id": sample.image_id, "class_name": sample.class_name}
        t0  = time.perf_counter()

        try:
            prompt = format_prompt(sample.class_name)
            inputs = wrapper.preprocess(sample.image, prompt)
            class_token_indices = get_class_token_indices(
                wrapper.processor, prompt, sample.class_name
            )

            # Build strategy per sample (superpixels depend on the image)
            strategy = build_strategy(pat_cfg, wrapper, sample.image)

            # Save one patch-layout overview for the whole run (first image only)
            if not _overview_saved:
                save_patch_overview(
                    sample.image, strategy,
                    eval_dir / "patch_overview.png",
                    title=f"{ds_cfg['name']} — {type(strategy).__name__} ({strategy.num_segments} segments)",
                )
                _overview_saved = True

            accumulated = salience_dropout(
                wrapper             = wrapper,
                pixel_values        = inputs["pixel_values"],
                input_ids           = inputs["input_ids"],
                attention_mask      = inputs["attention_mask"],
                layer_idx           = pipe_cfg["layer"],
                head_idx            = pipe_cfg["head"],
                class_token_indices = class_token_indices,
                strategy            = strategy,
                dropout_rounds      = pipe_cfg["dropout_rounds"],
                patches_per_drop    = pipe_cfg["patches_per_drop"],
                verbose             = args.verbose,
            )

            spatial_map = strategy.to_spatial(accumulated)
            mask = postprocess(
                spatial_map  = spatial_map,
                original_image = sample.image,
                gaussian_sigma = pp_cfg["gaussian_sigma"],
                use_dense_crf  = pp_cfg["use_dense_crf"],
            )

            elapsed        = time.perf_counter() - t0
            pred_coverage  = float((mask > 0.5).mean() * 100)
            gt_coverage    = float(sample.gt_mask.mean() * 100) if sample.gt_mask is not None else -1

            iou = compute_iou(mask, sample.gt_mask) if sample.gt_mask is not None else -1

            row.update({
                "iou":              round(iou, 4) if iou >= 0 else "n/a",
                "pred_coverage_pct": round(pred_coverage, 2),
                "gt_coverage_pct":  round(gt_coverage, 2),
                "elapsed_s":        round(elapsed, 1),
                "status":           "ok",
            })

            # Accumulate for summary
            if iou >= 0:
                class_ious.setdefault(sample.class_name, []).append(iou)

            # Optional mask save
            if args.save_masks:
                out_png = masks_dir / f"{sample.image_id}.png"
                save_mask_overlay(mask, sample.image, str(out_png))

            ok += 1
            all_ious     = [v for vs in class_ious.values() for v in vs]
            running_miou = np.mean(all_ious) if all_ious else 0.0

            # Progressive board update
            if board is not None and iou >= 0:
                board.update(sample, mask, iou, running_miou, done)

            print(
                f"[{done:4d}]  {sample.image_id:<40}  "
                f"IoU={iou:.3f}  cov={pred_coverage:5.1f}%  "
                f"mIoU={running_miou:.3f}  ({elapsed:.1f}s)"
            )

        except KeyboardInterrupt:
            print("\nInterrupted.")
            row["status"] = "interrupted"
            writer.writerow({k: row.get(k, "") for k in fieldnames})
            csv_file.flush()
            break
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"[{done:4d}]  {sample.image_id}  ERROR ({elapsed:.1f}s): {exc}")
            if args.verbose:
                traceback.print_exc()
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
    lines = _build_summary(class_ious, done, ok, run_slug)
    summary_path = eval_dir / "summary.txt"
    summary_path.write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\nCSV     : {csv_path.relative_to(ROOT)}")
    print(f"Summary : {summary_path.relative_to(ROOT)}")


def _build_summary(
    class_ious: dict[str, list[float]],
    done: int,
    ok: int,
    run_slug: str,
) -> list[str]:
    lines = [
        "=" * 60,
        f"  PnP-OVSS Evaluation Summary",
        f"  Run: {run_slug}",
        "=" * 60,
        f"  Samples processed : {done}   succeeded : {ok}",
        "",
        f"  {'Class':<20}  {'N':>5}  {'mIoU':>8}",
        "  " + "-" * 38,
    ]
    all_ious: list[float] = []
    for cls_name in sorted(class_ious):
        ious    = class_ious[cls_name]
        mean_iou = np.mean(ious)
        all_ious.extend(ious)
        lines.append(f"  {cls_name:<20}  {len(ious):>5}  {mean_iou:>8.4f}")

    lines += [
        "  " + "=" * 38,
        f"  {'Overall mIoU':<20}  {len(all_ious):>5}  {np.mean(all_ious):>8.4f}" if all_ious else "  No IoU computed",
        "=" * 60,
    ]
    return lines


if __name__ == "__main__":
    main()
