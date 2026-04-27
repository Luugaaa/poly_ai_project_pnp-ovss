"""
Batch evaluation — PnP-OVSS on a segmentation dataset.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.specs import resolve_dataset_spec
from models.blip_wrapper import BLIPWrapper
from scripts.experiment_runtime import InferenceEngine
from scripts.metrics_engine import MetricsEngine
from utils.config import load_config
from utils.dataset import build_dataset
from utils.live_board import LiveBoard
from utils.postprocess import save_mask_overlay
from utils.visualize import save_patch_overview


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PnP-OVSS batch evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--dataset", default=None,
                   choices=["voc", "folder", "chest_xray"],
                   help="Override dataset type from config.")
    p.add_argument("--root", default=None,
                   help="Override dataset root from config.")
    p.add_argument("--max_images", type=int, default=None,
                   help="Override max images from config.")
    p.add_argument("--split", default=None,
                   help="Dataset split override if supported by the dataset backend.")
    p.add_argument("--save_masks", action="store_true",
                   help="Save overlay PNG for each sample.")
    p.add_argument("--no_live_viz", action="store_true",
                   help="Disable progressive visualisation board.")
    p.add_argument("--no_crf", action="store_true",
                   help="Skip Dense CRF.")
    p.add_argument("--verbose", action="store_true",
                   help="Print error tracebacks.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

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

    ds_cfg = cfg["dataset"]
    pipe_cfg = cfg["pipeline"]
    pp_cfg = cfg["postprocess"]
    pat_cfg = cfg["patching"]

    patch_type = pat_cfg.get("type", "regular")
    patch_tag = "sup" if patch_type == "superpixel" else ("regf" if patch_type == "regular_free" else "reg")
    run_slug = (
        f"{ds_cfg['name']}_{ds_cfg.get('split', 'eval')}"
        f"_L{pipe_cfg['layer']}_H{pipe_cfg['head']}"
        f"_{patch_tag}"
        f"_dr{pipe_cfg['dropout_rounds']}"
        f"{'_filt' if not pipe_cfg.get('use_full_ensemble', True) else ''}"
        f"{'_blur' if pp_cfg.get('use_blur', False) else ''}"
        f"{'_crf' if pp_cfg['use_dense_crf'] else ''}"
    )

    eval_dir = ROOT / cfg["tuning"]["out_dir"] / f"eval_{run_slug}"
    masks_dir = eval_dir / "masks"
    csv_path = eval_dir / "results.csv"
    eval_dir.mkdir(parents=True, exist_ok=True)
    if args.save_masks:
        masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output : {eval_dir.relative_to(ROOT)}/")

    import torch
    model_cfg = cfg["model"]
    device = None if model_cfg["device"] == "auto" else torch.device(model_cfg["device"])
    wrapper = BLIPWrapper(
        model_name=model_cfg["name"],
        device=device,
        input_size=model_cfg.get("image_size"),
    )
    inference_engine = InferenceEngine(cfg, wrapper=wrapper)

    print(
        f"Model  : {wrapper.num_text_layers} layers × {wrapper.num_heads} heads | "
        f"L{pipe_cfg['layer']}/H{pipe_cfg['head']}"
    )

    print(f"Dataset: {ds_cfg['name']}  split={ds_cfg.get('split','n/a')}  max_images={ds_cfg.get('max_images','all')}")
    dataset = build_dataset(ds_cfg)
    dataset_spec = getattr(dataset, "dataset_spec", None)
    if dataset_spec is None:
        dataset_name = ds_cfg.get("name")
        if dataset_name is None:
            raise ValueError("dataset.name must be provided to resolve DatasetSpec.")
        dataset_spec = resolve_dataset_spec(dataset_name)
    metrics = MetricsEngine(dataset_spec)
    if hasattr(dataset, "num_images"):
        print(f"  → {dataset.num_images()} images loaded")

    fieldnames = [
        "image_id", "class_name",
        "iou", "dice", "pred_coverage_pct", "gt_coverage_pct",
        "elapsed_s", "status",
    ]
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()

    board = None if args.no_live_viz else LiveBoard(eval_dir)

    done = ok = 0
    _overview_saved = False

    from collections import defaultdict

    grouped_samples: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for sample in dataset:
        grouped_samples[str(sample["image_uid"])].append(sample)

    print("=" * 70)
    for image_uid, samples in grouped_samples.items():
        done += len(samples)
        base_sample = samples[0]
        image = base_sample["image"]

        classes = [str(s["class_name"]) for s in samples]

        filt_cfg = pipe_cfg.get("class_filtering", {})
        filt_mode = filt_cfg.get("mode", "gt_present")
        if filt_mode == "clip_topk":
            top_k = int(filt_cfg.get("top_k", 5))
            candidate_classes = dataset_spec.query_class_names
            selected_classes = _select_clip_topk_classes(image, candidate_classes, top_k)
        else:
            selected_classes = sorted(set(classes))

        t0 = time.perf_counter()

        try:
            final_masks_dict, strategy = inference_engine.infer_multiclass_masks(
                image=image,
                selected_classes=selected_classes,
                full_ensemble_classes=dataset_spec.query_class_names,
            )

            if not _overview_saved:
                save_patch_overview(
                    image, strategy, eval_dir / "patch_overview.png",
                    title=f"{ds_cfg['name']} — {type(strategy).__name__}",
                )
                _overview_saved = True

            for sample in samples:
                sample_image_id = str(sample["image_id"])
                sample_class = str(sample["class_name"])
                sample_gt_mask = sample.get("gt_mask")
                sample_image = sample["image"]

                row = {"image_id": sample_image_id, "class_name": sample_class}
                cls_name = sample_class

                if cls_name in final_masks_dict:
                    mask = final_masks_dict[cls_name]
                else:
                    if sample_gt_mask is not None:
                        shape = sample_gt_mask.shape
                    else:
                        shape = (sample_image.height, sample_image.width)
                    mask = np.zeros(shape, dtype=np.float32)

                elapsed = time.perf_counter() - t0
                iou, dice, pred_coverage, gt_coverage = metrics.record_sample(
                    sample_class, mask, sample_gt_mask
                )

                row.update({
                    "iou": round(iou, 4) if iou >= 0 else "n/a",
                    "dice": round(dice, 4) if dice >= 0 else "n/a",
                    "pred_coverage_pct": round(pred_coverage, 2),
                    "gt_coverage_pct": round(gt_coverage, 2),
                    "elapsed_s": round(elapsed, 1),
                    "status": "ok",
                })

                if args.save_masks:
                    out_png = masks_dir / f"{sample_image_id}.png"
                    save_mask_overlay(mask, sample_image, str(out_png))

                ok += 1
                running_miou, running_mdice = metrics.running_means()

                if board is not None and iou >= 0:
                    board.update(sample, mask, iou, running_miou, done)

                print(
                    f"[{done:4d}]  {sample_image_id:<40}  "
                    f"IoU={iou:.3f}  Dice={dice:.3f}  cov={pred_coverage:5.1f}%  "
                    f"mIoU={running_miou:.3f}  mDice={running_mdice:.3f}  ({elapsed:.1f}s)"
                )

                writer.writerow({k: row.get(k, "") for k in fieldnames})
                csv_file.flush()

            metrics.update_confusion_from_label_map(
                gt_label_map=base_sample.get("gt_label_map"),
                pred_masks=final_masks_dict,
            )

        except KeyboardInterrupt:
            print("\nInterrupted.")
            for sample in samples:
                row = {
                    "image_id": str(sample["image_id"]),
                    "class_name": str(sample["class_name"]),
                    "status": "interrupted",
                }
                writer.writerow({k: row.get(k, "") for k in fieldnames})
            csv_file.flush()
            break
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"[{done:4d}]  {base_sample['image_id']}  ERROR ({elapsed:.1f}s): {exc}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            for sample in samples:
                row = {
                    "image_id": str(sample["image_id"]),
                    "class_name": str(sample["class_name"]),
                    "elapsed_s": round(elapsed, 1),
                    "status": f"error:{exc}",
                }
                writer.writerow({k: row.get(k, "") for k in fieldnames})
            csv_file.flush()

    csv_file.close()

    if board is not None:
        board.finalize(metrics.class_ious)

    lines = metrics.build_summary(done=done, ok=ok, run_slug=run_slug)
    summary_path = eval_dir / "summary.txt"
    summary_path.write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\nCSV     : {csv_path.relative_to(ROOT)}")
    print(f"Summary : {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
