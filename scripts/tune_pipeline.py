"""
Pipeline hyperparameter tuning.

Searches over patch strategy and Salience DropOut parameters using a
config-driven objective (dice or iou).
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

from scripts.experiment_runtime import ExperimentRunner, InferenceEngine
from scripts.metrics_engine import MetricsEngine
from utils.config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PnP-OVSS pipeline hyperparameter tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--verbose", action="store_true", help="Print full tracebacks on errors.")
    return p.parse_args()


def _compute_combo_metrics(rows: list[tuple[str, np.ndarray, np.ndarray | None]], dataset_spec) -> tuple[float, float, float]:
    metrics = MetricsEngine(dataset_spec)
    coverages: list[float] = []

    for class_name, pred_mask, gt_mask in rows:
        coverages.append(float((pred_mask > 0.5).mean() * 100))
        metrics.record_sample(class_name, pred_mask, gt_mask)

    all_ious = [v for vs in metrics.class_ious.values() for v in vs]
    all_dices = [v for vs in metrics.class_dices.values() for v in vs]
    miou = float(np.mean(all_ious)) if all_ious else 0.0
    mdice = float(np.mean(all_dices)) if all_dices else 0.0
    cov = float(np.mean(coverages)) if coverages else 0.0
    return miou, mdice, cov


def _render_eval_rows(
    image_groups: dict[str, dict],
    inference_engine: InferenceEngine,
    dataset_spec,
) -> tuple[list[tuple[str, np.ndarray, np.ndarray | None]], dict[str, dict[str, np.ndarray]]]:
    rows: list[tuple[str, np.ndarray, np.ndarray | None]] = []
    masks_by_image: dict[str, dict[str, np.ndarray]] = {}

    for image_uid, group in image_groups.items():
        selected_classes = sorted(set(group["classes"]))
        masks, _strategy = inference_engine.infer_multiclass_masks(
            image=group["image"],
            selected_classes=selected_classes,
            full_ensemble_classes=dataset_spec.query_class_names,
        )
        masks_by_image[image_uid] = masks

        for sample in group["samples"]:
            class_name = str(sample["class_name"])
            gt_mask = sample.get("gt_mask")
            if class_name in masks:
                pred_mask = masks[class_name]
            else:
                if gt_mask is not None:
                    pred_mask = np.zeros_like(gt_mask, dtype=np.float32)
                else:
                    h, w = group["image"].height, group["image"].width
                    pred_mask = np.zeros((h, w), dtype=np.float32)
            rows.append((class_name, pred_mask, gt_mask))

    return rows, masks_by_image


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    tp_cfg = cfg.get("tune_pipeline", {})
    ds_cfg = cfg["dataset"]

    tune_max = tp_cfg.get("max_images", 50)
    metric_name = str(tp_cfg.get("metric", "dice")).lower()
    if metric_name not in {"dice", "iou"}:
        raise ValueError(f"Unsupported tune_pipeline.metric '{metric_name}'. Use 'dice' or 'iou'.")

    slug = f"tune_pipeline_{metric_name}_{ds_cfg.get('name', 'dataset')}"
    with ExperimentRunner(
        config_path=args.config,
        root=ROOT,
        slug=slug,
        output_root="outputs",
        dataset_override={"max_images": tune_max},
    ) as runner:
        writer = runner.writer
        dataset = runner.dataset
        dataset_spec = runner.dataset_spec
        assert writer is not None and dataset is not None and dataset_spec is not None

        print(f"Run dir : {writer.run_dir.relative_to(ROOT)}")
        print(f"Metric  : {metric_name}")

        samples = list(dataset)
        if not samples:
            raise RuntimeError("No samples found for tune_pipeline dataset.")

        image_groups: dict[str, dict] = {}
        for s in samples:
            image_uid = str(s["image_uid"])
            if image_uid not in image_groups:
                image_groups[image_uid] = {
                    "image": s["image"],
                    "classes": [],
                    "samples": [],
                }
            image_groups[image_uid]["classes"].append(str(s["class_name"]))
            image_groups[image_uid]["samples"].append(s)

        print(f"Dataset : {len(samples)} samples | {len(image_groups)} unique images")

        fieldnames = [
            "strategy", "segment_param",
            "dropout_rounds", "patches_per_drop",
            "miou", "miou_pct", "mdice", "mdice_pct",
            "score", "score_pct", "metric",
            "mask_coverage_mean", "elapsed_s", "status",
        ]
        csv_name = "tune_pipeline_results.csv"
        writer.open_csv(csv_name, fieldnames)

        reg_cfg = tp_cfg.get("regular", {})
        regf_cfg = tp_cfg.get("regular_free", {})
        sp_cfg = tp_cfg.get("superpixel", {})

        regular_grid_sizes = reg_cfg.get("grid_sizes", [8, 12, 24])
        regular_free_grid_sizes = regf_cfg.get("grid_sizes", [])
        sp_n_segments_list = sp_cfg.get("n_segments", [100, 200, 400])

        patches_per_drop = tp_cfg.get("patches_per_drop", [10, 25, 50])
        r_max = int(tp_cfg.get("dropout_rounds_max", 3))

        outer_combos = (
            [("regular", f"grid{g}", g) for g in regular_grid_sizes]
            + [("regular_free", f"grid{g}", g) for g in regular_free_grid_sizes]
            + [("superpixel", f"sp{n}", n) for n in sp_n_segments_list]
        )

        results: list[dict] = []
        save_overlay_ids = set(list(image_groups.keys())[:4])

        shared_wrapper = runner.inference_engine.wrapper if runner.inference_engine is not None else None

        for strategy_type, seg_label, seg_val in outer_combos:
            print(f"\n=== {strategy_type} / {seg_label} ===")

            # R=0 baseline
            combo_points = [(0, "n/a")] + [
                (r, pdrop) for pdrop in patches_per_drop for r in range(1, r_max + 1)
            ]

            for dropout_rounds, pdrop in combo_points:
                t0 = time.perf_counter()
                row = {
                    "strategy": strategy_type,
                    "segment_param": seg_label,
                    "dropout_rounds": dropout_rounds,
                    "patches_per_drop": pdrop,
                    "metric": metric_name,
                }

                try:
                    runner.cfg["patching"]["type"] = strategy_type
                    if strategy_type == "regular":
                        runner.cfg["patching"].setdefault("regular", {})["grid_size"] = int(seg_val)
                    elif strategy_type == "regular_free":
                        runner.cfg["patching"].setdefault("regular_free", {})["grid_size"] = int(seg_val)
                    else:
                        runner.cfg["patching"].setdefault("superpixel", {})["n_segments"] = int(seg_val)

                    runner.cfg["pipeline"]["dropout_rounds"] = int(dropout_rounds)
                    if pdrop != "n/a":
                        runner.cfg["pipeline"]["patches_per_drop"] = int(pdrop)

                    inf = InferenceEngine(runner.cfg, wrapper=shared_wrapper)
                    eval_rows, masks_by_image = _render_eval_rows(image_groups, inf, dataset_spec)
                    miou, mdice, cov = _compute_combo_metrics(eval_rows, dataset_spec)
                    score = mdice if metric_name == "dice" else miou
                    elapsed = time.perf_counter() - t0

                    row.update({
                        "miou": round(miou, 4),
                        "miou_pct": round(miou * 100, 1),
                        "mdice": round(mdice, 4),
                        "mdice_pct": round(mdice * 100, 1),
                        "score": round(score, 4),
                        "score_pct": round(score * 100, 1),
                        "mask_coverage_mean": round(cov, 2),
                        "elapsed_s": round(elapsed, 1),
                        "status": "ok",
                    })

                    print(
                        f"  [R={dropout_rounds:>2} p={pdrop:>3}] "
                        f"mIoU={miou:.4f}  mDice={mdice:.4f}  cov={cov:.1f}%  ({elapsed:.1f}s)"
                    )

                    if save_overlay_ids and (dropout_rounds == r_max or dropout_rounds == 0):
                        for iid in save_overlay_ids:
                            image = image_groups[iid]["image"]
                            for cls_name, mask in masks_by_image.get(iid, {}).items():
                                writer.save_mask_overlay(
                                    rel_path=f"masks/{seg_label}_R{dropout_rounds}_p{pdrop}_{cls_name}.png",
                                    mask=mask,
                                    image=image,
                                )

                except Exception as exc:
                    elapsed = time.perf_counter() - t0
                    print(f"  [R={dropout_rounds} p={pdrop}] ERROR ({elapsed:.1f}s): {exc}")
                    if args.verbose:
                        traceback.print_exc()
                    row.update({
                        "elapsed_s": round(elapsed, 1),
                        "status": f"error:{exc}",
                    })

                results.append(row)
                writer.write_csv_row(csv_name, row, fieldnames)

        ok = [r for r in results if r.get("status") == "ok"]
        print(f"\nCompleted rows: {len(ok)}/{len(results)}")
        if ok:
            best = max(ok, key=lambda r: (r.get("score", 0), -r.get("mask_coverage_mean", 100.0)))
            print("Best config:")
            print(f"  strategy={best['strategy']}  segment_param={best['segment_param']}")
            print(f"  dropout_rounds={best['dropout_rounds']}  patches_per_drop={best['patches_per_drop']}")
            print(f"  metric={metric_name}  score={best['score']}")

            summary_lines = [
                f"metric: {metric_name}",
                f"total_rows: {len(results)}",
                f"successful_rows: {len(ok)}",
                f"best_strategy: {best['strategy']}",
                f"best_segment_param: {best['segment_param']}",
                f"best_dropout_rounds: {best['dropout_rounds']}",
                f"best_patches_per_drop: {best['patches_per_drop']}",
                f"best_miou: {best.get('miou', '')}",
                f"best_mdice: {best.get('mdice', '')}",
                f"best_score: {best.get('score', '')}",
            ]
            writer.save_text("summary.txt", summary_lines)


if __name__ == "__main__":
    main()
