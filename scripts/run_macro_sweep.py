"""
Macro orchestrator for final PnP-OVSS sweeps.

This script runs, in order:
1) Baseline transfer evaluation runs.
2) Head/layer tuning over selected metrics.
3) Patch/dropout tuning over selected strategies.

All runs enforce hard sample caps for deadline-safe execution.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.experiment_runtime import ExperimentRunner, InferenceEngine
from scripts.metrics_engine import MetricsEngine
from utils.config import load_config


def _bold(msg: str) -> str:
    return f"\033[1m{msg}\033[0m"


def _log_block(msg: str) -> None:
    print("\n" + _bold(msg))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(merged.get(k), dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _resolve_transfer_config(path_like: str) -> Path:
    p = Path(path_like)
    if not p.is_absolute():
        p = ROOT / p
    if p.exists():
        return p

    fallback = ROOT / Path(path_like).name
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Transfer config not found: {path_like}")


def _dataset_root_for(name: str) -> str:
    n = name.strip().lower()
    if n == "voc":
        return "data/voc"
    if n in {"chest_xray", "chestxray", "kaggle_chest_xray"}:
        return "data/chest_xray"
    return "data/voc"


def _apply_macro_overrides(base_cfg: dict[str, Any], macro: dict[str, Any], sample_cap: int) -> dict[str, Any]:
    cfg = _deep_merge({}, base_cfg)

    dataset_name = str(macro.get("dataset_name", "voc")).strip().lower()
    transformer = str(macro.get("transformer", "blip")).strip().lower()
    target_resolution = int(macro.get("target_resolution", 336))

    cfg.setdefault("dataset", {})
    cfg["dataset"]["name"] = dataset_name
    cfg["dataset"]["root"] = _dataset_root_for(dataset_name)
    cfg["dataset"]["max_images"] = int(sample_cap)
    cfg["dataset"]["max_samples"] = int(sample_cap)

    if dataset_name == "voc":
        cfg["dataset"].setdefault("split", "val")
        cfg["dataset"].setdefault("download", True)

    cfg.setdefault("model", {})
    cfg["model"]["image_size"] = target_resolution
    if transformer == "blip":
        model_name = str(cfg["model"].get("name", "")).lower()
        if "blip" not in model_name:
            cfg["model"]["name"] = "Salesforce/blip-itm-large-flickr"
    elif transformer == "bridgetower":
        # Kept for schema compatibility; runtime support depends on model wrapper implementation.
        cfg["model"]["name"] = "BridgeTower/bridgetower-large-itm-mlm-itc"
    else:
        raise ValueError(f"Unsupported macro_sweep.transformer '{transformer}'.")

    cfg.setdefault("tuning", {})
    cfg["tuning"]["max_images"] = int(sample_cap)

    return cfg


def _write_temp_config(cfg: dict[str, Any], prefix: str) -> Path:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix=prefix, delete=False) as tf:
        yaml.safe_dump(cfg, tf, sort_keys=False)
        return Path(tf.name)


def _group_samples(dataset) -> dict[str, list[Mapping[str, object]]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for sample in dataset:
        grouped[str(sample["image_uid"])].append(sample)
    return grouped


def _render_eval_rows(
    image_groups: dict[str, list[Mapping[str, object]]],
    inference_engine: InferenceEngine,
    dataset_spec,
) -> tuple[list[tuple[str, np.ndarray, np.ndarray | None]], dict[str, dict[str, np.ndarray]]]:
    rows: list[tuple[str, np.ndarray, np.ndarray | None]] = []
    masks_by_image: dict[str, dict[str, np.ndarray]] = {}

    for image_uid, samples in image_groups.items():
        image = samples[0]["image"]
        selected_classes = sorted({str(s["class_name"]) for s in samples})

        masks, _ = inference_engine.infer_multiclass_masks(
            image=image,
            selected_classes=selected_classes,
            full_ensemble_classes=dataset_spec.query_class_names,
        )
        masks_by_image[image_uid] = masks

        for sample in samples:
            class_name = str(sample["class_name"])
            gt_mask = sample.get("gt_mask")
            if class_name in masks:
                pred_mask = masks[class_name]
            else:
                if gt_mask is not None:
                    pred_mask = np.zeros_like(gt_mask, dtype=np.float32)
                else:
                    h, w = image.height, image.width
                    pred_mask = np.zeros((h, w), dtype=np.float32)
            rows.append((class_name, pred_mask, gt_mask))

    return rows, masks_by_image


def _compute_combo_metrics(rows: list[tuple[str, np.ndarray, np.ndarray | None]], dataset_spec) -> tuple[float, float, float]:
    metrics = MetricsEngine(dataset_spec)
    coverages: list[float] = []

    for class_name, pred_mask, gt_mask in rows:
        coverages.append(float((pred_mask > 0.5).mean() * 100))
        metrics.record_sample(class_name, pred_mask, gt_mask)

    all_ious = [v for vals in metrics.class_ious.values() for v in vals]
    all_dices = [v for vals in metrics.class_dices.values() for v in vals]
    miou = float(np.mean(all_ious)) if all_ious else 0.0
    mdice = float(np.mean(all_dices)) if all_dices else 0.0
    cov = float(np.mean(coverages)) if coverages else 0.0
    return miou, mdice, cov


def _run_baseline_transfer(config_path: Path, slug: str, max_eval_samples: int) -> None:
    with ExperimentRunner(
        config_path=str(config_path),
        root=ROOT,
        slug=slug,
        output_root="outputs",
        dataset_override={"max_images": int(max_eval_samples), "max_samples": int(max_eval_samples)},
    ) as runner:
        writer = runner.writer
        dataset = runner.dataset
        dataset_spec = runner.dataset_spec
        inf = runner.inference_engine
        assert writer is not None and dataset is not None and dataset_spec is not None and inf is not None

        samples = list(dataset)
        if not samples:
            raise RuntimeError("No samples found for transfer baseline run.")

        grouped = _group_samples(samples)

        fieldnames = [
            "image_id", "class_name",
            "iou", "dice", "pred_coverage_pct", "gt_coverage_pct",
            "elapsed_s", "status",
        ]
        csv_name = "baseline_transfer_results.csv"
        writer.open_csv(csv_name, fieldnames)

        metrics = MetricsEngine(dataset_spec)
        done = 0
        ok = 0

        print(f"Run dir : {writer.run_dir.relative_to(ROOT)}")
        print(f"Samples : {len(samples)} | Images: {len(grouped)}")

        for image_uid, image_samples in grouped.items():
            t0 = time.perf_counter()
            image = image_samples[0]["image"]
            selected_classes = sorted({str(s["class_name"]) for s in image_samples})

            masks, _ = inf.infer_multiclass_masks(
                image=image,
                selected_classes=selected_classes,
                full_ensemble_classes=dataset_spec.query_class_names,
            )

            for sample in image_samples:
                done += 1
                class_name = str(sample["class_name"])
                image_id = str(sample["image_id"])
                gt_mask = sample.get("gt_mask")
                pred = masks.get(class_name)
                if pred is None:
                    if gt_mask is not None:
                        pred = np.zeros_like(gt_mask, dtype=np.float32)
                    else:
                        pred = np.zeros((image.height, image.width), dtype=np.float32)

                elapsed = time.perf_counter() - t0
                iou, dice, pred_cov, gt_cov = metrics.record_sample(class_name, pred, gt_mask)
                row = {
                    "image_id": image_id,
                    "class_name": class_name,
                    "iou": round(iou, 4) if iou >= 0 else "n/a",
                    "dice": round(dice, 4) if dice >= 0 else "n/a",
                    "pred_coverage_pct": round(pred_cov, 2),
                    "gt_coverage_pct": round(gt_cov, 2),
                    "elapsed_s": round(elapsed, 2),
                    "status": "ok",
                }
                writer.write_csv_row(csv_name, row, fieldnames)
                ok += 1

            metrics.update_confusion_from_label_map(
                gt_label_map=image_samples[0].get("gt_label_map"),
                pred_masks=masks,
            )

        lines = metrics.build_summary(done=done, ok=ok, run_slug=slug)
        writer.save_text("summary.txt", lines)


def _run_head_layer_tuning(metric: str, config_path: Path) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "tune_hyperparams.py"),
        "--config",
        str(config_path),
        "--metric",
        metric,
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _run_patch_tuning(
    config_path: Path,
    strategies: list[str],
    dropout_iterations: list[int],
    max_tune_samples: int,
) -> None:
    slug = "macro_patch_tuning"
    with ExperimentRunner(
        config_path=str(config_path),
        root=ROOT,
        slug=slug,
        output_root="outputs",
        dataset_override={"max_images": int(max_tune_samples), "max_samples": int(max_tune_samples)},
    ) as runner:
        writer = runner.writer
        dataset = runner.dataset
        dataset_spec = runner.dataset_spec
        assert writer is not None and dataset is not None and dataset_spec is not None

        samples = list(dataset)
        if not samples:
            raise RuntimeError("No samples found for patch tuning run.")

        image_groups = _group_samples(samples)
        print(f"Run dir : {writer.run_dir.relative_to(ROOT)}")
        print(f"Samples : {len(samples)} | Images: {len(image_groups)}")

        fieldnames = [
            "strategy", "dropout_rounds", "miou", "miou_pct",
            "mdice", "mdice_pct", "mask_coverage_mean", "elapsed_s", "status",
        ]
        csv_name = "macro_patch_tuning_results.csv"
        writer.open_csv(csv_name, fieldnames)

        results: list[dict[str, Any]] = []
        shared_wrapper = runner.inference_engine.wrapper if runner.inference_engine is not None else None

        for strategy in strategies:
            st = strategy.strip().lower()
            if st not in {"regular", "superpixel", "regular_free"}:
                raise ValueError(f"Unsupported patch strategy '{strategy}'.")

            runner.cfg.setdefault("patching", {})
            runner.cfg["patching"]["type"] = st
            if st == "regular":
                runner.cfg["patching"].setdefault("regular", {}).setdefault("grid_size", 21)
            elif st == "regular_free":
                runner.cfg["patching"].setdefault("regular_free", {}).setdefault("grid_size", 21)
            else:
                runner.cfg["patching"].setdefault("superpixel", {}).setdefault("n_segments", 256)

            for dr in dropout_iterations:
                dr_i = int(dr)
                t0 = time.perf_counter()
                row: dict[str, Any] = {
                    "strategy": st,
                    "dropout_rounds": dr_i,
                }

                try:
                    runner.cfg.setdefault("pipeline", {})
                    runner.cfg["pipeline"]["dropout_rounds"] = dr_i

                    inf = InferenceEngine(runner.cfg, wrapper=shared_wrapper)
                    eval_rows, _ = _render_eval_rows(image_groups, inf, dataset_spec)
                    miou, mdice, cov = _compute_combo_metrics(eval_rows, dataset_spec)
                    elapsed = time.perf_counter() - t0

                    row.update({
                        "miou": round(miou, 4),
                        "miou_pct": round(miou * 100, 1),
                        "mdice": round(mdice, 4),
                        "mdice_pct": round(mdice * 100, 1),
                        "mask_coverage_mean": round(cov, 2),
                        "elapsed_s": round(elapsed, 2),
                        "status": "ok",
                    })

                    print(
                        f"  strategy={st:<11} dr={dr_i:<2} "
                        f"mIoU={miou:.4f} mDice={mdice:.4f} cov={cov:.1f}% ({elapsed:.1f}s)"
                    )

                except Exception as exc:
                    elapsed = time.perf_counter() - t0
                    row.update({
                        "elapsed_s": round(elapsed, 2),
                        "status": f"error:{exc}",
                    })
                    print(f"  strategy={st:<11} dr={dr_i:<2} ERROR ({elapsed:.1f}s): {exc}")

                results.append(row)
                writer.write_csv_row(csv_name, row, fieldnames)

        ok_rows = [r for r in results if r.get("status") == "ok"]
        summary = [
            f"total_rows: {len(results)}",
            f"successful_rows: {len(ok_rows)}",
        ]
        if ok_rows:
            best = max(ok_rows, key=lambda r: r.get("mdice", 0.0))
            summary.extend([
                f"best_strategy: {best.get('strategy')}",
                f"best_dropout_rounds: {best.get('dropout_rounds')}",
                f"best_miou: {best.get('miou')}",
                f"best_mdice: {best.get('mdice')}",
            ])
        writer.save_text("summary.txt", summary)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run macro PnP-OVSS sweeps from configs/macro_config.yaml.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--macro-config", default="configs/macro_config.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    macro_path = Path(args.macro_config)
    if not macro_path.is_absolute():
        macro_path = ROOT / macro_path
    if not macro_path.exists():
        raise FileNotFoundError(f"Macro config not found: {macro_path}")

    macro_root = yaml.safe_load(macro_path.read_text()) or {}
    macro = macro_root.get("macro_sweep", {})
    if not macro:
        raise ValueError("macro_config.yaml must contain a 'macro_sweep' mapping.")

    max_eval_samples = int(macro.get("max_eval_samples", 200))
    max_tune_samples = int(macro.get("max_tune_samples", 30))

    transfer_cfgs = [
        _resolve_transfer_config(p)
        for p in macro.get("transfer_configs", [])
    ]
    base_tune_cfg = transfer_cfgs[0] if transfer_cfgs else (ROOT / "config_paper_baseline.yaml")

    if transfer_cfgs:
        _log_block("=== START: BASELINE TRANSFER SWEEP ===")
        for cfg_path in transfer_cfgs:
            print(_bold(f"Transfer config: {cfg_path.relative_to(ROOT)}"))
            raw_cfg = load_config(cfg_path)
            run_cfg = _apply_macro_overrides(raw_cfg, macro, sample_cap=max_eval_samples)
            temp_cfg = _write_temp_config(run_cfg, prefix="macro_transfer_")
            slug = f"transfer_{cfg_path.stem}_{run_cfg['dataset']['name']}"
            _run_baseline_transfer(temp_cfg, slug, max_eval_samples=max_eval_samples)
        _log_block("=== END: BASELINE TRANSFER SWEEP ===")

    head_cfg = macro.get("head_layer_tuning", {})
    if bool(head_cfg.get("run", False)):
        _log_block("=== START: HEAD/LAYER TUNING SWEEP ===")
        metrics = [str(m).strip().lower() for m in head_cfg.get("metrics", ["clip", "dice"])]
        for metric in metrics:
            if metric not in {"clip", "dice"}:
                raise ValueError(f"Unsupported head_layer_tuning metric '{metric}'.")
            print(_bold(f"Head/Layer metric: {metric}"))
            raw_cfg = load_config(base_tune_cfg)
            run_cfg = _apply_macro_overrides(raw_cfg, macro, sample_cap=max_tune_samples)
            run_cfg.setdefault("tuning", {})
            run_cfg["tuning"]["metric"] = metric
            run_cfg["tuning"]["max_images"] = max_tune_samples
            temp_cfg = _write_temp_config(run_cfg, prefix=f"macro_heads_{metric}_")
            _run_head_layer_tuning(metric=metric, config_path=temp_cfg)
        _log_block("=== END: HEAD/LAYER TUNING SWEEP ===")

    patch_cfg = macro.get("patch_settings_tuning", {})
    if bool(patch_cfg.get("run", False)):
        _log_block("=== START: PATCH TUNING SWEEP ===")
        strategies = [str(s).strip().lower() for s in patch_cfg.get("strategies", ["regular", "superpixel"])]
        dropout_iterations = [int(v) for v in patch_cfg.get("dropout_iterations_range", [1, 2, 3, 4, 5])]
        if not dropout_iterations:
            raise ValueError("patch_settings_tuning.dropout_iterations_range cannot be empty.")

        raw_cfg = load_config(base_tune_cfg)
        run_cfg = _apply_macro_overrides(raw_cfg, macro, sample_cap=max_tune_samples)
        run_cfg.setdefault("tune_pipeline", {})
        run_cfg["tune_pipeline"]["max_images"] = max_tune_samples

        temp_cfg = _write_temp_config(run_cfg, prefix="macro_patch_")
        _run_patch_tuning(
            config_path=temp_cfg,
            strategies=strategies,
            dropout_iterations=dropout_iterations,
            max_tune_samples=max_tune_samples,
        )
        _log_block("=== END: PATCH TUNING SWEEP ===")

    _log_block("=== MACRO SWEEP COMPLETE ===")


if __name__ == "__main__":
    main()
