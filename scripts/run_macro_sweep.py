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
import json
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import os
# Prevent CUDA memory fragmentation across many GradCAM passes in a long sweep.
# Must be set before the CUDA allocator initialises (first .cuda() call).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
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
        cfg["model"]["name"] = "BridgeTower/bridgetower-large-itm-mlm-itc"
    elif transformer == "albef":
        cfg["model"]["name"] = "Salesforce/albef-base-itm"
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

        # Release VRAM between images to prevent OOM during multi-combo sweeps.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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


def _run_baseline_transfer(
    config_path: Path,
    slug: str,
    max_eval_samples: int,
    output_dir: Path | None = None,
) -> None:
    with ExperimentRunner(
        config_path=str(config_path),
        root=ROOT,
        slug=slug,
        output_root="outputs",
        dataset_override={"max_images": int(max_eval_samples), "max_samples": int(max_eval_samples)},
        run_dir_override=output_dir,
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


def _run_head_layer_tuning(metric: str, config_path: Path) -> dict[str, Any] | None:
    """Run layer/head tuning and return {layer, head, score, metric} for the winner, or None."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", prefix="best_pipeline_", delete=False) as tf:
        best_json_path = Path(tf.name)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "tune_hyperparams.py"),
        "--config", str(config_path),
        "--metric", metric,
        "--output-best-json", str(best_json_path),
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)

    if best_json_path.exists():
        try:
            return json.loads(best_json_path.read_text())
        except Exception:
            pass
    return None


def _run_patch_tuning(
    config_path: Path,
    regular_grid_sizes: list[int],
    superpixel_segments: list[int],
    patches_per_drop_range: list[int],
    dropout_iterations: list[int],
    threshold_range: list[float],
    max_tune_samples: int,
    output_dir: Path | None = None,
    best_combo_out: Path | None = None,
) -> None:
    slug = "macro_patch_tuning"
    with ExperimentRunner(
        config_path=str(config_path),
        root=ROOT,
        slug=slug,
        output_root="outputs",
        dataset_override={"max_images": int(max_tune_samples), "max_samples": int(max_tune_samples)},
        run_dir_override=output_dir,
    ) as runner:
        writer = runner.writer
        dataset = runner.dataset
        dataset_spec = runner.dataset_spec
        assert writer is not None and dataset is not None and dataset_spec is not None

        samples = list(dataset)
        if not samples:
            raise RuntimeError("No samples found for patch tuning run.")

        image_groups = _group_samples(samples)
        # Hard-cap unique images to max_tune_samples regardless of dataset loader behaviour.
        image_groups = dict(list(image_groups.items())[:max_tune_samples])
        # Enforce 336×336 input resolution for all strategies in this sweep.
        runner.cfg.setdefault("model", {})["image_size"] = 336
        print(f"Run dir : {writer.run_dir.relative_to(ROOT)}")
        print(f"Samples : {len(samples)} | Images (capped): {len(image_groups)}")

        fieldnames = [
            "strategy", "granularity", "patches_per_drop", "dropout_rounds", "threshold",
            "miou", "miou_pct", "mdice", "mdice_pct", "mask_coverage_mean",
            "elapsed_s", "status",
        ]
        csv_name = "macro_patch_tuning_results.csv"
        writer.open_csv(csv_name, fieldnames)

        results: list[dict[str, Any]] = []
        shared_wrapper = runner.inference_engine.wrapper if runner.inference_engine is not None else None

        # ── Regular grid strategy loop ────────────────────────────────────────
        # Uses regular_free (pixel-space G×G grid) to support sizes beyond the
        # ViT patch grid limit (num_patches_per_side ≈ 21 for BLIP 336px/16px).
        for grid_size in regular_grid_sizes:
            runner.cfg.setdefault("patching", {})
            runner.cfg["patching"]["type"] = "regular_free"
            runner.cfg["patching"].setdefault("regular_free", {})["grid_size"] = grid_size

            for ppd in patches_per_drop_range:
                runner.cfg.setdefault("pipeline", {})["patches_per_drop"] = int(ppd)

                for dr in dropout_iterations:
                    runner.cfg["pipeline"]["dropout_rounds"] = int(dr)

                    for thresh in threshold_range:
                        runner.cfg.setdefault("postprocess", {})["threshold"] = float(thresh)

                        t0 = time.perf_counter()
                        row: dict[str, Any] = {
                            "strategy": "regular",
                            "granularity": grid_size,
                            "patches_per_drop": int(ppd),
                            "dropout_rounds": int(dr),
                            "threshold": thresh,
                        }

                        try:
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
                                f"  regular   g={grid_size:<3} ppd={int(ppd):<3} dr={int(dr):<2} thr={thresh:.2f} "
                                f"mIoU={miou:.4f} mDice={mdice:.4f} cov={cov:.1f}% ({elapsed:.1f}s)"
                            )

                        except Exception as exc:
                            elapsed = time.perf_counter() - t0
                            row.update({"elapsed_s": round(elapsed, 2), "status": f"error:{exc}"})
                            print(
                                f"  regular   g={grid_size:<3} ppd={int(ppd):<3} dr={int(dr):<2} thr={thresh:.2f} "
                                f"ERROR ({elapsed:.1f}s): {exc}"
                            )

                        results.append(row)
                        writer.write_csv_row(csv_name, row, fieldnames)

        # ── Superpixel strategy loop ──────────────────────────────────────────
        for n_segs in superpixel_segments:
            runner.cfg.setdefault("patching", {})
            runner.cfg["patching"]["type"] = "superpixel"
            runner.cfg["patching"].setdefault("superpixel", {})["n_segments"] = n_segs

            for ppd in patches_per_drop_range:
                runner.cfg.setdefault("pipeline", {})["patches_per_drop"] = int(ppd)

                for dr in dropout_iterations:
                    runner.cfg["pipeline"]["dropout_rounds"] = int(dr)

                    for thresh in threshold_range:
                        runner.cfg.setdefault("postprocess", {})["threshold"] = float(thresh)

                        t0 = time.perf_counter()
                        row: dict[str, Any] = {
                            "strategy": "superpixel",
                            "granularity": n_segs,
                            "patches_per_drop": int(ppd),
                            "dropout_rounds": int(dr),
                            "threshold": thresh,
                        }

                        try:
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
                                f"  superpixel n={n_segs:<4} ppd={int(ppd):<3} dr={int(dr):<2} thr={thresh:.2f} "
                                f"mIoU={miou:.4f} mDice={mdice:.4f} cov={cov:.1f}% ({elapsed:.1f}s)"
                            )

                        except Exception as exc:
                            elapsed = time.perf_counter() - t0
                            row.update({"elapsed_s": round(elapsed, 2), "status": f"error:{exc}"})
                            print(
                                f"  superpixel n={n_segs:<4} ppd={int(ppd):<3} dr={int(dr):<2} thr={thresh:.2f} "
                                f"ERROR ({elapsed:.1f}s): {exc}"
                            )

                        results.append(row)
                        writer.write_csv_row(csv_name, row, fieldnames)

        # ── Best config selection by mDice ────────────────────────────────────
        ok_rows = [r for r in results if r.get("status") == "ok"]

        # ── Zero-dice diagnostic ──────────────────────────────────────────────
        # If every successful combo returned mDice=0.0, run a single sanity
        # check at threshold=0.01 to distinguish "maps are sparse" from
        # "attention maps are broken/empty".
        if ok_rows and all(r.get("mdice", 0.0) == 0.0 for r in ok_rows):
            print("\n" + _bold("[DIAGNOSTIC] All combos returned mDice=0.0 — probing threshold=0.01 on 1 sample..."))
            first_group = dict(list(image_groups.items())[:1])
            runner.cfg.setdefault("patching", {})
            runner.cfg["patching"]["type"] = "regular_free"
            runner.cfg["patching"].setdefault("regular_free", {})["grid_size"] = regular_grid_sizes[0]
            runner.cfg.setdefault("pipeline", {})["dropout_rounds"] = 1
            runner.cfg.setdefault("postprocess", {})["threshold"] = 0.01
            try:
                inf_diag = InferenceEngine(runner.cfg, wrapper=shared_wrapper)
                diag_rows, _ = _render_eval_rows(first_group, inf_diag, dataset_spec)
                _, mdice_d, cov_d = _compute_combo_metrics(diag_rows, dataset_spec)
                if mdice_d > 0.0:
                    print(
                        f"  → mDice={mdice_d:.4f}  cov={cov_d:.1f}%  "
                        "Attention maps exist — sweep thresholds are too high."
                    )
                else:
                    print(
                        f"  → mDice=0.0  cov={cov_d:.1f}%  "
                        "Attention maps appear empty — verify layer/head selection."
                    )
            except Exception as exc:
                print(f"  → Diagnostic error: {exc}")

        summary = [
            f"total_rows: {len(results)}",
            f"successful_rows: {len(ok_rows)}",
        ]
        if ok_rows:
            best = max(ok_rows, key=lambda r: r.get("mdice", 0.0))
            summary.extend([
                f"best_strategy: {best.get('strategy')}",
                f"best_granularity: {best.get('granularity')}",
                f"best_dropout_rounds: {best.get('dropout_rounds')}",
                f"best_threshold: {best.get('threshold')}",
                f"best_miou: {best.get('miou')}",
                f"best_mdice: {best.get('mdice')}",
            ])
            if best_combo_out is not None:
                best_combo_out.parent.mkdir(parents=True, exist_ok=True)
                best_combo_out.write_text(json.dumps({
                    "strategy":       best.get("strategy"),
                    "granularity":    int(best.get("granularity", 16)),
                    "patches_per_drop": int(best.get("patches_per_drop", 10)),
                    "dropout_rounds": int(best.get("dropout_rounds", 1)),
                    "threshold":      float(best.get("threshold", 0.15)),
                }, indent=2))
                print(f"Best combo written to {best_combo_out}")
        writer.save_text("summary.txt", summary)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run macro PnP-OVSS sweeps from configs/macro_config.yaml.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--macro-config", default="configs/macro_config.yaml")
    p.add_argument("--dataset",     default=None,
                   help="Override macro_sweep.dataset_name (voc, chest_xray).")
    p.add_argument("--transformer", default=None,
                   help="Override macro_sweep.transformer (blip, bridgetower).")
    p.add_argument("--only-phase",
                   choices=["baseline", "head-tuning", "patch-tuning", "final-eval"],
                   default=None,
                   help="Run only one phase. Omit to run all phases in order.")
    p.add_argument("--output-phase-dir", default=None,
                   help="Exact directory for this phase's outputs (no timestamp sub-dir).")
    p.add_argument("--best-pipeline-out", default=None,
                   help="Path to write best {layer,head} JSON (head-tuning phase).")
    p.add_argument("--best-pipeline-in", default=None,
                   help="Path to read best {layer,head} JSON (patch-tuning / final-eval).")
    p.add_argument("--best-combo-out", default=None,
                   help="Path to write best patch combo JSON (patch-tuning phase).")
    p.add_argument("--best-combo-in", default=None,
                   help="Path to read best patch combo JSON (final-eval phase).")
    p.add_argument("--max-eval-samples", type=int, default=None,
                   help="Override max_eval_samples for baseline and final-eval phases.")
    p.add_argument("--max-tune-samples", type=int, default=None,
                   help="Override max_tune_samples for head-tuning and patch-tuning phases.")
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

    if args.dataset:
        macro["dataset_name"] = args.dataset
    if args.transformer:
        macro["transformer"] = args.transformer

    max_eval_samples = int(args.max_eval_samples or macro.get("max_eval_samples", 200))
    max_tune_samples = int(args.max_tune_samples or macro.get("max_tune_samples", 30))

    only = args.only_phase
    output_phase_dir = Path(args.output_phase_dir) if args.output_phase_dir else None

    transfer_cfgs = [
        _resolve_transfer_config(p)
        for p in macro.get("transfer_configs", [])
    ]
    base_tune_cfg = transfer_cfgs[0] if transfer_cfgs else (ROOT / "config_paper_baseline.yaml")

    # ── Resolve starting best_pipeline ───────────────────────────────────────
    # Priority: --best-pipeline-in file > pinned_pipeline in config
    best_pipeline: dict[str, Any] | None = None
    if args.best_pipeline_in:
        bp_path = Path(args.best_pipeline_in)
        if bp_path.exists():
            try:
                best_pipeline = json.loads(bp_path.read_text())
                print(_bold(f"Loaded best pipeline from {bp_path}: L={best_pipeline.get('layer')} H={best_pipeline.get('head')}"))
            except Exception as exc:
                print(f"Warning: could not read {bp_path}: {exc}")
    if best_pipeline is None:
        pinned = macro.get("pinned_pipeline", {})
        if pinned.get("layer") is not None and pinned.get("head") is not None:
            best_pipeline = {"layer": int(pinned["layer"]), "head": int(pinned["head"])}

    # ── Phase 1: Baseline ─────────────────────────────────────────────────────
    if only is None or only == "baseline":
        if transfer_cfgs:
            _log_block("=== START: PHASE 1 — BASELINE TRANSFER SWEEP ===")
            for cfg_path in transfer_cfgs:
                print(_bold(f"Transfer config: {cfg_path.relative_to(ROOT)}"))
                raw_cfg = load_config(cfg_path)
                run_cfg = _apply_macro_overrides(raw_cfg, macro, sample_cap=max_eval_samples)
                # Apply paper_pipeline defaults (L=8, H=10 for BLIP) for Phase 1
                paper = macro.get("paper_pipeline", {})
                if paper.get("layer") is not None and paper.get("head") is not None:
                    run_cfg.setdefault("pipeline", {})["layer"] = int(paper["layer"])
                    run_cfg["pipeline"]["head"] = int(paper["head"])
                    print(_bold(f"Phase 1 paper defaults: L={paper['layer']}, H={paper['head']}"))
                temp_cfg = _write_temp_config(run_cfg, prefix="macro_transfer_")
                slug = f"transfer_{cfg_path.stem}_{run_cfg['dataset']['name']}"
                _run_baseline_transfer(temp_cfg, slug, max_eval_samples, output_dir=output_phase_dir)
            _log_block("=== END: PHASE 1 — BASELINE TRANSFER SWEEP ===")

    # ── Phase 3: Head/Layer Tuning ────────────────────────────────────────────
    head_cfg = macro.get("head_layer_tuning", {})
    should_run_head = (only == "head-tuning") or (only is None and bool(head_cfg.get("run", False)))
    if should_run_head:
        _log_block("=== START: PHASE 3 — HEAD/LAYER TUNING ===")
        metrics = [str(m).strip().lower() for m in head_cfg.get("metrics", ["dice"])]
        tuning_results: dict[str, dict[str, Any]] = {}
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
            result = _run_head_layer_tuning(metric=metric, config_path=temp_cfg)
            if result is not None:
                tuning_results[metric] = result
        for preferred in ("dice", "clip"):
            if preferred in tuning_results:
                best_pipeline = tuning_results[preferred]
                break
        if best_pipeline is None and tuning_results:
            best_pipeline = next(iter(tuning_results.values()))
        if args.best_pipeline_out and best_pipeline is not None:
            out_path = Path(args.best_pipeline_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(best_pipeline, indent=2))
            print(_bold(f"Best pipeline saved → {out_path}  L={best_pipeline.get('layer')} H={best_pipeline.get('head')}"))
        _log_block("=== END: PHASE 3 — HEAD/LAYER TUNING ===")

    # ── Phase 2: Patch/DropOut Tuning ─────────────────────────────────────────
    patch_cfg = macro.get("patch_settings_tuning", {})
    should_run_patch = (only == "patch-tuning") or (only is None and bool(patch_cfg.get("run", False)))
    if should_run_patch:
        _log_block("=== START: PHASE 2 — PATCH/DROPOUT TUNING ===")
        regular_grid_sizes  = [int(v) for v in patch_cfg.get("regular_grid_sizes", [16, 32])]
        superpixel_segments = [int(v) for v in patch_cfg.get("superpixel_segments", [64, 128, 256])]
        patches_per_drop_range = [int(v) for v in patch_cfg.get("patches_per_drop_range", [10])]
        dropout_iterations  = [int(v) for v in patch_cfg.get("dropout_iterations_range", [0, 1, 2, 3])]
        threshold_range     = [float(v) for v in patch_cfg.get("threshold_range", [0.05, 0.1, 0.2])]
        if not dropout_iterations:
            raise ValueError("patch_settings_tuning.dropout_iterations_range cannot be empty.")

        raw_cfg = load_config(base_tune_cfg)
        run_cfg = _apply_macro_overrides(raw_cfg, macro, sample_cap=max_tune_samples)

        if best_pipeline is not None:
            best_L = int(best_pipeline["layer"])
            best_H = int(best_pipeline["head"])
            run_cfg.setdefault("pipeline", {})["layer"] = best_L
            run_cfg["pipeline"]["head"] = best_H
            print(_bold(f"Phase 2 pinned: L={best_L}, H={best_H}"))
        else:
            print(_bold("Phase 2: no L/H from Phase 3 — using config defaults"))

        run_cfg.setdefault("tune_pipeline", {})["max_images"] = max_tune_samples
        temp_cfg = _write_temp_config(run_cfg, prefix="macro_patch_")
        best_combo_out = Path(args.best_combo_out) if args.best_combo_out else None
        _run_patch_tuning(
            config_path=temp_cfg,
            regular_grid_sizes=regular_grid_sizes,
            superpixel_segments=superpixel_segments,
            patches_per_drop_range=patches_per_drop_range,
            dropout_iterations=dropout_iterations,
            threshold_range=threshold_range,
            max_tune_samples=max_tune_samples,
            output_dir=output_phase_dir,
            best_combo_out=best_combo_out,
        )
        _log_block("=== END: PHASE 2 — PATCH/DROPOUT TUNING ===")

    # ── Phase 4: Final Evaluation ─────────────────────────────────────────────
    if only is None or only == "final-eval":
        _log_block("=== START: PHASE 4 — FINAL EVALUATION ===")
        best_combo: dict[str, Any] | None = None
        if args.best_combo_in:
            bc_path = Path(args.best_combo_in)
            if bc_path.exists():
                try:
                    best_combo = json.loads(bc_path.read_text())
                    print(_bold(f"Loaded best combo: {best_combo}"))
                except Exception as exc:
                    print(f"Warning: could not read {bc_path}: {exc}")

        raw_cfg = load_config(base_tune_cfg)
        run_cfg = _apply_macro_overrides(raw_cfg, macro, sample_cap=max_eval_samples)

        if best_pipeline is not None:
            run_cfg.setdefault("pipeline", {})["layer"] = int(best_pipeline["layer"])
            run_cfg["pipeline"]["head"] = int(best_pipeline["head"])
            print(_bold(f"Phase 4 pipeline: L={best_pipeline['layer']}, H={best_pipeline['head']}"))

        if best_combo is not None:
            strategy = best_combo.get("strategy", "regular")
            gran     = int(best_combo.get("granularity", 16))
            ppd      = int(best_combo.get("patches_per_drop", 10))
            dr       = int(best_combo.get("dropout_rounds", 1))
            thr      = float(best_combo.get("threshold", 0.15))
            run_cfg.setdefault("pipeline", {})["dropout_rounds"]  = dr
            run_cfg["pipeline"]["patches_per_drop"] = ppd
            run_cfg.setdefault("postprocess", {})["threshold"] = thr
            if strategy == "regular":
                run_cfg.setdefault("patching", {})["type"] = "regular_free"
                run_cfg["patching"].setdefault("regular_free", {})["grid_size"] = gran
            else:
                run_cfg.setdefault("patching", {})["type"] = "superpixel"
                run_cfg["patching"].setdefault("superpixel", {})["n_segments"] = gran

        temp_cfg = _write_temp_config(run_cfg, prefix="macro_final_eval_")
        slug = f"final_eval_{run_cfg['dataset']['name']}"
        _run_baseline_transfer(temp_cfg, slug, max_eval_samples, output_dir=output_phase_dir)
        _log_block("=== END: PHASE 4 — FINAL EVALUATION ===")

    _log_block("=== MACRO SWEEP COMPLETE ===")


if __name__ == "__main__":
    main()
