"""
Quick smoke-test for the patch tuning pipeline.

Loads the model once, runs exactly one regular and one superpixel combo
at L=9 H=3 on 3 images, and reports mDice / coverage.
Run from repo root:
  python scripts/smoke_patch_tuning.py
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.experiment_runtime import InferenceEngine
from scripts.metrics_engine import MetricsEngine
from utils.config import load_config
from utils.dataset import build_dataset
from datasets.specs import resolve_dataset_spec

LAYER = 9
HEAD  = 3
MAX_IMAGES = 3
THRESHOLD  = 0.10

BASE_CFG_PATH = ROOT / "configs" / "config_paper_baseline.yaml"


def _group(dataset):
    groups = defaultdict(list)
    for s in dataset:
        groups[str(s["image_uid"])].append(s)
    return dict(groups)


def _run_combo(cfg: dict, image_groups: dict, dataset_spec, label: str) -> None:
    t0 = time.perf_counter()
    inf = InferenceEngine(cfg, wrapper=None)  # loads model fresh once here
    shared_wrapper = inf.wrapper

    rows = []
    for uid, samples in image_groups.items():
        image = samples[0]["image"]
        classes = sorted({str(s["class_name"]) for s in samples})
        masks, _ = inf.infer_multiclass_masks(image, classes,
                                              full_ensemble_classes=dataset_spec.query_class_names)
        for s in samples:
            cn = str(s["class_name"])
            pred = masks.get(cn)
            gt   = s.get("gt_mask")
            import numpy as np
            if pred is None:
                pred = np.zeros_like(gt, dtype=float) if gt is not None else np.zeros((image.height, image.width))
            rows.append((cn, pred, gt))

    engine = MetricsEngine(dataset_spec)
    for cn, pred, gt in rows:
        engine.record_sample(cn, pred, gt)

    all_ious   = [v for vs in engine.class_ious.values()  for v in vs]
    all_dices  = [v for vs in engine.class_dices.values() for v in vs]
    import numpy as np
    miou  = float(np.mean(all_ious))  if all_ious  else 0.0
    mdice = float(np.mean(all_dices)) if all_dices else 0.0
    cov   = float(np.mean([(pred > 0.5).mean() * 100 for _, pred, _ in rows]))
    elapsed = time.perf_counter() - t0
    print(f"  {label:<40} mIoU={miou:.4f}  mDice={mdice:.4f}  cov={cov:.1f}%  ({elapsed:.1f}s)")
    return shared_wrapper


def main() -> None:
    print(f"\n{'='*60}")
    print(f"  SMOKE TEST  L={LAYER} H={HEAD}  images={MAX_IMAGES}")
    print(f"{'='*60}")

    # ── Load base config and override key fields ──────────────────────────
    cfg = load_config(str(BASE_CFG_PATH))
    cfg["dataset"]["name"]       = "chest_xray"
    cfg["dataset"]["root"]       = "data/chest_xray"
    cfg["dataset"]["max_images"] = MAX_IMAGES
    cfg["dataset"]["max_samples"] = MAX_IMAGES
    cfg["model"]["image_size"]   = 336
    cfg["pipeline"]["layer"]     = LAYER
    cfg["pipeline"]["head"]      = HEAD
    cfg["pipeline"]["dropout_rounds"]  = 1
    cfg["pipeline"]["patches_per_drop"] = 10
    cfg["postprocess"]["threshold"]    = THRESHOLD
    cfg["postprocess"]["use_dense_crf"] = False  # skip CRF for speed

    dataset      = build_dataset(cfg["dataset"])
    dataset_spec = getattr(dataset, "dataset_spec", None) or resolve_dataset_spec("chest_xray")
    image_groups = _group(dataset)
    image_groups = dict(list(image_groups.items())[:MAX_IMAGES])
    print(f"  Loaded {len(image_groups)} images from chest_xray\n")

    # ── Combo 1: regular_free 16×16 ──────────────────────────────────────
    import copy
    cfg1 = copy.deepcopy(cfg)
    cfg1["patching"]["type"] = "regular_free"
    cfg1["patching"].setdefault("regular_free", {})["grid_size"] = 16
    _run_combo(cfg1, image_groups, dataset_spec, "regular_free  g=16  dr=1  thr=0.10")

    # ── Combo 2: superpixel 64 segments ──────────────────────────────────
    cfg2 = copy.deepcopy(cfg)
    cfg2["patching"]["type"] = "superpixel"
    cfg2["patching"].setdefault("superpixel", {})["n_segments"] = 64
    _run_combo(cfg2, image_groups, dataset_spec, "superpixel    n=64  dr=1  thr=0.10")

    print(f"\n{'='*60}")
    print("  SMOKE TEST COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
