"""
Config loader for PnP-OVSS.
Merges user YAML onto hardcoded defaults; missing keys are filled in silently.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# ── Defaults (mirrors config.yaml) ─────────────────────────────────────────
DEFAULTS: dict[str, Any] = {
    "model": {
        "name": "Salesforce/blip-itm-base-coco",
        "device": "auto",
        "image_size": 336,
    },
    "patching": {
        "type": "regular",
        "regular": {
            "grid_size": 21,
        },
        "regular_free": {
            "grid_size": 21,
        },
        "superpixel": {
            "n_segments": 100,
            "compactness": 10.0,
            "sigma": 1.0,
        },
    },
    "pipeline": {
        "layer": 7,
        "head": 9,
        "dropout_rounds": 3,
        "patches_per_drop": 10,
        "use_full_ensemble": False,
        "class_filtering": {
            "mode": "gt_present",   # gt_present | clip_topk
            "top_k": 5,
        },
    },
    "postprocess": {
        "threshold": 0.15,
        "gaussian_sigma": 0.05,
        "use_blur": False,
        "use_dense_crf": True,
    },
    "output": {
        "dir": "results",
        "save_iterations": True,
    },
    "dataset": {
        "name": "voc",
        "root": "data/voc",
        "split": "val",
        "download": True,
        "max_images": None,
        "max_samples": None,
        "min_pixels": None,
        "classes": None,
        "seed": 42,
    },
    "clip": {
        "model": "openai/clip-vit-large-patch14-336",
    },
    "tuning": {
        "out_dir": "experiments",
        "use_crf": True,
        "max_images": 50,
        "metric": "clip",
        "mask_threshold": 0.25,
        "blip_batch_size": 4,
    },
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """
    Load config from *path* (YAML) and merge onto defaults.
    If *path* is None or missing, returns defaults unchanged.
    """
    if path is None:
        return _deep_copy(DEFAULTS)

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        user = yaml.safe_load(f) or {}

    return _deep_merge(DEFAULTS, user)


def make_run_slug(class_name: str, cfg: dict[str, Any]) -> str:
    """
    Build a short, human-readable directory name that uniquely identifies a
    config combination.  Two runs with identical parameters produce the same
    slug (→ overwrite).  Any change to a parameter produces a different slug
    (→ new directory).

    Format
    ------
    {class}_L{layer}_H{head}_{patch_tag}_dr{rounds}_pd{per_drop}_sig{sigma}[_crf]

    For superpixels, patch_tag is ``sup{n_segments}``.
    For regular-free patches it is ``regf{grid_size}``.
    For regular patches it is ``reg``.

    Examples
    --------
    elephant_L7_H9_reg_dr3_pd10_sig0.05_crf
    elephant_L9_H7_sup576_dr3_pd50_sig0.05
    giraffe_L5_H3_reg_dr2_pd10_sig0.03_crf
    """
    pipe = cfg["pipeline"]
    pp   = cfg["postprocess"]
    pat  = cfg["patching"]

    if pat["type"] == "superpixel":
        patch_tag = f"sup{pat['superpixel']['n_segments']}"
    elif pat["type"] == "regular_free":
        patch_tag = f"regf{pat.get('regular_free', {}).get('grid_size', 21)}"
    else:
        patch_tag = "reg"

    crf_tag = "_crf" if pp["use_dense_crf"] else ""

    return (
        f"{class_name}"
        f"_L{pipe['layer']}"
        f"_H{pipe['head']}"
        f"_{patch_tag}"
        f"_dr{pipe['dropout_rounds']}"
        f"_pd{pipe['patches_per_drop']}"
        f"_sig{pp['gaussian_sigma']}"
        f"{crf_tag}"
    )


# ── Internal helpers ────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    result = {**base}
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _deep_copy(d: dict) -> dict:
    import copy
    return copy.deepcopy(d)
