"""
PnP-OVSS — main inference entry point
======================================
Usage
-----
    python3 main.py                                  # uses config.yaml defaults
    python3 main.py --config config.yaml             # explicit config
    python3 main.py --layer 5 --head 3               # CLI overrides
    python3 main.py --patching superpixel            # swap to superpixels
    python3 main.py --no_crf --no_iter_viz           # skip optional steps
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.blip_wrapper import BLIPWrapper, get_device
from utils.config import load_config, make_run_slug
from utils.data_loader import (
    format_prompt,
    get_class_from_path,
    get_class_token_indices,
    load_image,
)
from core.patch_strategy import build_strategy
from core.salience_dropout import salience_dropout
from utils.postprocess import postprocess, save_mask_overlay


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PnP-OVSS: training-free open-vocabulary semantic segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",    default="config.yaml",
                   help="Path to YAML config file.")
    p.add_argument("--image",     default=None,
                   help="Input image path (overrides config).  "
                        "Class is inferred from parent directory name.")
    p.add_argument("--layer",     type=int, default=None,
                   help="Cross-attention layer index (0-based).")
    p.add_argument("--head",      type=int, default=None,
                   help="Attention head index (0-based).")
    p.add_argument("--patching",  choices=["regular", "regular_free", "superpixel"], default=None,
                   help="Patching strategy.")
    p.add_argument("--threshold", type=float, default=None,
                   help="Salience binarisation threshold T.")
    p.add_argument("--sigma",     type=float, default=None,
                   help="Gaussian blur σ as fraction of spatial map side.")
    p.add_argument("--no_crf",    action="store_true",
                   help="Skip Dense CRF post-processing.")
    p.add_argument("--no_iter_viz", action="store_true",
                   help="Skip per-iteration visualisation.")
    p.add_argument("--out_dir",   default=None,
                   help="Output directory (overrides config).")
    return p.parse_args()


# ── Env check ────────────────────────────────────────────────────────────────

def validate_environment() -> None:
    print("=" * 62)
    print("  PnP-OVSS — Environment Check")
    print("=" * 62)
    print(f"  PyTorch    : {torch.__version__}")
    mps  = torch.backends.mps.is_available()
    cuda = torch.cuda.is_available()
    print(f"  MPS        : {mps}")
    print(f"  CUDA       : {cuda}")
    print(f"  Device     : {'mps' if mps else 'cuda' if cuda else 'cpu'}")

    def _chk(pkg, imp=None):
        try:   __import__(imp or pkg); print(f"  {pkg:<16}: OK")
        except ImportError:            print(f"  {pkg:<16}: MISSING")

    _chk("transformers"); _chk("scipy"); _chk("matplotlib")
    _chk("PIL", "PIL");   _chk("skimage", "skimage"); _chk("pydensecrf")
    print("=" * 62)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Load + apply CLI overrides to config ──────────────────────────────
    cfg = load_config(args.config)

    if args.layer     is not None: cfg["pipeline"]["layer"]           = args.layer
    if args.head      is not None: cfg["pipeline"]["head"]            = args.head
    if args.patching  is not None: cfg["patching"]["type"]            = args.patching
    if args.threshold is not None: cfg["postprocess"]["threshold"]    = args.threshold
    if args.sigma     is not None: cfg["postprocess"]["gaussian_sigma"] = args.sigma
    if args.no_crf:                cfg["postprocess"]["use_dense_crf"] = False
    if args.no_iter_viz:           cfg["output"]["save_iterations"]   = False
    if args.out_dir   is not None: cfg["output"]["dir"]               = args.out_dir

    image_rel  = args.image or "images/elephant/elephant.png"
    image_path = ROOT / image_rel
    # Slug is built after class_name is known (below); placeholder here.
    _base_out  = ROOT / cfg["output"]["dir"]

    validate_environment()
    print()

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}\n"
            "Place your image at  images/<class_name>/<image>.png"
        )

    # ── Load model ────────────────────────────────────────────────────────
    model_cfg = cfg["model"]
    device = (
        None if model_cfg["device"] == "auto"
        else torch.device(model_cfg["device"])
    )
    wrapper = BLIPWrapper(model_name=model_cfg["name"], device=device)

    P = wrapper.num_patches_per_side
    print(
        f"\nModel:  {wrapper.num_text_layers} layers × {wrapper.num_heads} heads | "
        f"{wrapper.image_size}px | {wrapper.patch_size}px patch | {P}×{P} grid"
    )

    # ── Validate hyperparameters ──────────────────────────────────────────
    layer = cfg["pipeline"]["layer"]
    head  = cfg["pipeline"]["head"]
    if not (0 <= layer < wrapper.num_text_layers):
        raise ValueError(f"layer {layer} out of range [0, {wrapper.num_text_layers-1}]")
    if not (0 <= head < wrapper.num_heads):
        raise ValueError(f"head {head} out of range [0, {wrapper.num_heads-1}]")

    # ── Load data ─────────────────────────────────────────────────────────
    image      = load_image(image_path)
    class_name = get_class_from_path(image_path)
    prompt     = format_prompt(class_name)

    # Build slug now that class_name is known; create run-specific directory
    slug    = make_run_slug(class_name, cfg)
    out_dir = _base_out / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nImage   : {image_path.relative_to(ROOT)}")
    print(f"Class   : '{class_name}'")
    print(f"Prompt  : '{prompt}'")
    print(f"Patching: {cfg['patching']['type']}")
    print(f"Layer   : {layer}   Head : {head}")
    print(f"Run dir : results/{slug}/")

    inputs = wrapper.preprocess(image, prompt)
    pixel_values   = inputs["pixel_values"]
    input_ids      = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    class_token_indices = get_class_token_indices(wrapper.processor, prompt, class_name)
    print(f"Class token indices: {class_token_indices}")

    # ── Build patch strategy ──────────────────────────────────────────────
    strategy = build_strategy(cfg["patching"], wrapper, image)
    print(f"Segments: {strategy.num_segments}")

    # ── Iteration viz setup ───────────────────────────────────────────────
    save_iters = cfg["output"]["save_iterations"]
    viz_dir    = out_dir / "iterations" if save_iters else None
    if viz_dir:
        viz_dir.mkdir(parents=True, exist_ok=True)

    # ── Salience DropOut ──────────────────────────────────────────────────
    pipe = cfg["pipeline"]
    print(f"\n{'─'*62}")
    print(f"  Salience DropOut  (layer={layer}, head={head}, "
          f"rounds={pipe['dropout_rounds']}, drop={pipe['patches_per_drop']})")
    print(f"{'─'*62}")

    accumulated = salience_dropout(
        wrapper=wrapper,
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        layer_idx=layer,
        head_idx=head,
        class_token_indices=class_token_indices,
        strategy=strategy,
        dropout_rounds=pipe["dropout_rounds"],
        patches_per_drop=pipe["patches_per_drop"],
        verbose=True,
        viz_dir=viz_dir,
        original_image=image if save_iters else None,
    )  # [N]

    print(f"\nAccumulated salience — max: {accumulated.max():.2e}  "
          f"mean: {accumulated.mean():.2e}")

    # ── Spatial conversion + post-processing ─────────────────────────────
    pp   = cfg["postprocess"]
    spatial_map = strategy.to_spatial(accumulated)  # [H', W'] numpy

    print("\nPost-processing …")
    mask = postprocess(
        spatial_map=spatial_map,
        original_image=image,
        gaussian_sigma=pp["gaussian_sigma"],
        use_dense_crf=pp["use_dense_crf"],
    )
    print(f"  shape {mask.shape}   range [{mask.min():.3f}, {mask.max():.3f}]")

    # ── Save ─────────────────────────────────────────────────────────────
    overlay_path  = out_dir / "overlay.png"
    salience_path = out_dir / "salience.npy"

    save_mask_overlay(mask, image, str(overlay_path))
    np.save(str(salience_path), accumulated.cpu().numpy())

    print(f"\nSaved to  results/{slug}/")
    print(f"  overlay.png")
    print(f"  salience.npy")
    if viz_dir:
        n_iter = len(list(viz_dir.glob("pass_*.png")))
        print(f"  iterations/pass_01.png … pass_{n_iter:02d}.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
