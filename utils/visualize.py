"""
Iteration-level visualisation for Salience DropOut
====================================================
Saves one PNG per dropout pass showing:

  Panel 1 — Input image for this pass (dropped regions shown in red)
  Panel 2 — Salience from this pass only (M̃^(k,t))
  Panel 3 — Accumulated salience so far (Σ_{s≤t} M̃^(k,s))

For regular patches, dropped regions are red rectangles on the patch grid.
For superpixels, dropped regions are the actual superpixel shapes.

Files are written to:
    {viz_dir}/pass_{t+1:02d}.png          (during main.py)
    {viz_dir}/pass_{t+1:02d}_{tag}.png    (during grid search)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from core.patch_strategy import PatchStrategy, RegularPatchStrategy


def save_iteration_viz(
    t: int,
    total_passes: int,
    current_scores: np.ndarray,        # [N] float — this pass's segment scores
    accumulated_scores: np.ndarray,    # [N] float — cumulative scores
    dropped: Set[int],
    strategy: PatchStrategy,
    original_image: Image.Image,
    out_dir: Path,
    tag: str = "",
) -> None:
    """
    Save a 3-panel iteration visualisation PNG.

    Parameters
    ----------
    t                  : int — 0-based pass index.
    total_passes       : int
    current_scores     : ndarray [N]  — GradCAM scores for this pass.
    accumulated_scores : ndarray [N]  — running sum up to and including t.
    dropped            : set[int]     — segment indices zeroed in this pass's input.
    strategy           : PatchStrategy
    original_image     : PIL.Image.Image — for display.
    out_dir            : Path — directory to save the PNG.
    tag                : str — optional suffix for the filename.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"pass_{t+1:02d}" + (f"_{tag}" if tag else "") + ".png"
    out_path = out_dir / fname

    # Convert segment scores to spatial maps
    cur_spatial  = strategy.to_spatial(
        __scores_to_tensor(current_scores, strategy)
    )
    acc_spatial  = strategy.to_spatial(
        __scores_to_tensor(accumulated_scores, strategy)
    )
    drop_mask    = strategy.drop_overlay(dropped)   # bool [H', W']

    # Normalise for display
    cur_display  = _normalize(cur_spatial)
    acc_display  = _normalize(acc_spatial)

    # Build the masked-image panel
    img_arr = np.array(original_image.resize(
        (cur_spatial.shape[1], cur_spatial.shape[0]), Image.BILINEAR
    ), dtype=np.float32) / 255.0                # [H', W', 3]

    masked_img = _overlay_dropped(img_arr, drop_mask, strategy)

    # ── Plot ─────────────────────────────────────────────────────────────
    pass_label = "Initial" if t == 0 else f"Drop {t}"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f"Pass {t+1}/{total_passes}  —  {pass_label}  "
        f"({len(dropped)} segments dropped)",
        fontsize=12,
    )

    axes[0].imshow(masked_img)
    axes[0].set_title("Input (dropped = red)", fontsize=10)
    axes[0].axis("off")
    _add_grid_lines(axes[0], strategy)

    axes[1].imshow(cur_display, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title(f"Salience — pass {t+1}", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(acc_display, cmap="jet", vmin=0, vmax=1)
    axes[2].set_title(f"Accumulated (passes 1–{t+1})", fontsize=10)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_patch_overview(
    image: Image.Image,
    strategy: PatchStrategy,
    out_path: Path,
    title: str = "",
) -> None:
    """
    Save a single fixed visualisation of the patch layout on one image.
    Called once at the start of evaluate / tune runs so the user can see
    exactly what the active patch strategy looks like.

    For RegularPatchStrategy : draws the P×P grid over the model-input
                                resolution image, with patch count in the title.
    For SuperpixelPatchStrategy : draws segment boundaries (skimage find_boundaries)
                                   coloured by segment index, with the actual
                                   segment count in the title.
    """
    from core.patch_strategy import SuperpixelPatchStrategy

    img_size = strategy._img_size   # model input resolution (e.g. 384)
    img_resized = np.array(
        image.resize((img_size, img_size), Image.LANCZOS), dtype=np.uint8
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        title or f"Patch strategy: {type(strategy).__name__}",
        fontsize=13,
    )

    # ── Left: original image at model resolution ──────────────────────────
    axes[0].imshow(img_resized)
    axes[0].set_title(f"Image ({img_size}×{img_size})", fontsize=10)
    axes[0].axis("off")

    # ── Right: patch overlay ──────────────────────────────────────────────
    axes[1].imshow(img_resized)

    if isinstance(strategy, RegularPatchStrategy):
        P  = strategy._P
        ps = strategy._ps
        # Draw grid lines at patch boundaries
        for i in range(1, P):
            axes[1].axhline(i * ps - 0.5, color="yellow", linewidth=0.7, alpha=0.8)
            axes[1].axvline(i * ps - 0.5, color="yellow", linewidth=0.7, alpha=0.8)
        # Label a few patches with their flat index
        for r in range(0, P, max(1, P // 6)):
            for c in range(0, P, max(1, P // 6)):
                cx = c * ps + ps // 2
                cy = r * ps + ps // 2
                axes[1].text(
                    cx, cy, str(r * P + c),
                    color="white", fontsize=5, ha="center", va="center",
                    fontweight="bold",
                )
        axes[1].set_title(
            f"Regular patches — {P}×{P} = {P*P} segments", fontsize=10
        )

    else:
        # Superpixel strategy: draw boundaries using skimage
        try:
            from skimage.segmentation import find_boundaries, mark_boundaries
            labels   = strategy.get_labels()               # [H, W] int32
            boundary = find_boundaries(labels, mode="outer")  # [H, W] bool

            # Colour each superpixel lightly by index
            cmap    = plt.get_cmap("tab20")
            K       = strategy.num_segments
            colored = cmap(labels % 20)[..., :3]           # [H, W, 3] float
            blend   = 0.35 * colored + 0.65 * (img_resized / 255.0)
            blend   = np.clip(blend, 0, 1)

            axes[1].imshow(blend)
            # Draw boundaries in white
            bnd_rgba               = np.zeros((*boundary.shape, 4), dtype=np.float32)
            bnd_rgba[boundary, :3] = 1.0
            bnd_rgba[boundary, 3]  = 0.9
            axes[1].imshow(bnd_rgba)
            axes[1].set_title(
                f"Superpixels (SLIC) — {K} segments", fontsize=10
            )
        except ImportError:
            # skimage not available: fall back to label map
            axes[1].imshow(strategy.get_labels(), cmap="tab20", alpha=0.6)
            axes[1].set_title(f"Superpixels — {strategy.num_segments} segments", fontsize=10)

    axes[1].axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Patch overview → {out_path}")


# ── Private helpers ──────────────────────────────────────────────────────────

def _normalize(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-10:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def _overlay_dropped(
    img: np.ndarray,          # [H, W, 3] float32 ∈ [0,1]
    drop_mask: np.ndarray,    # [H, W] bool
    strategy: PatchStrategy,
) -> np.ndarray:
    """
    Overlay dropped regions in semi-transparent red.
    For regular patches, also draw grid lines.
    """
    result = img.copy()
    if drop_mask.any():
        red_channel = np.zeros_like(img)
        red_channel[..., 0] = 1.0                  # pure red
        alpha = 0.55
        result[drop_mask] = (
            (1 - alpha) * img[drop_mask] + alpha * red_channel[drop_mask]
        )
    return np.clip(result, 0.0, 1.0)


def _add_grid_lines(ax, strategy: PatchStrategy) -> None:
    """Draw patch grid lines for RegularPatchStrategy only."""
    if not isinstance(strategy, RegularPatchStrategy):
        return
    P = strategy._P
    for i in range(1, P):
        ax.axhline(i - 0.5, color="white", linewidth=0.4, alpha=0.5)
        ax.axvline(i - 0.5, color="white", linewidth=0.4, alpha=0.5)


def __scores_to_tensor(scores: np.ndarray, strategy: PatchStrategy):
    """Wrap numpy scores as a CPU tensor for strategy.to_spatial()."""
    import torch
    return torch.tensor(scores, dtype=torch.float32)
