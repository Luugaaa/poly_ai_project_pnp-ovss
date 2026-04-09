"""
Post-processing pipeline (§3.3 of PnP-OVSS)
=============================================
Accepts a 2D float spatial map at any resolution (from either strategy)
and produces a final segmentation mask at the original image resolution.

Steps:
  1. Normalise to [0, 1]
  2. Gaussian blur  (σ as fraction of map's shorter side)
  3. Resize to original image resolution
  4. Dense CRF  (optional)
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter


def postprocess(
    spatial_map: np.ndarray | torch.Tensor,
    original_image: Image.Image,
    gaussian_sigma: float = 0.05,
    use_dense_crf: bool = True,
) -> np.ndarray:
    """
    Parameters
    ----------
    spatial_map    : ndarray or Tensor [H', W'] — salience at any resolution.
                     For regular patches this is typically [P, P] (24×24).
                     For superpixels this is [img_size, img_size] (384×384).
    original_image : PIL.Image.Image
    gaussian_sigma : float — σ as fraction of the map's shorter side.
    use_dense_crf  : bool

    Returns
    -------
    mask : ndarray [H, W] float32 ∈ [0, 1]
    """
    if isinstance(spatial_map, torch.Tensor):
        spatial_map = spatial_map.cpu().float().numpy()

    sal = _normalize(spatial_map.astype(np.float32))

    # Gaussian blur relative to map resolution
    short_side = min(sal.shape)
    sigma_px   = max(gaussian_sigma * short_side, 0.5)
    blurred    = _normalize(gaussian_filter(sal, sigma=sigma_px))

    # Resize to original image resolution
    H, W = original_image.size[1], original_image.size[0]
    blurred_pil     = Image.fromarray((blurred * 255).astype(np.uint8), mode="L")
    blurred_resized = (
        np.array(blurred_pil.resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0
    )

    if use_dense_crf:
        mask = _apply_dense_crf(blurred_resized, np.array(original_image))
    else:
        mask = blurred_resized

    return mask.astype(np.float32)


def save_mask_overlay(
    mask: np.ndarray,
    original_image: Image.Image,
    save_path: str,
) -> None:
    """
    Save a 3-panel visualisation: original | salience heatmap | overlay.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].imshow(original_image)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Salience Map", fontsize=11)
    axes[1].axis("off")

    grey     = np.array(original_image.convert("L"), dtype=np.float32) / 255.0
    grey_rgb = np.stack([grey, grey, grey], axis=-1)
    heat_rgb = cm.jet(mask)[..., :3]
    overlay  = np.clip(0.5 * grey_rgb + 0.5 * heat_rgb, 0.0, 1.0)
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=11)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Private helpers ──────────────────────────────────────────────────────────

def _normalize(arr: np.ndarray) -> np.ndarray:
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def _apply_dense_crf(prob_map: np.ndarray, image_rgb: np.ndarray) -> np.ndarray:
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError:
        warnings.warn(
            "pydensecrf not installed — Dense CRF skipped. "
            "Falling back to Gaussian-blurred map.",
            stacklevel=3,
        )
        return prob_map

    H, W = prob_map.shape
    fg   = np.clip(prob_map.flatten().astype(np.float64), 1e-5, 1 - 1e-5)
    bg   = 1.0 - fg
    probs   = np.stack([bg, fg], axis=0).astype(np.float32)
    unary   = unary_from_softmax(probs)

    d = dcrf.DenseCRF2D(W, H, 2)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(
        sxy=50, srgb=13,
        rgbim=np.ascontiguousarray(image_rgb, dtype=np.uint8),
        compat=10,
    )

    Q       = d.inference(5)
    refined = np.argmax(Q, axis=0).reshape(H, W).astype(np.float32)
    return refined
