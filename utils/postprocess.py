from __future__ import annotations
import torch.nn.functional as F
import torch

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


import warnings

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter


def postprocess(
    spatial_map: np.ndarray | torch.Tensor,
    original_image: Image.Image,
    gaussian_sigma: float = 0.05,
    use_blur: bool = False,
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

    blurred = sal
    if use_blur:
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
    gt_mask: np.ndarray | None = None,
) -> None:
    """
    Save a 3-panel visualisation: original | salience heatmap | overlay.
    Accepts masks at any resolution — upsamples to image size when needed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    W_img, H_img = original_image.size
    if mask.shape != (H_img, W_img):
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        mask = np.array(
            mask_pil.resize((W_img, H_img), Image.BILINEAR), dtype=np.float32
        ) / 255.0

    n_panels = 4 if gt_mask is not None else 3
    n_panels = 4 if gt_mask is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4))

    axes[0].imshow(original_image)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Prediction Mask", fontsize=11)
    axes[1].axis("off")

    grey     = np.array(original_image.convert("L"), dtype=np.float32) / 255.0
    grey_rgb = np.stack([grey, grey, grey], axis=-1)
    heat_rgb = cm.jet(mask)[..., :3]
    overlay  = np.clip(0.5 * grey_rgb + 0.5 * heat_rgb, 0.0, 1.0)
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=11)
    axes[2].axis("off")
    
    if gt_mask is not None:
        axes[3].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        axes[3].set_title("Ground Truth", fontsize=11)
        axes[3].axis("off")
    
    if gt_mask is not None:
        axes[3].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        axes[3].set_title("Ground Truth", fontsize=11)
        axes[3].axis("off")

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


def postprocess_multiclass(
    spatial_maps: dict[str, np.ndarray | torch.Tensor],
    original_image: Image.Image,
    threshold: float = 0.15,
    gaussian_sigma: float = 0.05,
    use_blur: bool = False,
    use_dense_crf: bool = True,
) -> dict[str, np.ndarray]:
    """
    Processes all classes simultaneously, matching the paper's logic of adding a background
    channel, doing a joint softmax/CRF, and extracting the final argmax.
    """
    H, W = original_image.size[1], original_image.size[0]
    class_names = list(spatial_maps.keys())
    num_classes = len(class_names)
    
    if num_classes == 0:
        return {}

    # 1. Normalise, apply threshold, and resize each map.
    H_feat, W_feat = None, None
    processed_maps = []
    
    for cls in class_names:
        smap = spatial_maps[cls]
        if isinstance(smap, torch.Tensor):
            smap = smap.cpu().float().numpy()
        H_feat, W_feat = smap.shape
        
        raw_smap = smap.astype(np.float32)
        # Min-max normalise feature-level map to build the boolean threshold mask
        norm_smap = _normalize(raw_smap)
        mask = (norm_smap >= threshold).astype(np.float32)
        
        # Mask the *raw* prob map, as done in the paper: Blip_final_pred = pred_map * thresholded_pred_map
        smap = raw_smap * mask
        
        # Resize to original image resolution using bilinear interpolation
        # Using torch for exact match to mode='bilinear', align_corners=True

        smap_t = torch.from_numpy(smap).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
        smap_resized_t = torch.nn.functional.interpolate(smap_t, size=(H, W), mode='bilinear', align_corners=True).squeeze()
        smap_resized = smap_resized_t.numpy()
        
        # Min-max normalise again (Scale_0_1)
        smap_resized = _normalize(smap_resized)
        processed_maps.append(smap_resized)
        
    stack_preds = np.stack(processed_maps, axis=0) # [C, H, W]
    
    # 2. Add Background Channel
    # The paper calculates background from the ensemble max map:
    max_map = np.max(stack_preds, axis=0)
    bg_map = (max_map == 0.0).astype(np.float32)
    # The paper's logic concatenates background as the first channel
    pred_w_bg = np.concatenate([bg_map[np.newaxis, ...], stack_preds], axis=0) # [C+1, H, W]
    
    # 3. Optional Gaussian blur (applied per channel)
    if use_blur:
        sigma_px = gaussian_sigma * max(H, W)
        blurred_preds = []
        for i in range(pred_w_bg.shape[0]):
            b = gaussian_filter(pred_w_bg[i], sigma=sigma_px)
            mn_val = b.min()
            b = b - mn_val
            mx_val = b.max()
            if mx_val > 0:
                b = b / mx_val
            blurred_preds.append(b)
        pred_w_bg = np.stack(blurred_preds, axis=0) # [C+1, H, W]
    
    # 4. Dense CRF
    if use_dense_crf:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
        
        # Softmax over the channels because we have raw [0,1] features per channel
        # The paper applies softmax before unary
        output_logits = torch.from_numpy(pred_w_bg)
        output_probs = F.softmax(output_logits, dim=0).cpu().numpy()
        
        fg = output_probs.reshape(num_classes + 1, -1)
        unary = unary_from_softmax(fg)
        
        U = np.ascontiguousarray(unary)
        d = dcrf.DenseCRF2D(W, H, num_classes + 1)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=3, compat=7)
        d.addPairwiseBilateral(
            sxy=50, srgb=5,
            rgbim=np.ascontiguousarray(np.array(original_image), dtype=np.uint8),
            compat=10,
        )
        
        Q = d.inference(10)
        argmax_idx = np.argmax(Q, axis=0).reshape(H, W)
    else:
        argmax_idx = np.argmax(pred_w_bg, axis=0)
        
    # 5. Extract boolean masks for each class
    final_masks = {}
    for i, cls in enumerate(class_names):
        # i + 1 because index 0 is background
        final_masks[cls] = (argmax_idx == (i + 1)).astype(np.float32)
        
    return final_masks
