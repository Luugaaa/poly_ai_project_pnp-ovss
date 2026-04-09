"""
Patch strategies for PnP-OVSS
==============================
Defines how the image is divided into segments for cross-attention
aggregation and Salience DropOut.

Two concrete strategies are provided:

  RegularPatchStrategy
    Treats each of the model's internal P×P grid patches as one segment.
    This is the approach described in the paper.

  SuperpixelPatchStrategy
    Uses SLIC superpixels to define segments.  BLIP still processes the
    image with its own 16×16 grid internally; we aggregate the patch-level
    GradCAM scores by averaging within each superpixel region.  Dropout
    then zeros out entire superpixel blobs, forcing the model to look at
    less discriminative — but spatially coherent — object parts.

Both strategies share the same interface so they can be swapped without
changing any other code.

Interface
---------
  strategy.num_segments          → int
  strategy.aggregate(flat)       → Tensor[N]   (flat: Tensor[P*P])
  strategy.top_k(scores, rem, k) → set[int]
  strategy.mask_segments(pv, dr) → Tensor      (pv: pixel_values)
  strategy.to_spatial(scores)    → ndarray[H,W] at model-input resolution
  strategy.drop_overlay(dropped) → ndarray[H,W] bool
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Set

import numpy as np
import torch


# ── Abstract base ───────────────────────────────────────────────────────────

class PatchStrategy(ABC):

    @property
    @abstractmethod
    def num_segments(self) -> int:
        """Total number of segments (patches or superpixels)."""

    @abstractmethod
    def aggregate(self, flat_scores: torch.Tensor) -> torch.Tensor:
        """
        Convert flat grid-patch GradCAM scores to per-segment scores.

        Parameters
        ----------
        flat_scores : Tensor [P*P]  — one value per BLIP grid patch.

        Returns
        -------
        Tensor [num_segments]
        """

    @abstractmethod
    def top_k(
        self,
        scores: torch.Tensor,
        remaining: Set[int],
        k: int,
    ) -> Set[int]:
        """
        Return the *k* segment indices from *remaining* with highest scores.
        Handles the edge case |remaining| < k.
        """

    @abstractmethod
    def mask_segments(
        self,
        pixel_values: torch.Tensor,
        dropped: Set[int],
    ) -> torch.Tensor:
        """
        Return a copy of *pixel_values* with the pixel regions that belong
        to *dropped* segments set to zero.

        Parameters
        ----------
        pixel_values : Tensor [1, 3, H, W]
        dropped      : set of segment indices (flat int for regular, int for superpixel)
        """

    @abstractmethod
    def to_spatial(self, scores: torch.Tensor) -> np.ndarray:
        """
        Map segment scores to a 2D spatial float32 array at model-input
        resolution (image_size × image_size).  Values are NOT normalised.

        Returns
        -------
        ndarray [img_size, img_size]  float32
        """

    @abstractmethod
    def drop_overlay(self, dropped: Set[int]) -> np.ndarray:
        """
        Boolean mask of dropped regions at model-input resolution.

        Returns
        -------
        ndarray [img_size, img_size]  bool
        """

    # ── Shared helper ───────────────────────────────────────────────────────

    @staticmethod
    def _top_k_from_list(
        scores: torch.Tensor,
        segment_list: list[int],
        k: int,
    ) -> Set[int]:
        if not segment_list:
            return set()
        idxs = torch.tensor(segment_list, dtype=torch.long, device=scores.device)
        vals = scores[idxs]
        k = min(k, len(segment_list))
        top = torch.topk(vals, k).indices.tolist()
        return {segment_list[i] for i in top}


# ── Regular grid patches ────────────────────────────────────────────────────

class RegularPatchStrategy(PatchStrategy):
    """
    Each of BLIP's internal P×P grid patches is treated as one segment.
    Segment index = row * P + col  (flat, row-major).
    """

    def __init__(self, num_patches_per_side: int, patch_size: int) -> None:
        self._P  = num_patches_per_side
        self._ps = patch_size
        self._img_size = num_patches_per_side * patch_size

    @property
    def num_segments(self) -> int:
        return self._P * self._P

    def aggregate(self, flat_scores: torch.Tensor) -> torch.Tensor:
        # Identity: each patch is already its own segment.
        return flat_scores

    def top_k(self, scores: torch.Tensor, remaining: Set[int], k: int) -> Set[int]:
        return self._top_k_from_list(scores, list(remaining), k)

    def mask_segments(
        self,
        pixel_values: torch.Tensor,
        dropped: Set[int],
    ) -> torch.Tensor:
        if not dropped:
            return pixel_values
        P, ps = self._P, self._ps
        masked = pixel_values.clone()
        for flat_idx in dropped:
            r = flat_idx // P
            c = flat_idx % P
            masked[:, :, r * ps : (r + 1) * ps, c * ps : (c + 1) * ps] = 0.0
        return masked

    def to_spatial(self, scores: torch.Tensor) -> np.ndarray:
        """[P*P] → [P, P] float32"""
        return scores.cpu().float().numpy().reshape(self._P, self._P)

    def drop_overlay(self, dropped: Set[int]) -> np.ndarray:
        """[P, P] bool"""
        P = self._P
        overlay = np.zeros((P, P), dtype=bool)
        for flat_idx in dropped:
            overlay[flat_idx // P, flat_idx % P] = True
        return overlay


# ── SLIC superpixels ────────────────────────────────────────────────────────

class SuperpixelPatchStrategy(PatchStrategy):
    """
    SLIC superpixels over the model-input image define the segments.

    BLIP still tokenises the image using its internal 16×16 patch grid.
    Each superpixel's GradCAM score is the mean of the scores of all
    overlapping grid patches weighted by overlap area.

    Parameters
    ----------
    image          : PIL.Image.Image — original image (any size).
    model_img_size : int             — model input resolution (e.g. 384).
    model_patch_sz : int             — model patch size (e.g. 16).
    n_segments     : int             — target number of superpixels.
    compactness    : float           — SLIC compactness.
    sigma          : float           — SLIC pre-smoothing Gaussian sigma.
    device         : torch.device   — for output tensors.
    """

    def __init__(
        self,
        image,                          # PIL.Image.Image
        model_img_size: int,
        model_patch_sz: int,
        n_segments: int = 100,
        compactness: float = 10.0,
        sigma: float = 1.0,
        device: torch.device | None = None,
    ) -> None:
        try:
            from skimage.segmentation import slic as _slic
        except ImportError:
            raise ImportError(
                "scikit-image is required for SuperpixelPatchStrategy.\n"
                "  Install: pip install scikit-image"
            )

        from PIL import Image as PILImage

        self._img_size = model_img_size
        self._ps       = model_patch_sz
        self._P        = model_img_size // model_patch_sz
        self._device   = device

        # Resize original image to model input size for SLIC
        img_resized = image.resize((model_img_size, model_img_size), PILImage.LANCZOS)
        img_np = np.array(img_resized, dtype=np.uint8)  # [H, W, 3]

        # Run SLIC
        labels = _slic(
            img_np,
            n_segments=n_segments,
            compactness=compactness,
            sigma=sigma,
            start_label=0,
            channel_axis=-1,
        ).astype(np.int32)              # [H, W]  values 0 … K-1

        self._labels = labels
        self._K = int(labels.max()) + 1  # actual number of superpixels

        # Precompute: for each superpixel, the flat patch indices it overlaps.
        P, ps = self._P, self._ps
        # pixel (r,c) → patch flat index = (r//ps)*P + (c//ps)
        row_idx, col_idx = np.mgrid[0:model_img_size, 0:model_img_size]
        patch_flat = (row_idx // ps) * P + (col_idx // ps)  # [H, W]

        self._sp_patches: list[np.ndarray] = []
        for sp in range(self._K):
            mask = labels == sp
            self._sp_patches.append(patch_flat[mask])   # flat patch indices for sp

    @property
    def num_segments(self) -> int:
        return self._K

    def aggregate(self, flat_scores: torch.Tensor) -> torch.Tensor:
        """[P*P] → [K] by mean of overlapping patch scores."""
        scores_np = flat_scores.cpu().float().numpy()
        result = np.zeros(self._K, dtype=np.float32)
        for sp in range(self._K):
            idxs = self._sp_patches[sp]
            if len(idxs) > 0:
                result[sp] = scores_np[idxs].mean()
        dev = flat_scores.device if self._device is None else self._device
        return torch.tensor(result, device=dev)

    def top_k(self, scores: torch.Tensor, remaining: Set[int], k: int) -> Set[int]:
        return self._top_k_from_list(scores, list(remaining), k)

    def mask_segments(
        self,
        pixel_values: torch.Tensor,
        dropped: Set[int],
    ) -> torch.Tensor:
        if not dropped:
            return pixel_values
        masked = pixel_values.clone()
        for sp in dropped:
            sp_mask = torch.from_numpy(self._labels == sp).to(pixel_values.device)
            # sp_mask: [H, W] bool — broadcast over channel dim
            masked[0, :, sp_mask] = 0.0
        return masked

    def to_spatial(self, scores: torch.Tensor) -> np.ndarray:
        """[K] → [img_size, img_size] float32"""
        scores_np = scores.cpu().float().numpy()
        spatial = np.zeros((self._img_size, self._img_size), dtype=np.float32)
        for sp in range(self._K):
            spatial[self._labels == sp] = scores_np[sp]
        return spatial

    def drop_overlay(self, dropped: Set[int]) -> np.ndarray:
        """[img_size, img_size] bool"""
        overlay = np.zeros((self._img_size, self._img_size), dtype=bool)
        for sp in dropped:
            overlay[self._labels == sp] = True
        return overlay

    def get_labels(self) -> np.ndarray:
        """Return the [H, W] int32 superpixel label map (for visualisation)."""
        return self._labels


# ── Factory ─────────────────────────────────────────────────────────────────

def build_strategy(cfg: dict, wrapper, image=None) -> PatchStrategy:
    """
    Instantiate the correct strategy from a config dict.

    Parameters
    ----------
    cfg     : dict  — the 'patching' section of config.yaml.
    wrapper : BLIPWrapper
    image   : PIL.Image.Image — required when cfg['type'] == 'superpixel'.
    """
    kind = cfg.get("type", "regular")

    if kind == "regular":
        return RegularPatchStrategy(
            num_patches_per_side=wrapper.num_patches_per_side,
            patch_size=wrapper.patch_size,
        )

    if kind == "superpixel":
        if image is None:
            raise ValueError("image must be provided for SuperpixelPatchStrategy")
        sp_cfg = cfg.get("superpixel", {})
        return SuperpixelPatchStrategy(
            image=image,
            model_img_size=wrapper.image_size,
            model_patch_sz=wrapper.patch_size,
            n_segments=sp_cfg.get("n_segments", 100),
            compactness=sp_cfg.get("compactness", 10.0),
            sigma=sp_cfg.get("sigma", 1.0),
            device=wrapper.device,
        )

    raise ValueError(f"Unknown patching type '{kind}'. Choose 'regular' or 'superpixel'.")
