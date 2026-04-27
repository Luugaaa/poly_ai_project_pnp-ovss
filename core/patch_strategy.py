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

    RegularFreePatchStrategy
        Uses a regular G×G grid built directly in pixel space at model-input
        resolution. Segment boundaries are independent from the ViT token grid.
        GradCAM token scores are mapped to these pixel-space cells via overlap.

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
import torch.nn.functional as F


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
    Each of BLIP's internal P×P grid patches is treated as one segment,
    or optionally grouped into a coarser G×G grid (grid_size < P).

    With ``grid_size=G``, model patches are grouped into a coarser G×G grid.
    If ``G`` does not divide ``P``, coarse cells use uneven patch spans
    (difference at most one patch) so all P×P model patches are still covered.
    ``grid_size=None`` or ``grid_size=P`` gives the native patch-level
    resolution (default behaviour).

    Segment index = row * G + col  (flat, row-major over the coarse grid).
    """

    def __init__(
        self,
        num_patches_per_side: int,
        patch_size: int,
        grid_size: int | None = None,
    ) -> None:
        self._P  = num_patches_per_side
        self._ps = patch_size
        self._img_size = num_patches_per_side * patch_size
        G = grid_size if grid_size is not None else num_patches_per_side
        if G <= 0:
            raise ValueError(f"grid_size must be positive, got {G}")
        if G > num_patches_per_side:
            raise ValueError(
                f"grid_size {G} must be <= num_patches_per_side {num_patches_per_side}"
            )
        self._G     = G
        # Partition patch indices into G bins; supports non-divisible G.
        # Boundaries are in patch coordinates in [0, P].
        self._bounds = [
            (i * num_patches_per_side) // G for i in range(G + 1)
        ]

    @property
    def num_segments(self) -> int:
        return self._G * self._G

    def aggregate(self, flat_scores: torch.Tensor) -> torch.Tensor:
        """[P*P] → [G*G]  (average-pool model patches within each coarse cell)."""
        if self._G == self._P:
            return flat_scores   # native resolution — identity
        arr = flat_scores.reshape(self._P, self._P)
        pooled = []
        for r in range(self._G):
            r0, r1 = self._bounds[r], self._bounds[r + 1]
            for c in range(self._G):
                c0, c1 = self._bounds[c], self._bounds[c + 1]
                pooled.append(arr[r0:r1, c0:c1].mean())
        return torch.stack(pooled).flatten()   # [G*G]

    def top_k(self, scores: torch.Tensor, remaining: Set[int], k: int) -> Set[int]:
        return self._top_k_from_list(scores, list(remaining), k)

    def mask_segments(
        self,
        pixel_values: torch.Tensor,
        dropped: Set[int],
    ) -> torch.Tensor:
        if not dropped:
            return pixel_values
        G, ps = self._G, self._ps
        masked = pixel_values.clone()
        for flat_idx in dropped:
            r = flat_idx // G
            c = flat_idx % G
            r0, r1 = self._bounds[r] * ps, self._bounds[r + 1] * ps
            c0, c1 = self._bounds[c] * ps, self._bounds[c + 1] * ps
            masked[:, :, r0:r1, c0:c1] = 0.0
        return masked

    def to_spatial(self, scores: torch.Tensor) -> np.ndarray:
        """[G*G] → [G, G] float32  (postprocess bilinearly upsamples to image size)."""
        return scores.cpu().float().numpy().reshape(self._G, self._G)

    def drop_overlay(self, dropped: Set[int]) -> np.ndarray:
        """[G, G] bool"""
        G = self._G
        overlay = np.zeros((G, G), dtype=bool)
        for flat_idx in dropped:
            overlay[flat_idx // G, flat_idx % G] = True
        return overlay


class RegularFreePatchStrategy(PatchStrategy):
    """
    Pixel-space regular grid strategy.

    Segments are defined as a regular G×G partition over the model-input
    image (img_size × img_size), independent from the ViT patch grid P×P.
    Token-level GradCAM scores are aggregated per segment by area overlap.

    Segment index = row * G + col  (flat, row-major over pixel-space grid).
    """

    def __init__(
        self,
        model_img_size: int,
        model_patch_sz: int,
        grid_size: int,
        device: torch.device | None = None,
    ) -> None:
        if grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {grid_size}")
        if grid_size > model_img_size:
            raise ValueError(
                f"grid_size {grid_size} must be <= model_img_size {model_img_size}"
            )

        self._img_size = model_img_size
        self._ps = model_patch_sz
        self._P = model_img_size // model_patch_sz
        self._G = grid_size
        self._device = device
        self._bounds = [(i * model_img_size) // grid_size for i in range(grid_size + 1)]

        # Pixel-space label map [H, W] with flat segment ids in [0, G*G-1].
        labels = np.zeros((model_img_size, model_img_size), dtype=np.int32)
        for r in range(grid_size):
            r0, r1 = self._bounds[r], self._bounds[r + 1]
            for c in range(grid_size):
                c0, c1 = self._bounds[c], self._bounds[c + 1]
                labels[r0:r1, c0:c1] = r * grid_size + c
        self._labels = labels

        # Precompute patch-flat indices overlapping each segment.
        row_idx, col_idx = np.mgrid[0:model_img_size, 0:model_img_size]
        patch_flat = (row_idx // model_patch_sz) * self._P + (col_idx // model_patch_sz)
        self._seg_patches: list[np.ndarray] = []
        K = grid_size * grid_size
        for seg in range(K):
            seg_mask = labels == seg
            self._seg_patches.append(patch_flat[seg_mask])

    @property
    def num_segments(self) -> int:
        return self._G * self._G

    def aggregate(self, flat_scores: torch.Tensor) -> torch.Tensor:
        """
        [P*P] → [G*G] via bilinear upsample to img_size then average-pool into grid cells.

        Upsampling the ViT token grid to full pixel resolution before binning
        decouples the evaluation granularity from the ViT's native patch size
        and produces smoother, interpolated salience estimates per segment.
        """
        P = self._P
        # Reshape to [1, 1, P, P] and bilinearly upsample to model-input resolution
        sal_2d = flat_scores.float().reshape(1, 1, P, P)
        sal_full = F.interpolate(
            sal_2d,
            size=(self._img_size, self._img_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze()  # [img_size, img_size]

        # Average-pool upsampled salience into G×G pixel-space grid cells
        G = self._G
        result = torch.zeros(self.num_segments, dtype=torch.float32)
        for r in range(G):
            r0, r1 = self._bounds[r], self._bounds[r + 1]
            for c in range(G):
                c0, c1 = self._bounds[c], self._bounds[c + 1]
                result[r * G + c] = sal_full[r0:r1, c0:c1].mean()

        dev = flat_scores.device if self._device is None else self._device
        return result.to(dev)

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
        for seg in dropped:
            seg_mask = torch.from_numpy(self._labels == seg).to(pixel_values.device)
            masked[0, :, seg_mask] = 0.0
        return masked

    def to_spatial(self, scores: torch.Tensor) -> np.ndarray:
        """[G*G] -> [img_size, img_size] float32"""
        scores_np = scores.cpu().float().numpy()
        spatial = np.zeros((self._img_size, self._img_size), dtype=np.float32)
        for seg in range(self.num_segments):
            spatial[self._labels == seg] = scores_np[seg]
        return spatial

    def drop_overlay(self, dropped: Set[int]) -> np.ndarray:
        """[img_size, img_size] bool"""
        overlay = np.zeros((self._img_size, self._img_size), dtype=bool)
        for seg in dropped:
            overlay[self._labels == seg] = True
        return overlay

    def get_labels(self) -> np.ndarray:
        """Return the [H, W] int32 segment label map."""
        return self._labels


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
        grid_size = cfg.get("regular", {}).get("grid_size", None)
        return RegularPatchStrategy(
            num_patches_per_side=wrapper.num_patches_per_side,
            patch_size=wrapper.patch_size,
            grid_size=grid_size,
        )

    if kind == "regular_free":
        rf_cfg = cfg.get("regular_free", {})
        grid_size = rf_cfg.get("grid_size", wrapper.num_patches_per_side)
        return RegularFreePatchStrategy(
            model_img_size=wrapper.image_size,
            model_patch_sz=wrapper.patch_size,
            grid_size=grid_size,
            device=wrapper.device,
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

    raise ValueError(
        f"Unknown patching type '{kind}'. Choose 'regular', 'regular_free' or 'superpixel'."
    )
