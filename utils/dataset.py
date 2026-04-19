"""
Dataset loaders for PnP-OVSS evaluation
=========================================
Two backends are provided:

  PascalVOCDataset
    Wraps torchvision's VOCSegmentation (year=2012, auto-download).
    Yields one sample per (image, class) pair — so a single image that
    contains both "cat" and "dog" produces two samples.
    Ground-truth masks are binary: 1 where the class is present, 0 elsewhere
    (boundary pixels with label 255 are treated as "ignore / unknown").

  FolderDataset
    Simple folder structure you can populate yourself:

        data/folder/
          {class_name}/
            {image_id}.jpg      (or .png)
            {image_id}_mask.png  (binary: 255=foreground, 0=background)

    Mask files are optional — if absent, IoU cannot be computed but the
    pipeline still runs (useful for inference-only mode).

Both return ``EvalSample`` named tuples so the evaluate script is
backend-agnostic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
from PIL import Image


# ── VOC class registry ───────────────────────────────────────────────────────
# Index 0 = background; indices 1–20 = classes; 255 = void/boundary (ignore).
VOC_CLASSES: List[str] = [
    "background",   # 0  — never used as a query class
    "aeroplane",    # 1
    "bicycle",      # 2
    "bird",         # 3
    "boat",         # 4
    "bottle",       # 5
    "bus",          # 6
    "car",          # 7
    "cat",          # 8
    "chair",        # 9
    "cow",          # 10
    "diningtable",  # 11
    "dog",          # 12
    "horse",        # 13
    "motorbike",    # 14
    "person",       # 15
    "pottedplant",  # 16
    "sheep",        # 17
    "sofa",         # 18
    "train",        # 19
    "tvmonitor",    # 20
]

# Map class name → VOC pixel value (1–20)
VOC_CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(VOC_CLASSES)}


@dataclass
class EvalSample:
    """One (image, class) evaluation unit."""
    image_id:   str                     # unique identifier for logging / CSV
    class_name: str
    image:      Image.Image             # RGB PIL image (original resolution)
    gt_mask:    Optional[np.ndarray]    # bool [H, W] — None if unavailable
    gt_label_map: Optional[np.ndarray] = None  # uint8 [H, W] full semantic labels (VOC only)
    source:     str = ""                # "voc" | "folder"


# ── Pascal VOC 2012 ──────────────────────────────────────────────────────────

class PascalVOCDataset:
    """
    Pascal VOC 2012 segmentation dataset.

    Parameters
    ----------
    root        : str | Path — where to store / find the data.
                  torchvision downloads to root/VOCdevkit/VOC2012/ on first run.
    split       : "train" | "val" | "trainval"
    max_images  : int, optional — cap the number of *images* (not samples).
    min_pixels  : int — minimum GT foreground pixels for a class to be included.
                  Filters out near-invisible classes.
    classes     : list[str], optional — restrict to these class names (VOC_CLASSES).
                  None means all 20 classes.
    seed        : int — random seed for shuffling before max_images cap.
    download    : bool — pass True to auto-download (~2 GB).
    """

    def __init__(
        self,
        root: str | Path = "data/voc",
        split: str = "val",
        max_images: Optional[int] = 200,
        min_pixels: int = 200,
        classes: Optional[List[str]] = None,
        seed: int = 42,
        download: bool = True,
    ) -> None:
        try:
            import torchvision.datasets as tvd
        except ImportError:
            raise ImportError("torchvision is required for PascalVOCDataset.")

        self.root       = Path(root)
        self.min_pixels = min_pixels
        self.classes    = set(classes) if classes else set(VOC_CLASSES[1:])

        ds = tvd.VOCSegmentation(
            root=str(self.root),
            year="2012",
            image_set=split,
            download=download,
        )

        # Shuffle then cap
        indices = list(range(len(ds)))
        random.seed(seed)
        random.shuffle(indices)
        if max_images is not None:
            indices = indices[:max_images]

        self._items: list[tuple[Image.Image, np.ndarray, str]] = []
        for i in indices:
            img, mask_pil = ds[i]
            mask_np = np.array(mask_pil, dtype=np.uint8)   # [H, W] values 0–20, 255
            # Build image_id from the underlying file path
            img_path = ds.images[i]
            image_id = Path(img_path).stem
            self._items.append((img.convert("RGB"), mask_np, image_id))

    def __len__(self) -> int:
        return sum(
            1
            for img, mask, _ in self._items
            for cls_name in self._classes_in_mask(mask)
        )

    def __iter__(self) -> Iterator[EvalSample]:
        for image, mask_np, image_id in self._items:
            for cls_name in self._classes_in_mask(mask_np):
                cls_idx  = VOC_CLASS_TO_IDX[cls_name]
                gt_mask  = (mask_np == cls_idx)             # bool [H, W]
                yield EvalSample(
                    image_id   = f"{image_id}_{cls_name}",
                    class_name = cls_name,
                    image      = image,
                    gt_mask    = gt_mask,
                    gt_label_map = mask_np,
                    source     = "voc",
                )

    def _classes_in_mask(self, mask: np.ndarray) -> list[str]:
        """Return class names present in the mask with >= min_pixels pixels."""
        threshold = self.min_pixels if self.min_pixels is not None else 1
        present = []
        for cls_name in self.classes:
            idx = VOC_CLASS_TO_IDX[cls_name]
            if (mask == idx).sum() >= threshold:
                present.append(cls_name)
        return sorted(present)

    def num_images(self) -> int:
        return len(self._items)


# ── Folder dataset ────────────────────────────────────────────────────────────

class FolderDataset:
    """
    Minimal folder-based dataset.

    Expected layout
    ---------------
    root/
      {class_name}/
        {image_id}.jpg   (or .png, .jpeg)
        {image_id}_mask.png   ← optional; 255=foreground, 0=background

    All subdirectories of ``root`` are treated as class names.
    """

    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        root: str | Path,
        max_images: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        items: list[tuple[Path, str]] = []
        for cls_dir in sorted(self.root.iterdir()):
            if not cls_dir.is_dir():
                continue
            class_name = cls_dir.name
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() not in self._IMG_EXTS:
                    continue
                if img_path.stem.endswith("_mask"):
                    continue
                items.append((img_path, class_name))

        random.seed(seed)
        random.shuffle(items)
        if max_images is not None:
            items = items[:max_images]
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[EvalSample]:
        for img_path, class_name in self._items:
            image = Image.open(img_path).convert("RGB")

            # Look for optional mask: {stem}_mask.png
            mask_path = img_path.with_name(img_path.stem + "_mask.png")
            gt_mask: Optional[np.ndarray] = None
            if mask_path.exists():
                m = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
                gt_mask = m > 127                           # bool [H, W]

            yield EvalSample(
                image_id   = f"{class_name}_{img_path.stem}",
                class_name = class_name,
                image      = image,
                gt_mask    = gt_mask,
                gt_label_map = None,
                source     = "folder",
            )


# ── Factory ───────────────────────────────────────────────────────────────────

def build_dataset(cfg: dict) -> "PascalVOCDataset | FolderDataset":
    """
    Instantiate a dataset from the ``dataset`` section of config.yaml.

    cfg keys
    --------
    name        : "voc" | "folder"
    root        : path to data directory
    split       : "val" | "train" | "trainval"  (VOC only)
    max_images  : int or null
    min_pixels  : int  (VOC only)
    classes     : list[str] or null  (VOC only)
    seed        : int
    download    : bool  (VOC only)
    """
    name = cfg.get("name", "voc")

    if name == "voc":
        return PascalVOCDataset(
            root       = cfg.get("root", "data/voc"),
            split      = cfg.get("split", "val"),
            max_images = cfg.get("max_images", None),   # None = no cap
            min_pixels = cfg.get("min_pixels", None),   # None = no filter
            classes    = cfg.get("classes", None),
            seed       = cfg.get("seed", 42),
            download   = cfg.get("download", True),
        )

    if name == "folder":
        return FolderDataset(
            root       = cfg.get("root", "data/folder"),
            max_images = cfg.get("max_images", None),
            seed       = cfg.get("seed", 42),
        )

    raise ValueError(f"Unknown dataset name '{name}'. Choose 'voc' or 'folder'.")
