"""
Dataset loaders for PnP-OVSS evaluation.

All loaders emit dictionary samples with explicit keys:
- image_uid: stable per-image ID (no class suffix)
- class_name: semantic class
- image_id: per-sample ID for CSV/logging
- image, gt_mask, gt_label_map, source
"""

from __future__ import annotations

import random
from itertools import islice
from pathlib import Path
from typing import Iterator, List, Optional, TypedDict

import numpy as np
from PIL import Image

from datasets.core import DatasetSpec
from datasets.specs import (
    ChestXrayDatasetSpec,
    VOC_CLASS_NAMES,
    VOCDatasetSpec,
)


INPUT_SIZE = 336
VOC_CLASSES: List[str] = VOC_CLASS_NAMES
VOC_CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(VOC_CLASSES)}


class EvalSample(TypedDict):
    image_uid: str
    image_id: str
    class_name: str
    image: Image.Image
    gt_mask: Optional[np.ndarray]
    gt_label_map: Optional[np.ndarray]
    source: str


def _resize_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)


def _resize_mask_nearest(mask: np.ndarray) -> np.ndarray:
    m = Image.fromarray(mask.astype(np.uint8), mode="L")
    return np.array(m.resize((INPUT_SIZE, INPUT_SIZE), Image.NEAREST), dtype=np.uint8)


class PascalVOCDataset:
    """Pascal VOC 2012 segmentation dataset."""

    def __init__(
        self,
        root: str | Path = "data/voc",
        split: str = "val",
        max_images: Optional[int] = 200,
        min_pixels: Optional[int] = 200,
        classes: Optional[List[str]] = None,
        seed: int = 42,
        download: bool = True,
        max_samples: Optional[int] = None,
    ) -> None:
        try:
            import torchvision.datasets as tvd
        except ImportError:
            raise ImportError("torchvision is required for PascalVOCDataset.")

        self.dataset_spec: DatasetSpec = VOCDatasetSpec()
        self.root = Path(root)
        self.min_pixels = min_pixels
        self.max_samples = max_samples
        default_classes = self.dataset_spec.query_class_names
        self.classes = set(classes) if classes else set(default_classes)

        ds = tvd.VOCSegmentation(
            root=str(self.root),
            year="2012",
            image_set=split,
            download=download,
        )

        indices = list(range(len(ds)))
        random.seed(seed)
        random.shuffle(indices)
        if max_images is not None:
            indices = indices[:max_images]

        self._items: list[tuple[Image.Image, np.ndarray, str]] = []
        for i in indices:
            img, mask_pil = ds[i]
            mask_np = np.array(mask_pil, dtype=np.uint8)
            image_uid = Path(ds.images[i]).stem
            self._items.append((img.convert("RGB"), mask_np, image_uid))

    def __len__(self) -> int:
        total = sum(
            1
            for _, mask, _ in self._items
            for _cls_name in self._classes_in_mask(mask)
        )
        if self.max_samples is not None:
            return min(total, self.max_samples)
        return total

    def __iter__(self) -> Iterator[EvalSample]:
        generator = self._iter_impl()
        if self.max_samples is not None:
            yield from islice(generator, self.max_samples)
        else:
            yield from generator

    def _iter_impl(self) -> Iterator[EvalSample]:
        for image, mask_np, image_uid in self._items:
            resized_image = _resize_rgb(image)
            resized_label_map = _resize_mask_nearest(mask_np)
            for cls_name in self._classes_in_mask(mask_np):
                cls_idx = VOC_CLASS_TO_IDX[cls_name]
                gt_mask = resized_label_map == cls_idx
                yield EvalSample(
                    image_uid=image_uid,
                    image_id=f"{image_uid}_{cls_name}",
                    class_name=cls_name,
                    image=resized_image,
                    gt_mask=gt_mask,
                    gt_label_map=resized_label_map,
                    source="voc",
                )

    def _classes_in_mask(self, mask: np.ndarray) -> list[str]:
        threshold = self.min_pixels if self.min_pixels is not None else 1
        present = []
        for cls_name in self.classes:
            idx = VOC_CLASS_TO_IDX[cls_name]
            if (mask == idx).sum() >= threshold:
                present.append(cls_name)
        return sorted(present)

    def num_images(self) -> int:
        return len(self._items)


class _FolderDatasetSpec(DatasetSpec):
    def __init__(self, class_names: list[str]) -> None:
        self._class_names = class_names

    @property
    def class_names(self) -> List[str]:
        return list(self._class_names)

    @property
    def background_index(self) -> int:
        return -1

    @property
    def ignore_label(self) -> int:
        return 255

    @property
    def id_to_name(self) -> dict[int, str]:
        return {i: n for i, n in enumerate(self._class_names)}


class FolderDataset:
    """Simple folder-based dataset layout."""

    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        root: str | Path,
        max_images: Optional[int] = None,
        seed: int = 42,
        max_samples: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self.max_samples = max_samples
        items: list[tuple[Path, str]] = []
        class_names: set[str] = set()
        for cls_dir in sorted(self.root.iterdir()):
            if not cls_dir.is_dir():
                continue
            class_name = cls_dir.name
            class_names.add(class_name)
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
        self.dataset_spec: DatasetSpec = _FolderDatasetSpec(sorted(class_names))

    def __len__(self) -> int:
        n = len(self._items)
        if self.max_samples is not None:
            return min(n, self.max_samples)
        return n

    def __iter__(self) -> Iterator[EvalSample]:
        generator = self._iter_impl()
        if self.max_samples is not None:
            yield from islice(generator, self.max_samples)
        else:
            yield from generator

    def _iter_impl(self) -> Iterator[EvalSample]:
        for img_path, class_name in self._items:
            image_uid = img_path.stem
            image = _resize_rgb(Image.open(img_path))
            mask_path = img_path.with_name(img_path.stem + "_mask.png")
            gt_mask: Optional[np.ndarray] = None
            if mask_path.exists():
                m = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
                gt_mask = _resize_mask_nearest(m) > 127

            yield EvalSample(
                image_uid=image_uid,
                image_id=f"{class_name}_{image_uid}",
                class_name=class_name,
                image=image,
                gt_mask=gt_mask,
                gt_label_map=None,
                source="folder",
            )


class ChestXrayDataset:
    """Kaggle chest X-ray dataset with image/mask pairing from CXR_png and masks."""

    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        root: str | Path = "data/chest_xray",
        max_images: Optional[int] = None,
        seed: int = 42,
        max_samples: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self.max_samples = max_samples
        self.dataset_spec: DatasetSpec = ChestXrayDatasetSpec()
        cxr_dir, mask_dir = self._resolve_pair_dirs(self.root)
        self.cxr_dir = cxr_dir
        self.mask_dir = mask_dir

        mask_by_stem = {}
        for p in sorted(mask_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in self._IMG_EXTS:
                mask_by_stem[p.stem] = p

        items: list[tuple[Path, Path]] = []
        for img_path in sorted(cxr_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in self._IMG_EXTS:
                continue
            mask_path = mask_by_stem.get(img_path.stem)
            if mask_path is None:
                continue
            items.append((img_path, mask_path))

        random.seed(seed)
        random.shuffle(items)
        if max_images is not None:
            items = items[:max_images]
        self._items = items

    @staticmethod
    def _resolve_pair_dirs(root: Path) -> tuple[Path, Path]:
        candidates = [
            (root / "CXR_png", root / "masks"),
            (root / "Lung Segmentation" / "CXR_png", root / "Lung Segmentation" / "masks"),
            (root / "data" / "Lung Segmentation" / "CXR_png", root / "data" / "Lung Segmentation" / "masks"),
        ]
        for img_dir, m_dir in candidates:
            if img_dir.exists() and m_dir.exists():
                return img_dir, m_dir

        found_cxr = next((p for p in root.rglob("CXR_png") if p.is_dir()), None)
        found_masks = next((p for p in root.rglob("masks") if p.is_dir()), None)
        if found_cxr is None or found_masks is None:
            raise FileNotFoundError(
                "Could not find paired CXR_png and masks directories under "
                f"{root}"
            )
        return found_cxr, found_masks

    def __len__(self) -> int:
        n = len(self._items)
        if self.max_samples is not None:
            return min(n, self.max_samples)
        return n

    def __iter__(self) -> Iterator[EvalSample]:
        generator = self._iter_impl()
        if self.max_samples is not None:
            yield from islice(generator, self.max_samples)
        else:
            yield from generator

    def _iter_impl(self) -> Iterator[EvalSample]:
        for img_path, mask_path in self._items:
            image_uid = img_path.stem
            image = _resize_rgb(Image.open(img_path))
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
            gt_mask = _resize_mask_nearest(mask) > 127

            yield EvalSample(
                image_uid=image_uid,
                image_id=image_uid,
                class_name="lungs",
                image=image,
                gt_mask=gt_mask,
                gt_label_map=None,
                source="chest_xray",
            )

    def num_images(self) -> int:
        return len(self._items)


def build_dataset(cfg: dict) -> PascalVOCDataset | FolderDataset | ChestXrayDataset:
    """Instantiate a dataset from the config dataset section."""

    name = (cfg.get("name", "voc") or "voc").strip().lower()
    max_samples = cfg.get("max_samples", None)

    if name == "voc":
        return PascalVOCDataset(
            root=cfg.get("root", "data/voc"),
            split=cfg.get("split", "val"),
            max_images=cfg.get("max_images", None),
            min_pixels=cfg.get("min_pixels", None),
            classes=cfg.get("classes", None),
            seed=cfg.get("seed", 42),
            download=cfg.get("download", True),
            max_samples=max_samples,
        )

    if name == "folder":
        return FolderDataset(
            root=cfg.get("root", "data/folder"),
            max_images=cfg.get("max_images", None),
            seed=cfg.get("seed", 42),
            max_samples=max_samples,
        )

    if name in {"chest_xray", "chestxray", "kaggle_chest_xray"}:
        return ChestXrayDataset(
            root=cfg.get("root", "data/chest_xray"),
            max_images=cfg.get("max_images", None),
            seed=cfg.get("seed", 42),
            max_samples=max_samples,
        )

    raise ValueError(
        f"Unknown dataset name '{name}'. Choose 'voc', 'folder', or 'chest_xray'."
    )
