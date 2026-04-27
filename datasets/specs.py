from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from datasets.core import DatasetSpec


VOC_CLASS_NAMES: List[str] = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


@dataclass(frozen=True)
class VOCDatasetSpec(DatasetSpec):
    @property
    def class_names(self) -> List[str]:
        return VOC_CLASS_NAMES

    @property
    def background_index(self) -> int:
        return 0

    @property
    def ignore_label(self) -> int:
        return 255

    @property
    def id_to_name(self) -> Dict[int, str]:
        return {i: name for i, name in enumerate(self.class_names)}


@dataclass(frozen=True)
class ChestXrayDatasetSpec(DatasetSpec):
    @property
    def class_names(self) -> List[str]:
        return ["lungs"]

    @property
    def background_index(self) -> int:
        return -1

    @property
    def ignore_label(self) -> int:
        return 255

    @property
    def id_to_name(self) -> Dict[int, str]:
        return {0: "lungs"}


def resolve_dataset_spec(dataset_name: str) -> DatasetSpec:
    name = (dataset_name or "voc").strip().lower()
    if name == "voc":
        return VOCDatasetSpec()
    if name in {"chest_xray", "chestxray", "kaggle_chest_xray"}:
        return ChestXrayDatasetSpec()
    raise ValueError(
        f"No DatasetSpec registered for dataset name '{dataset_name}'. "
        "Supported: voc, chest_xray"
    )
