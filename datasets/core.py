from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class DatasetSpec(ABC):
    """Dataset metadata contract used by loaders and metrics."""

    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        """Ordered class names used by this dataset."""

    @property
    @abstractmethod
    def background_index(self) -> int:
        """Background class index, or -1 if there is no explicit background class."""

    @property
    @abstractmethod
    def ignore_label(self) -> int:
        """Ignored semantic label value in dense label maps."""

    @property
    @abstractmethod
    def id_to_name(self) -> Dict[int, str]:
        """Class id to class name mapping."""

    @property
    def name_to_id(self) -> Dict[str, int]:
        return {name: idx for idx, name in self.id_to_name.items()}

    @property
    def class_count(self) -> int:
        return len(self.class_names)

    @property
    def query_class_names(self) -> List[str]:
        if 0 <= self.background_index < len(self.class_names):
            return [c for i, c in enumerate(self.class_names) if i != self.background_index]
        return list(self.class_names)
