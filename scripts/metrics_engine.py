from __future__ import annotations

from typing import Optional

import numpy as np

from datasets.core import DatasetSpec


class MetricsEngine:
    """Accumulates per-class and global segmentation metrics."""

    def __init__(self, dataset_spec: DatasetSpec) -> None:
        self.dataset_spec = dataset_spec
        n_classes = len(dataset_spec.class_names)
        self.global_hist = np.zeros((n_classes, n_classes), dtype=np.float64)
        self.class_ious: dict[str, list[float]] = {}
        self.class_dices: dict[str, list[float]] = {}

    @staticmethod
    def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        pred_bin = pred_mask > 0.5
        inter = np.logical_and(pred_bin, gt_mask).sum()
        union = np.logical_or(pred_bin, gt_mask).sum()
        if union == 0:
            return 1.0 if inter == 0 else 0.0
        return float(inter) / float(union)

    @staticmethod
    def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        pred_bin = pred_mask > 0.5
        inter = np.logical_and(pred_bin, gt_mask).sum()
        pred_sum = pred_bin.sum()
        gt_sum = gt_mask.sum()
        denom = pred_sum + gt_sum
        if denom == 0:
            return 1.0
        return float(2 * inter) / float(denom)

    def record_sample(
        self,
        class_name: str,
        pred_mask: np.ndarray,
        gt_mask: Optional[np.ndarray],
    ) -> tuple[float, float, float, float]:
        pred_coverage = float((pred_mask > 0.5).mean() * 100)
        if gt_mask is None:
            return -1.0, -1.0, pred_coverage, -1.0

        gt_coverage = float(gt_mask.mean() * 100)
        iou = self.compute_iou(pred_mask, gt_mask)
        dice = self.compute_dice(pred_mask, gt_mask)
        self.class_ious.setdefault(class_name, []).append(iou)
        self.class_dices.setdefault(class_name, []).append(dice)
        return iou, dice, pred_coverage, gt_coverage

    def update_confusion_from_label_map(
        self,
        gt_label_map: Optional[np.ndarray],
        pred_masks: dict[str, np.ndarray],
    ) -> None:
        if gt_label_map is None:
            return

        gt = gt_label_map.astype(np.int64)
        n_classes = len(self.dataset_spec.class_names)
        bg = self.dataset_spec.background_index

        fill_value = bg if 0 <= bg < n_classes else 0
        pred = np.full_like(gt, fill_value=fill_value, dtype=np.int64)
        name_to_id = self.dataset_spec.name_to_id
        for cls_name, mask in pred_masks.items():
            cls_idx = name_to_id.get(cls_name)
            if cls_idx is None:
                continue
            pred[mask > 0.5] = cls_idx

        ignore = self.dataset_spec.ignore_label
        valid = (gt != ignore) & (gt >= 0) & (gt < n_classes)
        if not np.any(valid):
            return

        hist = np.bincount(
            n_classes * gt[valid] + pred[valid],
            minlength=n_classes ** 2,
        ).reshape(n_classes, n_classes)
        self.global_hist += hist

    def running_means(self) -> tuple[float, float]:
        all_ious = [v for vs in self.class_ious.values() for v in vs]
        all_dices = [v for vs in self.class_dices.values() for v in vs]
        running_miou = float(np.mean(all_ious)) if all_ious else 0.0
        running_mdice = float(np.mean(all_dices)) if all_dices else 0.0
        return running_miou, running_mdice

    def build_summary(self, done: int, ok: int, run_slug: str) -> list[str]:
        lines = [
            "=" * 60,
            "  PnP-OVSS Evaluation Summary",
            f"  Run: {run_slug}",
            "=" * 60,
            f"  Samples processed : {done}   succeeded : {ok}",
            "",
            f"  {'Class':<20}  {'N':>5}  {'mIoU':>8}  {'mDice':>8}",
            "  " + "-" * 49,
        ]

        all_ious: list[float] = []
        all_dices: list[float] = []
        for cls_name in sorted(self.class_ious):
            ious = self.class_ious[cls_name]
            dices = self.class_dices.get(cls_name, [])
            mean_iou = float(np.mean(ious))
            mean_dice = float(np.mean(dices)) if dices else float("nan")
            all_ious.extend(ious)
            all_dices.extend(dices)
            lines.append(
                f"  {cls_name:<20}  {len(ious):>5}  {mean_iou:>8.4f}  {mean_dice:>8.4f}"
            )

        if all_ious:
            lines += [
                "  " + "=" * 49,
                f"  {'Overall Mean IoU':<20}  {len(all_ious):>5}  {float(np.mean(all_ious)):>8.4f}",
                f"  {'Overall Mean Dice':<20}  {len(all_dices):>5}  {float(np.mean(all_dices)):>8.4f}",
                "=" * 60,
            ]
        else:
            lines += ["  No IoU/Dice computed", "=" * 60]

        if self.global_hist.sum() > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                iu = np.diag(self.global_hist) / (
                    self.global_hist.sum(axis=1)
                    + self.global_hist.sum(axis=0)
                    - np.diag(self.global_hist)
                )
                dice = (2 * np.diag(self.global_hist)) / (
                    self.global_hist.sum(axis=1) + self.global_hist.sum(axis=0)
                )

            valid = self.global_hist.sum(axis=1) > 0
            lines += [
                "",
                "Global Metrics (multiclass confusion matrix)",
                f"  Global Mean IoU: {float(np.nanmean(iu[valid])):.4f}",
                f"  Global Mean Dice: {float(np.nanmean(dice[valid])):.4f}",
            ]

            bg = self.dataset_spec.background_index
            if 0 <= bg < len(self.dataset_spec.class_names):
                lines += [
                    f"  Background IoU: {float(iu[bg]):.4f}",
                    f"  Background Dice: {float(dice[bg]):.4f}",
                ]

        return lines
