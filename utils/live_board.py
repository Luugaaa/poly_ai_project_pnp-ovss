"""
Progressive evaluation visualisation
======================================
Maintains two files that update after every sample:

  {eval_dir}/progress_board.png
    Rolling grid of the last N results.  Each cell shows:
      top-left  : original image thumbnail
      top-right : predicted mask overlay
      bottom    : class label + IoU score (colour-coded green/yellow/red)
    The grid is re-rendered and overwritten after each sample, so you can
    open it in any image viewer that auto-reloads (e.g. macOS Preview with
    spacebar, or VS Code image preview).

  {eval_dir}/miou_curve.png
    Running mIoU plotted against sample index — lets you see whether
    performance is stable or drifting as more samples are processed.

Both files are written atomically (written to a .tmp then renamed) so
an image viewer never reads a half-written file.

Usage
-----
    board = LiveBoard(eval_dir, grid_cols=5, max_cells=30, thumb_size=160)
    board.update(sample, mask, iou, running_miou, class_ious)   # call after each sample
    board.finalize(class_ious)                                    # call at the end
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.dataset import EvalSample


# ── Colour helpers ────────────────────────────────────────────────────────────

def _iou_colour(iou: float) -> tuple[int, int, int]:
    """Green (high IoU) → yellow → red (low IoU)."""
    if iou >= 0.5:
        return (60, 180, 60)
    if iou >= 0.25:
        return (220, 180, 0)
    return (210, 50, 50)


def _make_cell(
    image: Image.Image,
    mask: np.ndarray,
    class_name: str,
    iou: float,
    thumb_size: int,
) -> Image.Image:
    """
    Build one grid cell: [original | overlay] stacked above a label bar.

    Returns a square PIL image of size (thumb_size*2, thumb_size + label_px).
    """
    label_px = 22
    W = thumb_size * 2
    H = thumb_size + label_px

    cell = Image.new("RGB", (W, H), (30, 30, 30))

    # ── Left: original thumbnail ─────────────────────────────────────────
    orig_thumb = image.convert("RGB").resize(
        (thumb_size, thumb_size), Image.LANCZOS
    )
    cell.paste(orig_thumb, (0, 0))

    # ── Right: overlay (grey + jet heatmap) ─────────────────────────────
    grey    = np.array(image.convert("L").resize((thumb_size, thumb_size), Image.LANCZOS), np.float32) / 255.0
    grey_rgb = np.stack([grey, grey, grey], axis=-1)

    # jet colormap applied to mask resized to thumb
    import matplotlib.cm as cm
    mask_thumb = np.array(
        Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (thumb_size, thumb_size), Image.LANCZOS
        ), dtype=np.float32
    ) / 255.0
    heat_rgb = cm.jet(mask_thumb)[..., :3]
    overlay  = np.clip(0.45 * grey_rgb + 0.55 * heat_rgb, 0, 1)
    overlay_pil = Image.fromarray((overlay * 255).astype(np.uint8))
    cell.paste(overlay_pil, (thumb_size, 0))

    # ── Label bar ────────────────────────────────────────────────────────
    draw  = ImageDraw.Draw(cell)
    colour = _iou_colour(iou)
    draw.rectangle([(0, thumb_size), (W, H)], fill=colour)

    label = f"{class_name}  IoU={iou:.3f}"
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except Exception:
        font = ImageFont.load_default()

    # centre the text
    bbox = draw.textbbox((0, 0), label, font=font)
    tw   = bbox[2] - bbox[0]
    tx   = max(2, (W - tw) // 2)
    draw.text((tx, thumb_size + 3), label, fill=(255, 255, 255), font=font)

    return cell


# ── Main class ────────────────────────────────────────────────────────────────

class LiveBoard:
    """
    Parameters
    ----------
    eval_dir   : Path — evaluation output directory.
    grid_cols  : int  — number of columns in the rolling grid.
    max_cells  : int  — maximum cells kept in the rolling window.
    thumb_size : int  — pixel size of each thumbnail (each cell is 2×thumb wide).
    """

    def __init__(
        self,
        eval_dir: Path,
        grid_cols: int = 5,
        max_cells: int = 30,
        thumb_size: int = 140,
    ) -> None:
        self.eval_dir   = Path(eval_dir)
        self.grid_cols  = grid_cols
        self.max_cells  = max_cells
        self.thumb_size = thumb_size

        self._cells:   List[Image.Image]  = []   # rolling window of cell images
        self._ious:    List[float]        = []   # all IoU values seen
        self._indices: List[int]          = []   # sample indices for mIoU curve
        self._miou_history: List[float]   = []   # running mIoU after each sample

        self._board_path = self.eval_dir / "progress_board.png"
        self._curve_path = self.eval_dir / "miou_curve.png"

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        sample: EvalSample,
        mask: np.ndarray,
        iou: float,
        running_miou: float,
        done: int,
    ) -> None:
        """Call after each successfully processed sample."""
        # Build cell
        cell = _make_cell(
            image      = sample.image,
            mask       = mask,
            class_name = sample.class_name,
            iou        = iou,
            thumb_size = self.thumb_size,
        )
        self._cells.append(cell)
        if len(self._cells) > self.max_cells:
            self._cells.pop(0)

        # Track mIoU curve
        if iou >= 0:
            self._ious.append(iou)
            self._indices.append(done)
            self._miou_history.append(running_miou)

        # Render both files
        self._render_grid()
        self._render_curve()

    def finalize(self, class_ious: Dict[str, List[float]]) -> None:
        """Call at the end of evaluation to write the final summary board."""
        self._render_summary(class_ious)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render_grid(self) -> None:
        """Write the rolling grid PNG atomically."""
        cells  = self._cells
        n      = len(cells)
        cols   = min(self.grid_cols, n)
        rows   = math.ceil(n / cols) if cols > 0 else 1
        W_cell = self.thumb_size * 2
        H_cell = self.thumb_size + 22

        canvas = Image.new("RGB", (cols * W_cell, rows * H_cell), (15, 15, 15))
        for i, cell in enumerate(cells):
            r, c = divmod(i, cols)
            canvas.paste(cell, (c * W_cell, r * H_cell))

        _atomic_save(canvas, self._board_path)

    def _render_curve(self) -> None:
        """Write the running mIoU curve PNG atomically."""
        if len(self._miou_history) < 2:
            return

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(self._indices, self._miou_history, linewidth=1.5, color="#4da6ff")
        ax.fill_between(self._indices, self._miou_history, alpha=0.15, color="#4da6ff")
        ax.set_xlabel("Samples processed", fontsize=10)
        ax.set_ylabel("Running mIoU", fontsize=10)
        ax.set_title(
            f"Running mIoU — latest: {self._miou_history[-1]:.4f}  "
            f"(n={len(self._ious)})",
            fontsize=11,
        )
        ax.set_ylim(0, max(max(self._miou_history) * 1.15, 0.1))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        tmp = self._curve_path.with_suffix(".tmp.png")
        fig.savefig(str(tmp), dpi=110, bbox_inches="tight")
        plt.close(fig)
        tmp.replace(self._curve_path)

    def _render_summary(self, class_ious: Dict[str, List[float]]) -> None:
        """
        Final bar chart of per-class mIoU, saved alongside the other files.
        """
        if not class_ious:
            return

        classes    = sorted(class_ious)
        means      = [float(np.mean(class_ious[c])) for c in classes]
        counts     = [len(class_ious[c]) for c in classes]
        overall    = float(np.mean([v for vs in class_ious.values() for v in vs]))
        colours    = [
            "#4caf50" if m >= 0.4 else "#ff9800" if m >= 0.2 else "#f44336"
            for m in means
        ]

        fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.7), 5))
        bars = ax.bar(classes, means, color=colours, edgecolor="white", linewidth=0.5)

        # Annotate bars with N and mIoU value
        for bar, m, n in zip(bars, means, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{m:.3f}\n(n={n})",
                ha="center", va="bottom", fontsize=8,
            )

        ax.axhline(overall, color="white", linewidth=1.5, linestyle="--",
                   label=f"Overall mIoU = {overall:.4f}")
        ax.set_ylim(0, min(1.05, max(means) * 1.35 + 0.05))
        ax.set_ylabel("mIoU", fontsize=11)
        ax.set_title("Per-class mIoU Summary", fontsize=13)
        ax.tick_params(axis="x", rotation=35)
        ax.legend(fontsize=10)
        fig.tight_layout()

        out = self.eval_dir / "summary_chart.png"
        tmp = out.with_suffix(".tmp.png")
        fig.savefig(str(tmp), dpi=120, bbox_inches="tight")
        plt.close(fig)
        tmp.replace(out)


# ── Utility ───────────────────────────────────────────────────────────────────

def _atomic_save(img: Image.Image, path: Path) -> None:
    """Write PIL image to a .tmp file then rename, so readers never see partial writes."""
    tmp = path.with_suffix(".tmp.png")
    img.save(str(tmp))
    tmp.replace(path)
