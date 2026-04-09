"""
Salience DropOut (§3.2 of PnP-OVSS)
=====================================
Runs 1 + dropout_rounds forward passes.  Each round:
  1. Forward pass → GradCAM flat scores [P*P]
  2. Aggregate via PatchStrategy → segment scores [N]
  3. Accumulate
  4. Select TOP-k highest-scoring remaining segments → zero their pixels

The accumulated [N] segment score vector is returned.  Callers convert it
to a spatial map via  strategy.to_spatial(scores).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set

import numpy as np
import torch
from PIL import Image

from core.gradcam import compute_gradcam_salience
from core.patch_strategy import PatchStrategy
from models.blip_wrapper import BLIPWrapper


def salience_dropout(
    wrapper: BLIPWrapper,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int,
    head_idx: int,
    class_token_indices: List[int],
    strategy: PatchStrategy,
    dropout_rounds: int = 3,
    patches_per_drop: int = 10,
    verbose: bool = True,
    viz_dir: Optional[Path] = None,
    original_image: Optional[Image.Image] = None,
) -> torch.Tensor:
    """
    Full Salience DropOut loop.

    Parameters
    ----------
    wrapper             : BLIPWrapper
    pixel_values        : Tensor [1, 3, H, W]
    input_ids           : Tensor [1, text_len]
    attention_mask      : Tensor [1, text_len]
    layer_idx           : int — cross-attention layer (0-based)
    head_idx            : int — attention head (0-based)
    class_token_indices : list[int]
    strategy            : PatchStrategy — regular or superpixel
    dropout_rounds      : int — number of dropout iterations after initial pass
    patches_per_drop    : int — segments removed per round
    verbose             : bool — print per-pass info
    viz_dir             : Path, optional — save iteration PNGs here
    original_image      : PIL.Image, optional — needed when viz_dir is set

    Returns
    -------
    accumulated : Tensor [N]  — sum of per-segment GradCAM scores over all passes.
    """
    N = strategy.num_segments
    accumulated = torch.zeros(N, device=wrapper.device)
    remaining: Set[int] = set(range(N))
    dropped:   Set[int] = set()
    current_pv = pixel_values
    total_passes = 1 + dropout_rounds

    for t in range(total_passes):
        pass_label = "initial" if t == 0 else f"dropout round {t}"
        if verbose:
            print(
                f"    [SalienceDropOut] pass {t+1}/{total_passes} "
                f"({pass_label}) — {len(dropped)} segments zeroed"
            )

        # ── Forward pass + GradCAM ────────────────────────────────────────
        attn, attn_grad = wrapper.forward_with_gradcam(
            current_pv, input_ids, attention_mask, layer_idx
        )
        if attn is None or attn_grad is None:
            if verbose:
                print(
                    f"    [SalienceDropOut] WARNING: no attention at layer "
                    f"{layer_idx}, pass {t+1}. Skipping."
                )
            break

        flat_scores = compute_gradcam_salience(
            attn, attn_grad, head_idx, class_token_indices
        )                                   # [P*P]

        seg_scores = strategy.aggregate(flat_scores)  # [N]

        # Explicitly zero dropped segments (robustness across backends)
        for idx in dropped:
            seg_scores[idx] = 0.0

        accumulated = accumulated + seg_scores

        # ── Iteration visualisation ───────────────────────────────────────
        if viz_dir is not None and original_image is not None:
            from utils.visualize import save_iteration_viz
            save_iteration_viz(
                t=t,
                total_passes=total_passes,
                current_scores=seg_scores.cpu().numpy(),
                accumulated_scores=accumulated.cpu().numpy(),
                dropped=dropped,
                strategy=strategy,
                original_image=original_image,
                out_dir=viz_dir,
            )

        # ── Dropout step (skip after final pass) ──────────────────────────
        if t < dropout_rounds:
            new_drops = strategy.top_k(seg_scores, remaining, k=patches_per_drop)
            dropped   |= new_drops
            remaining -= new_drops
            current_pv = strategy.mask_segments(pixel_values, dropped)

    return accumulated  # [N]
