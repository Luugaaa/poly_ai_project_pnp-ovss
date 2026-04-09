"""
GradCAM-style map sharpening (§3.1 of PnP-OVSS)
================================================
Paper equation (1):

    M̃^(k) = max(0, ∂L_ITM / ∂M^(k))  ⊗  M^(k)

Returns a FLAT [P*P] tensor of per-patch salience values.
The caller (salience_dropout) passes these through the PatchStrategy
to aggregate into segment scores and convert to spatial maps.
"""

from __future__ import annotations

from typing import List

import torch


def extract_class_attention(
    attn: torch.Tensor,
    head_idx: int,
    class_token_indices: List[int],
) -> torch.Tensor:
    """
    Extract the flat per-patch attention for a given head and class token(s).

    Parameters
    ----------
    attn               : Tensor [B, num_heads, text_len, img_len]
                         img_len = P*P + 1  (patch tokens + image CLS)
    head_idx           : int
    class_token_indices: list[int] — averaged when multi-token class name

    Returns
    -------
    Tensor [P*P]  — flat attention over patch tokens (CLS dropped).
    """
    head_attn  = attn[0, head_idx]                           # [text_len, img_len]
    class_attn = head_attn[class_token_indices].mean(dim=0)  # [img_len]
    return class_attn[1:]                                    # [P*P] — drop image CLS


def compute_gradcam_salience(
    attn: torch.Tensor,
    attn_grad: torch.Tensor,
    head_idx: int,
    class_token_indices: List[int],
) -> torch.Tensor:
    """
    Compute the flat GradCAM salience vector M̃^(k) for one class.

        M̃^(k) = ReLU( ∂L_ITM/∂M^(k) )  ⊗  M^(k)       (equation 1)

    Zeroing of dropped patches is NOT done here — the PatchStrategy handles
    it at the pixel level (before the forward pass), so the attention over
    masked patches is already near zero.

    Parameters
    ----------
    attn      : Tensor [B, num_heads, text_len, img_len]
    attn_grad : Tensor [B, num_heads, text_len, img_len]
    head_idx  : int
    class_token_indices : list[int]

    Returns
    -------
    salience : Tensor [P*P]
    """
    M = extract_class_attention(attn,      head_idx, class_token_indices)
    G = extract_class_attention(attn_grad, head_idx, class_token_indices)
    return torch.relu(G) * M   # [P*P]
