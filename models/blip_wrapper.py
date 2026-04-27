"""
BLIPWrapper
===========
Thin wrapper around ``BlipForImageTextRetrieval`` (Salesforce/blip-itm-base-coco)
that exposes the two operations needed by PnP-OVSS:

1. ``forward_with_gradcam`` — single forward pass that captures the
   cross-attention probability tensor at a chosen layer and its gradient
   w.r.t. the ITM "matching" loss (needed for GradCAM, §3.1).

2. ``mask_patches`` — zero-out image pixel regions corresponding to a set
   of (row, col) patch indices (needed for Salience DropOut, §3.2).

Design notes
------------
* The cross-attention in BLIP's text encoder is realised as a BERT-style
  decoder: ``model.text_encoder.encoder.layer[i].crossattention.self`` is
  a ``BlipTextSelfAttention`` that computes text-to-image attention weights
  of shape ``[B, num_heads, text_len, img_len]``.

* Gradients are captured via ``tensor.retain_grad()`` inside a forward hook.
  A backward hook is registered as a more reliable fallback for backends
  (e.g. MPS) where ``retain_grad`` may not flush the grad buffer in time.

* Device priority: MPS → CUDA → CPU.
"""

from __future__ import annotations

import math
from typing import Optional, Set, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import BlipForImageTextRetrieval, BlipProcessor


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class BLIPWrapper:
    """
    Wraps ``BlipForImageTextRetrieval`` for PnP-OVSS.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.  Defaults to the BLIP-base ITM
        checkpoint used in the paper.
    device : torch.device, optional
        Overrides the automatic device selection.
    """

    DEFAULT_MODEL = "Salesforce/blip-itm-base-coco"
    LARGE_MODEL   = "Salesforce/blip-itm-large-coco"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[torch.device] = None,
        input_size: Optional[int] = None,
    ) -> None:
        self.device = device if device is not None else get_device()
        self.input_size = input_size
        print(f"[BLIPWrapper] Loading '{model_name}' on device '{self.device}' …")
        self.processor: BlipProcessor = BlipProcessor.from_pretrained(model_name)
        self.model: BlipForImageTextRetrieval = (
            BlipForImageTextRetrieval.from_pretrained(model_name)
        )
        self.model.to(self.device)
        self.model.eval()
        print("[BLIPWrapper] Ready.")

    # ------------------------------------------------------------------
    # Architecture properties (derived from model config)
    # ------------------------------------------------------------------

    @property
    def num_text_layers(self) -> int:
        """Number of cross-attention layers in the text encoder."""
        return len(self.model.text_encoder.encoder.layer)

    @property
    def num_heads(self) -> int:
        """Number of attention heads per cross-attention layer."""
        return self.model.text_encoder.config.num_attention_heads

    @property
    def patch_size(self) -> int:
        """ViT patch size in pixels (typically 16)."""
        return self.model.vision_model.config.patch_size

    @property
    def image_size(self) -> int:
        """Expected square input image size in pixels (typically 384)."""
        return self.input_size if self.input_size is not None else self.model.vision_model.config.image_size

    @property
    def num_patches_per_side(self) -> int:
        """P — number of patches along one image dimension (image_size / patch_size)."""
        return self.image_size // self.patch_size

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, image: Image.Image, text: str) -> dict[str, torch.Tensor]:
        """
        Run ``BlipProcessor`` on (image, text) and move tensors to ``self.device``.

        Returns a dict with keys ``pixel_values``, ``input_ids``,
        ``attention_mask``.
        """
        kwargs = {
            "images": image,
            "text": text,
            "return_tensors": "pt",
            "padding": True,
        }
        if self.input_size is not None:
            kwargs["size"] = {"height": self.input_size, "width": self.input_size}
        inputs = self.processor(**kwargs)
        return {k: v.to(self.device) for k, v in inputs.items()}

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image only → ``pixel_values`` [1, 3, H, W].

        Use this to cache the image tensor when the same image is paired with
        multiple text prompts (avoids redundant vision preprocessing).
        """
        kwargs = {"images": image, "return_tensors": "pt"}
        if self.input_size is not None:
            kwargs["size"] = {"height": self.input_size, "width": self.input_size}
        out = self.processor(**kwargs)
        return out["pixel_values"].to(self.device)

    def preprocess_text(self, texts: list[str] | str) -> dict[str, torch.Tensor]:
        """
        Tokenize one or a list of prompts → ``{input_ids, attention_mask}``.

        Uses right-padding so token positions are stable regardless of batch
        size — class token indices computed on individual prompts remain valid.
        """
        out = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return {k: v.to(self.device) for k, v in out.items()}

    # ------------------------------------------------------------------
    # Forward pass with gradient capture
    # ------------------------------------------------------------------

    def forward_with_gradcam(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Run a full forward pass through the BLIP ITM head and collect the
        cross-attention probability tensor at ``layer_idx`` together with its
        gradient w.r.t. the ITM "matching" loss.

        The ITM loss is ``CrossEntropy(itm_logits, label=1)`` where label 1
        means "matching" — exactly as in §3.1 of the paper.

        Parameters
        ----------
        pixel_values    : Tensor [1, 3, H, W]
        input_ids       : Tensor [1, text_len]
        attention_mask  : Tensor [1, text_len]
        layer_idx       : int — which cross-attention layer to probe

        Returns
        -------
        attn  : Tensor [1, num_heads, text_len, img_len]  (detached, on device)
        grad  : Tensor [1, num_heads, text_len, img_len]  (detached, on device)
            ``img_len = P*P + 1``  (patch tokens + image CLS token)
        Both are ``None`` if the hook did not fire (e.g. wrong layer index).
        """
        # Vision encoder runs with no_grad — gradients only through text encoder.
        image_embeds = self.encode_image_embeds(pixel_values)
        return self.forward_with_gradcam_from_embeds(
            image_embeds, input_ids, attention_mask, layer_idx
        )

    # ------------------------------------------------------------------
    # Split vision / text-encoder forward
    # ------------------------------------------------------------------

    def encode_image_embeds(
        self,
        pixel_values: torch.Tensor,  # [B, 3, H, W]
    ) -> torch.Tensor:
        """
        Run the vision encoder with no_grad and return image embeddings.

        Calling this separately avoids storing ViT activations for backward,
        which is the dominant memory and compute cost in a full forward pass.

        Returns
        -------
        Tensor [B, img_len, D]  — detached, on device.
        """
        with torch.no_grad():
            vision_out = self.model.vision_model(
                pixel_values=pixel_values,
                interpolate_pos_encoding=True,
                return_dict=True,
            )
        return vision_out.last_hidden_state  # [B, img_len, D]

    def forward_with_gradcam_from_embeds(
        self,
        image_embeds: torch.Tensor,    # [B, img_len, D]  — pre-computed, no grad
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Like ``forward_with_gradcam`` but takes pre-computed image embeddings.

        Gradients flow only through the text encoder and ITM head — the vision
        encoder is never called.  This is ~2-3× faster because:
          * ViT activations (large) are never stored for backward.
          * ViT backward pass is skipped entirely.

        Use ``encode_image_embeds`` to produce ``image_embeds``, optionally
        caching results when the same image is queried multiple times.

        Parameters
        ----------
        image_embeds    : Tensor [B, img_len, D]  — from ``encode_image_embeds``.
        input_ids       : Tensor [B, text_len]
        attention_mask  : Tensor [B, text_len]
        layer_idx       : int — cross-attention layer to probe.

        Returns
        -------
        attn, grad : same contract as ``forward_with_gradcam``.
        """
        captured: dict = {}

        def _forward_hook(module, inp, output):
            if not (isinstance(output, tuple) and len(output) >= 2):
                return
            attn_probs: torch.Tensor = output[1]
            if attn_probs.requires_grad:
                attn_probs.retain_grad()
            captured["attn"] = attn_probs
            captured["grad"] = None

            def _grad_hook(g: torch.Tensor) -> None:
                captured["grad"] = g.detach().clone()

            if attn_probs.requires_grad:
                attn_probs.register_hook(_grad_hook)

        target = (
            self.model.text_encoder.encoder.layer[layer_idx].crossattention.self
        )
        hook_handle = target.register_forward_hook(_forward_hook)

        self.model.zero_grad()

        image_attn_mask = torch.ones(
            image_embeds.shape[:2], dtype=torch.long, device=self.device
        )

        with torch.enable_grad():
            text_out = self.model.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attn_mask,
                output_attentions=True,
                return_dict=True,
            )
            itm_logits = self.model.itm_head(
                text_out.last_hidden_state[:, 0, :]
            )

            loss = itm_logits[:, 1].sum()
            attn_tensor = captured.get("attn")
            if attn_tensor is not None:
                grad_tuple = torch.autograd.grad(
                    outputs=loss,
                    inputs=attn_tensor,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
                if grad_tuple and grad_tuple[0] is not None:
                    captured["grad"] = grad_tuple[0].detach().clone()

        hook_handle.remove()

        attn = captured.get("attn")
        if attn is None:
            return None, None

        grad = captured.get("grad")
        if grad is None and attn.grad is not None:
            grad = attn.grad.detach().clone()

        return attn.detach(), grad

    # ------------------------------------------------------------------
    # Patch masking (for Salience DropOut)
    # ------------------------------------------------------------------

    def mask_patches(
        self,
        pixel_values: torch.Tensor,
        dropped_patches: Set[Tuple[int, int]],
    ) -> torch.Tensor:
        """
        Return a copy of ``pixel_values`` with the pixel regions
        corresponding to ``dropped_patches`` set to zero.

        ``dropped_patches`` is a ``set`` of ``(row, col)`` indices in the
        P×P patch grid (0-indexed).  Patch ``(r, c)`` maps to the pixel
        region ``[r*patch_size : (r+1)*patch_size, c*patch_size : (c+1)*patch_size]``.

        Setting the pixels to 0 (in the normalised space) corresponds to a
        neutral "missing" signal — consistent with the paper's description of
        "zeroing out" image patches.
        """
        if not dropped_patches:
            return pixel_values  # nothing to do

        ps = self.patch_size
        masked = pixel_values.clone()
        for row, col in dropped_patches:
            r0, r1 = row * ps, (row + 1) * ps
            c0, c1 = col * ps, (col + 1) * ps
            masked[:, :, r0:r1, c0:c1] = 0.0
        return masked
