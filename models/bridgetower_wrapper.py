"""
BridgeTowerWrapper
==================
Thin wrapper around BridgeTower image-text matching models that mirrors the
BLIP wrapper contract used by the tuning and salience pipeline.

BridgeTower uses a ViT-L/14 vision encoder and a RoBERTa text encoder with a
6-layer cross-modal encoder. The wrapper exposes the same GradCAM-facing API
as BLIP so the existing salience-dropout code can reuse it unchanged.
"""

from __future__ import annotations

from typing import Optional, Set, Tuple

import torch
from PIL import Image

try:
    from transformers import BridgeTowerForImageTextMatching as _BridgeTowerMatchingModel
except ImportError:  # pragma: no cover - depends on transformers version
    from transformers import BridgeTowerForImageAndTextRetrieval as _BridgeTowerMatchingModel

from transformers import BridgeTowerProcessor


def get_device() -> torch.device:
    """Return the best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class BridgeTowerWrapper:
    """Wraps BridgeTower retrieval models for PnP-OVSS."""

    DEFAULT_MODEL = "BridgeTower/bridgetower-large-itm-mlm-itc"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[torch.device] = None,
        input_size: Optional[int] = None,
    ) -> None:
        self.device = device if device is not None else get_device()
        self.input_size = input_size
        print(f"[BridgeTowerWrapper] Loading '{model_name}' on device '{self.device}' …")
        self.processor: BridgeTowerProcessor = BridgeTowerProcessor.from_pretrained(model_name)
        self.model = _BridgeTowerMatchingModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print("[BridgeTowerWrapper] Ready.")

    @property
    def _base_model(self):
        return getattr(self.model, "bridgetower", self.model)

    @property
    def num_text_layers(self) -> int:
        return len(self._base_model.text_model.encoder.layer)

    @property
    def num_heads(self) -> int:
        return int(self._base_model.config.num_attention_heads)

    @property
    def patch_size(self) -> int:
        patch_embedding = self._base_model.vision_model.visual.embeddings.patch_embedding
        kernel_size = patch_embedding.kernel_size
        if isinstance(kernel_size, tuple):
            return int(kernel_size[0])
        return int(kernel_size)

    @property
    def image_size(self) -> int:
        if self.input_size is not None:
            return int(self.input_size)
        vision_embeddings = self._base_model.vision_model.visual.embeddings
        image_size = getattr(vision_embeddings, "image_size", None)
        if image_size is not None:
            return int(image_size)
        return int(getattr(self._base_model.config.vision_config, "image_size", 336))

    @property
    def num_patches_per_side(self) -> int:
        return self.image_size // self.patch_size

    def preprocess(self, image: Image.Image, text: str) -> dict[str, torch.Tensor]:
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
        kwargs = {"images": image, "return_tensors": "pt"}
        if self.input_size is not None:
            kwargs["size"] = {"height": self.input_size, "width": self.input_size}
        out = self.processor(**kwargs)
        return out["pixel_values"].to(self.device)

    def preprocess_text(self, texts: list[str] | str) -> dict[str, torch.Tensor]:
        out = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return {k: v.to(self.device) for k, v in out.items()}

    def _capture_cross_attention(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        captured: dict[str, torch.Tensor] = {}

        def _hook(_module, _inputs, output):
            if isinstance(output, tuple) and len(output) >= 2:
                attn_probs = output[1]
                if isinstance(attn_probs, torch.Tensor):
                    captured["attn"] = attn_probs

        target = self._base_model.cross_modal_text_layers[layer_idx].crossattention.self
        hook_handle = target.register_forward_hook(_hook)

        try:
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_token_type_idx": 1,
            }

            with torch.enable_grad():
                outputs = self.model(
                    **model_inputs,
                    labels=torch.ones(input_ids.shape[0], dtype=torch.long, device=self.device),
                )

            attn = captured.get("attn")
            if attn is None:
                raise RuntimeError(f"BridgeTower cross-attention hook did not fire for layer {layer_idx}.")

            loss = getattr(outputs, "loss", None)
            if loss is None:
                logits = getattr(outputs, "logits", None)
                if logits is None:
                    raise RuntimeError("BridgeTower model did not return logits or loss.")
                loss = logits[:, -1].sum()

            grad = torch.autograd.grad(loss, attn, retain_graph=False, create_graph=False, allow_unused=True)[0]
            if grad is None:
                grad = attn.grad
            if grad is None:
                raise RuntimeError("BridgeTower cross-attention gradient was not captured.")

            return attn, grad
        finally:
            hook_handle.remove()

    def get_cross_attention(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_idx: int,
        head_idx: int,
    ) -> torch.Tensor:
        attn, _ = self._capture_cross_attention(pixel_values, input_ids, attention_mask, layer_idx)
        return attn[:, head_idx].detach()

    def forward_with_gradcam(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        attn, grad = self._capture_cross_attention(pixel_values, input_ids, attention_mask, layer_idx)
        return attn.detach(), grad.detach()

    def mask_patches(
        self,
        pixel_values: torch.Tensor,
        dropped_patches: Set[Tuple[int, int]],
    ) -> torch.Tensor:
        if not dropped_patches:
            return pixel_values

        ps = self.patch_size
        masked = pixel_values.clone()
        for row, col in dropped_patches:
            r0, r1 = row * ps, (row + 1) * ps
            c0, c1 = col * ps, (col + 1) * ps
            masked[:, :, r0:r1, c0:c1] = 0.0
        return masked