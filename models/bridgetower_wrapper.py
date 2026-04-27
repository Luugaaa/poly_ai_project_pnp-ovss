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

from transformers import BridgeTowerImageProcessor, BridgeTowerProcessor
from transformers import AutoTokenizer


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
    PROCESSOR_MODEL = "BridgeTower/bridgetower-base-itm-mlm"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[torch.device] = None,
        input_size: Optional[int] = None,
    ) -> None:
        self.device = device if device is not None else get_device()
        self.input_size = input_size
        print(f"[BridgeTowerWrapper] Loading '{model_name}' on device '{self.device}' …")
        # Load processor from the same model being used so image_size matches.
        # PROCESSOR_MODEL (base) has image_size=288; large model expects a different size.
        image_processor = BridgeTowerImageProcessor.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.PROCESSOR_MODEL, use_fast=True)
        self.processor: BridgeTowerProcessor = BridgeTowerProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
        )
        # RoBERTa/BridgeTower has a <pad> token (ID 1) but doesn't set pad_token attribute
        # Explicitly set it so padding=True works in tokenizer calls
        if self.processor.tokenizer.pad_token is None:
            # Token ID 1 is <pad> in RoBERTa vocab
            self.processor.tokenizer.pad_token_id = 1

        # Cache the processor's native image size so preprocess never overrides it.
        # BridgeTowerImageProcessor.size returns a SizeDict (not a plain dict),
        # so we check for the .shortest_edge attribute before dict/int fallbacks.
        try:
            size_cfg = self.processor.image_processor.size
            if hasattr(size_cfg, "shortest_edge"):
                self._native_image_size: int = int(size_cfg.shortest_edge)
            elif hasattr(size_cfg, "get"):
                self._native_image_size = int(
                    size_cfg.get("shortest_edge", size_cfg.get("height", 288))
                )
            else:
                self._native_image_size = int(size_cfg)
        except (AttributeError, TypeError, ValueError):
            self._native_image_size = 288

        self.model = _BridgeTowerMatchingModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"[BridgeTowerWrapper] Ready. Native image size: {self._native_image_size}px")

    @property
    def _base_model(self):
        return getattr(self.model, "bridgetower", self.model)

    @property
    def num_text_layers(self) -> int:
        """Number of cross-modal encoder layers (not text encoder layers).
        
        BridgeTower has 6 cross-modal layers that perform cross-attention between
        image and text. This is what we tune for layer/head selection.
        """
        return len(self._base_model.cross_modal_text_layers)

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
        # Always use the native size from the processor that was loaded for this
        # model variant. Ignores input_size so callers can't accidentally force
        # the BLIP default (336) onto BridgeTower.
        return self._native_image_size

    @property
    def num_patches_per_side(self) -> int:
        return self.image_size // self.patch_size

    def preprocess(self, image: Image.Image, text: str) -> dict[str, torch.Tensor]:
        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding=True,
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        out = self.processor(images=image, return_tensors="pt")
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
                    interpolate_pos_encoding=True,
                )

            attn = captured.pop("attn", None)
            captured.clear()
            if attn is None:
                raise RuntimeError(f"BridgeTower cross-attention hook did not fire for layer {layer_idx}.")

            loss = getattr(outputs, "loss", None)
            if loss is None:
                logits = getattr(outputs, "logits", None)
                if logits is None:
                    raise RuntimeError("BridgeTower model did not return logits or loss.")
                loss = logits[:, -1].sum()
            # Free the output object early; the grad computation only needs loss.
            del outputs

            grad = torch.autograd.grad(loss, attn, retain_graph=False, create_graph=False, allow_unused=True)[0]
            del loss
            if grad is None:
                raise RuntimeError("BridgeTower cross-attention gradient was not captured.")

            return attn.detach(), grad.detach()
        finally:
            hook_handle.remove()
            captured.clear()

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
        return self._capture_cross_attention(pixel_values, input_ids, attention_mask, layer_idx)

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