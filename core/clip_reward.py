"""
CLIP-based reward function for annotation-free hyperparameter tuning (§3.4)
===========================================================================
Implements equations (6) and (7) from PnP-OVSS.

    Reward = Σ_{k∈K(I)}  𝟙[ Pr(M^(k) ⊗ I, k) > Pr(0, k) ]          (6)

    Pr(I, k) = exp(f(I, k)) / Σ_{k'∈K(I)} exp(f(I, k'))               (7)

where
  M^(k)       — binary/soft segmentation mask for class k  [H, W] float ∈ [0,1]
  M^(k) ⊗ I  — image with non-foreground pixels zeroed (element-wise multiply)
  0           — all-black image of the same size
  f(I, k)     — CLIP cosine similarity between image I and text "a photo of {k}"
  K(I)        — all classes present in image I (image-level labels, no pixel GT)

The paper uses CLIP ViT-L/14 at 336×336 resolution.

Usage
-----
    reward = CLIPReward()                         # loads model once
    score  = reward.compute_reward(
        image      = pil_image,
        mask       = mask_np,          # [H, W] float32 ∈ [0, 1]
        class_name = "elephant",
        all_classes = ["elephant"],    # K(I) — classes present in this image
    )                                  # → 0 or 1
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


_PROMPT_TEMPLATE = "a photo of a {}"   # matches CLIP zero-shot convention


class CLIPReward:
    """
    Wraps CLIP ViT-L/14-336 for the PnP-OVSS reward function.

    Parameters
    ----------
    model_name : str  — HuggingFace CLIP model identifier.
    device     : torch.device, optional — auto-selected if None.
    """

    DEFAULT_MODEL  = "openai/clip-vit-large-patch14-336"
    CLIP_IMG_SIZE  = 336

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[torch.device] = None,
    ) -> None:
        from transformers import CLIPModel, CLIPProcessor

        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        self.device = device
        print(f"[CLIPReward] Loading '{model_name}' on '{device}' …")
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model     = CLIPModel.from_pretrained(model_name).to(device)
        self._model.eval()
        print("[CLIPReward] Ready.")

    # ── Public API ────────────────────────────────────────────────────────────

    def compute_reward(
        self,
        image:       Image.Image,
        mask:        np.ndarray,
        class_name:  str,
        all_classes: List[str],
    ) -> int:
        """
        Compute the binary reward for one (image, class) pair.

        Parameters
        ----------
        image       : PIL.Image.Image — original RGB image.
        mask        : ndarray [H, W]  float32 ∈ [0, 1] — predicted salience mask.
        class_name  : str             — class k to evaluate.
        all_classes : list[str]       — K(I), all classes present in this image.
                      Used as the denominator of the softmax (eq. 7).

        Returns
        -------
        1 if the masked image is more CLIP-similar to class_name than the black
        image is; 0 otherwise.
        """
        # Build text prompts for all classes in K(I)
        texts = [_PROMPT_TEMPLATE.format(c) for c in all_classes]

        # Masked image: M^(k) ⊗ I
        masked = _apply_mask(image, mask)

        # Black image: all zeros, same size
        black = Image.new("RGB", image.size, (0, 0, 0))

        # CLIP softmax probabilities for both images over K(I)
        pr_masked = self._clip_probs(masked, texts, all_classes)
        pr_black  = self._clip_probs(black,  texts, all_classes)

        # Equation (6): reward += 1 if masked > black for class k
        return int(pr_masked.get(class_name, 0.0) > pr_black.get(class_name, 0.0))

    def compute_batch_reward(
        self,
        image:       Image.Image,
        masks:       Dict[str, np.ndarray],
        all_classes: List[str],
    ) -> Dict[str, int]:
        """
        Compute rewards for all classes at once (single CLIP pass per image).

        Parameters
        ----------
        masks : dict[class_name → mask ndarray [H, W]]

        Returns
        -------
        dict[class_name → 0 or 1]
        """
        texts     = [_PROMPT_TEMPLATE.format(c) for c in all_classes]
        black     = Image.new("RGB", image.size, (0, 0, 0))
        pr_black  = self._clip_probs(black, texts, all_classes)

        rewards: Dict[str, int] = {}
        for cls, mask in masks.items():
            masked    = _apply_mask(image, mask)
            pr_masked = self._clip_probs(masked, texts, all_classes)
            rewards[cls] = int(
                pr_masked.get(cls, 0.0) > pr_black.get(cls, 0.0)
            )
        return rewards

    # ── Internal ─────────────────────────────────────────────────────────────

    def _clip_probs(
        self,
        image: Image.Image,
        texts: List[str],
        class_names: List[str],
    ) -> Dict[str, float]:
        """
        Run CLIP on (image, texts), return softmax probabilities keyed by class.

        Equation (7):  Pr(I, k) = exp(f(I,k)) / Σ_{k'} exp(f(I,k'))
        """
        inputs = self._processor(
            text   = texts,
            images = image,
            return_tensors = "pt",
            padding = True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out        = self._model(**inputs)
            # logits_per_image: [1, num_texts]
            logits     = out.logits_per_image[0]          # [num_texts]
            probs      = F.softmax(logits, dim=0).cpu().tolist()

        return {cls: p for cls, p in zip(class_names, probs)}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _apply_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """
    Element-wise multiply image pixels by the mask.

    mask values in [0, 1]:  0 → black (zeroed),  1 → original pixel retained.
    The paper uses the continuous mask (not hard-thresholded) as the weight.
    """
    img_np = np.array(image.convert("RGB"), dtype=np.float32)   # [H, W, 3]

    # Resize mask to image resolution if needed
    H, W   = img_np.shape[:2]
    mH, mW = mask.shape
    if (mH, mW) != (H, W):
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (W, H), Image.BILINEAR
        )
        mask = np.array(mask_pil, dtype=np.float32) / 255.0

    masked = img_np * mask[..., np.newaxis]                      # broadcast over channels
    return Image.fromarray(masked.astype(np.uint8))
