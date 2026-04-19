"""
Data-loading utilities for PnP-OVSS
=====================================
The class name is inferred from the **parent directory** of the image file,
so ``images/elephant/elephant.png`` → class ``"elephant"``.

The text prompt is formatted exactly as in the paper (§3):

    "A picture of [class]"

Token indices for the class name are found by encoding the prefix
``"A picture of"`` and offsetting past it (accounting for the ``[CLS]``
special token).  Multi-token class names are fully captured and averaged
over in ``gradcam.py``.

We skip the first three content tokens ("A", "picture", "of") as the paper
instructs: they carry no semantic class information.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

from PIL import Image
from transformers import BlipProcessor

# ---- Prompt template -------------------------------------------------------
PROMPT_PREFIX = "A picture of"   # paper §3
_SKIP_WORDS = ["a", "picture", "of"]  # tokens to exclude from attention maps


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_class_from_path(image_path: Union[str, Path]) -> str:
    """
    Infer the semantic class label from the parent directory name.

    Examples
    --------
    >>> get_class_from_path("images/elephant/elephant.png")
    'elephant'
    >>> get_class_from_path("/data/tennis racket/img001.jpg")
    'tennis racket'
    """
    return Path(image_path).parent.name


def format_prompt(classes: List[str] | str) -> str:
    """Return the VLM text prompt for the given class names concatenated."""
    if isinstance(classes, str):
        classes = [classes]
    return f"{PROMPT_PREFIX} " + " ".join(classes)

def get_class_token_indices(
    processor: BlipProcessor,
    prompt: str,
    classes: List[str] | str,
) -> Dict[str, List[int]]:
    """
    Tokenise ``prompt`` and return the token position indices that correspond
    to the class names, skipping ``"A picture of"``.

    Returns a dictionary mapping each class name to a list of its token indices.
    """
    if isinstance(classes, str):
        classes = [classes]

    tokenizer = processor.tokenizer
    
    # We expect [CLS] a picture of [class1_tokens] [class2_tokens] ... [SEP]
    prefix_ids = tokenizer.encode(PROMPT_PREFIX, add_special_tokens=False)
    start_idx = 1 + len(prefix_ids)
    
    class_indices = {}
    curr_idx = start_idx
    for cls in classes:
        # Avoid special tokens for individual class words as they are in the middle of a string
        cls_ids = tokenizer(cls, add_special_tokens=False).input_ids
        num_tokens = len(cls_ids)
        class_indices[cls] = list(range(curr_idx, curr_idx + num_tokens))
        curr_idx += num_tokens
        
    return class_indices


def load_image(image_path: Union[str, Path]) -> Image.Image:
    """Load an image from disk and return it as an RGB PIL Image."""
    return Image.open(image_path).convert("RGB")
