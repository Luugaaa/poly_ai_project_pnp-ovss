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
from typing import List, Union

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


def format_prompt(class_name: str) -> str:
    """Return the VLM text prompt for the given class name."""
    return f"{PROMPT_PREFIX} {class_name}"


def get_class_token_indices(
    processor: BlipProcessor,
    prompt: str,
    class_name: str,  # kept for clarity / future validation
) -> List[int]:
    """
    Tokenise ``prompt`` and return the token position indices that correspond
    to the class name, skipping ``"A picture of"``.

    The returned indices are positions in the full ``input_ids`` tensor
    (including the leading ``[CLS]`` at position 0).

    Strategy
    --------
    1. Encode the full prompt with special tokens → full token sequence.
    2. Encode ``PROMPT_PREFIX`` without special tokens → prefix length.
    3. Class tokens start at  ``1 + prefix_len``  (1 for ``[CLS]``).
    4. Class tokens end before ``[SEP]`` (last token).

    Parameters
    ----------
    processor  : BlipProcessor — provides the tokeniser.
    prompt     : str           — e.g. "A picture of elephant"
    class_name : str           — e.g. "elephant"  (for documentation only)

    Returns
    -------
    List[int]  — indices into the ``input_ids`` sequence.
    """
    tokenizer = processor.tokenizer

    full_ids: List[int] = tokenizer.encode(prompt, add_special_tokens=True)
    prefix_ids: List[int] = tokenizer.encode(PROMPT_PREFIX, add_special_tokens=False)

    prefix_len = len(prefix_ids)
    # Position 0 = [CLS], positions 1..prefix_len = prefix tokens,
    # positions prefix_len+1 .. len-2 = class tokens,
    # position len-1 = [SEP]
    class_start = 1 + prefix_len
    class_end = len(full_ids) - 1  # exclusive

    indices = list(range(class_start, class_end))

    if not indices:
        # Fallback: use all non-special tokens
        indices = list(range(1, len(full_ids) - 1))

    return indices


def load_image(image_path: Union[str, Path]) -> Image.Image:
    """Load an image from disk and return it as an RGB PIL Image."""
    return Image.open(image_path).convert("RGB")
