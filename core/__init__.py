from .gradcam import compute_gradcam_salience, extract_class_attention
from .salience_dropout import salience_dropout
from .patch_strategy import (
    PatchStrategy,
    RegularPatchStrategy,
    RegularFreePatchStrategy,
    SuperpixelPatchStrategy,
    build_strategy,
)
from .clip_reward import CLIPReward

__all__ = [
    "compute_gradcam_salience",
    "extract_class_attention",
    "salience_dropout",
    "PatchStrategy",
    "RegularPatchStrategy",
    "RegularFreePatchStrategy",
    "SuperpixelPatchStrategy",
    "build_strategy",
    "CLIPReward",
]
