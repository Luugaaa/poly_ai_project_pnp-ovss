import re

with open("utils/postprocess.py", "r") as f:
    text = f.read()

target = """
    # 2. Add Background Channel
    max_map = np.max(stack_preds, axis=0)
    bg_map = (max_map == 0.0).astype(np.float32)
    # The paper's logic allows Blip_max_map == 0 to form the actual background
    # We concatenate it as the first channel
    pred_w_bg = np.concatenate([bg_map[np.newaxis, ...], stack_preds], axis=0) # [C+1, H, W]
"""

replacement = """
    # 2. Add Background Channel
    # The paper calculates background from the ensemble max map:
    max_map = np.max(stack_preds, axis=0)
    bg_map = (max_map == 0.0).astype(np.float32)
    # The paper's logic concatenates background as the first channel
    pred_w_bg = np.concatenate([bg_map[np.newaxis, ...], stack_preds], axis=0) # [C+1, H, W]
"""
text = text.replace(target, replacement)
with open("utils/postprocess.py", "w") as f:
    f.write(text)
