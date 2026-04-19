import re

with open("utils/postprocess.py", "r") as f:
    text = f.read()

target = """        # Ensure we don't normalize entirely flat (all-zero) maps
        if max_v > min_v:
            raw_map = (raw_map - min_v) / (max_v - min_v)
        
        # Apply mask threshold before upsampling, just as original paper does
        stack_preds_raw = (raw_map >= min_thresh) * raw_map
"""

replacement = """        # Threshold is calculated on the normalized map:
        norm_map = raw_map.copy()
        if max_v > min_v:
            norm_map = (norm_map - min_v) / (max_v - min_v)
        
        # Apply mask threshold using normalized map, BUT keep original raw magnitudes for stack_preds, just like original paper does
        stack_preds_raw = (norm_map >= min_thresh) * raw_map
"""

text = text.replace(target, replacement)
with open("utils/postprocess.py", "w") as f:
    f.write(text)
