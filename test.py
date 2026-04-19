import re

with open("utils/postprocess.py", "r") as f:
    text = f.read()

# Our current logic:
# 1. normalize feature-level map
# 2. threshold map (set values < threshold to 0) -> paper does map * (map >= threshold) Wait, but then they interpolate the raw pred map multiplied by threshold mask.
# Let's check the paper precisely:
#   thresholded_pred_map[i] = (pred_map[i] - min) / (max - min)
#   thresholded_pred_map = bool(thresholded_pred_map >= args.threshold)
#   Blip_final_pred = pred_map * thresholded_pred_map
#   Blip_final_pred = interpolate(Blip_final_pred, mode='bilinear')
#   Blip_final_pred = Scale_0_1(Blip_final_pred)
# 
# Wait, let's look at `Scale_0_1`. What is `Scale_0_1`?
