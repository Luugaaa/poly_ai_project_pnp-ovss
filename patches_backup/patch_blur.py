import re

with open("utils/postprocess.py", "r") as f:
    text = f.read()

# The paper blur uses scale*max(img_shape). The img_shape passed in inference is (label_trues.shape[0], label_trues.shape[1]) which is the ORIGINAL IMAGE SHAPE (H, W).
# So they blur using sigma = 0.05 * max(H, W)
# Our code blurs using sigma = 0.05 * min(H_feat, W_feat)
# Also our normalization has `mx - mn` while the paper's does `att / att.max()`.
# Let's fix our postprocess.py to literally match the paper.

target = """
    # 3. Gaussian Blur (applied per channel)
    short_side = min(H_feat, W_feat)
    sigma_px = max(gaussian_sigma * short_side, 0.5)
    
    blurred_preds = []
    for i in range(pred_w_bg.shape[0]):
        b = gaussian_filter(pred_w_bg[i], sigma=sigma_px)
        b = _normalize(b)
        blurred_preds.append(b)
"""

replacement = """
    # 3. Gaussian Blur (applied per channel)
    # The original paper uses 0.05 * max(H, W) where H,W are the original image sizes.
    sigma_px = gaussian_sigma * max(H, W)
    
    blurred_preds = []
    for i in range(pred_w_bg.shape[0]):
        b = gaussian_filter(pred_w_bg[i], sigma=sigma_px)
        # The paper normalizes via: X = (X - X.min()); X = X / X.max()
        mn_val = b.min()
        b = b - mn_val
        mx_val = b.max()
        if mx_val > 0:
            b = b / mx_val
        blurred_preds.append(b)
"""
text = text.replace(target, replacement)
with open("utils/postprocess.py", "w") as f:
    f.write(text)
print("done")
