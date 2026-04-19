import re

with open("utils/postprocess.py", "r") as f:
    text = f.read()

target = """
        # Min-max normalise feature-level map
        smap = _normalize(smap.astype(np.float32))
        
        # Threshold: set values < threshold to 0
        smap = smap * (smap >= threshold)
        
        # Resize to original image resolution
        smap_pil = Image.fromarray((smap * 255).astype(np.uint8), mode="L")
        smap_resized = np.array(smap_pil.resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0
        
        # Min-max normalise again
        smap_resized = _normalize(smap_resized)
"""

replacement = """
        raw_smap = smap.astype(np.float32)
        # Min-max normalise feature-level map to build the boolean threshold mask
        norm_smap = _normalize(raw_smap)
        mask = (norm_smap >= threshold).astype(np.float32)
        
        # Mask the *raw* prob map, as done in the paper: Blip_final_pred = pred_map * thresholded_pred_map
        smap = raw_smap * mask
        
        # Resize to original image resolution using bilinear interpolation
        # Using torch for exact match to mode='bilinear', align_corners=True
        import torch
        smap_t = torch.from_numpy(smap).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
        smap_resized_t = torch.nn.functional.interpolate(smap_t, size=(H, W), mode='bilinear', align_corners=True).squeeze()
        smap_resized = smap_resized_t.numpy()
        
        # Min-max normalise again (Scale_0_1)
        smap_resized = _normalize(smap_resized)
"""
text = text.replace(target, replacement)
with open("utils/postprocess.py", "w") as f:
    f.write(text)
print("done")
