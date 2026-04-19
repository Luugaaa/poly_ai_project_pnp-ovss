import re

with open("scripts/evaluate.py", "r") as f:
    text = f.read()

target = """                # The ground truth filtered approach (with drop iterations)
                gt_classes = [c for c in sample.class_names if c != 'background']
                if not gt_classes:
                    gt_classes = [c for c in VOC_CLASSES if c != 'background']
                
                filt_maps, filt_cams = salience_dropout(
                    model, blip_wrapper, image, gt_classes, device, 
                    config['experiment']['drop_iterations']
                )

                # 4. Post-process (Thresholding, CRF)
                pred_mask = postprocess_spatial_maps(
                    filt_maps, 
                    original_image=sample.image.copy(), 
                    config=config
                )"""

replacement = """                # PASS 1: Ambient 20-class Full Prompt Prediction
                all_classes = [c for c in VOC_CLASSES if c != 'background']
                ambient_maps, ambient_cams = salience_dropout(
                    model, blip_wrapper, image, all_classes, device, 
                    config['experiment']['drop_iterations']
                )

                # PASS 2: Ground-Truth Filtered Predictions
                gt_classes = [c for c in sample.class_names if c != 'background']
                if not gt_classes:
                    gt_classes = all_classes # Fallback
                
                filt_maps, filt_cams = salience_dropout(
                    model, blip_wrapper, image, gt_classes, device, 
                    config['experiment']['drop_iterations']
                )

                # Combine Passes (Full Filter Ensemble logic)
                combined_maps = {}
                for cls_name in gt_classes:
                    if cls_name in ambient_maps and cls_name in filt_maps:
                        # Average the spatial arrays before post-processing
                        combined_maps[cls_name] = (ambient_maps[cls_name] + filt_maps[cls_name]) / 2.0
                    elif cls_name in filt_maps:
                        combined_maps[cls_name] = filt_maps[cls_name]

                # 4. Post-process (Thresholding, CRF)
                pred_mask = postprocess_spatial_maps(
                    combined_maps, 
                    original_image=sample.image.copy(), 
                    config=config
                )"""

text = text.replace(target, replacement)

with open("scripts/evaluate.py", "w") as f:
    f.write(text)

import yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

config['patching']['type'] = 'superpixel'

with open("config.yaml", "w") as f:
    yaml.dump(config, f)
