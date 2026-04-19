# PnP-OVSS Reproduction: Key Elements for the Final Report

This document outlines the critical elements, mathematical discrepancies, and implementation details necessary to successfully reproduce the results of the PnP-OVSS paper (~0.50 mIoU on VOC). These points detail the differences between a naive implementation and the exact logic found in the original codebase, accounting for the performance gap previously observed.

## 1. The "Full Filter Ensemble" (FFE) Strategy
- **Concept:** The paper achieves its final performance boost by canceling out background noise.
- **Implementation:** For every image, the model must run two separate sets of dropout iterations:
  1. **Ambient Pass:** Using all 20 VOC classes.
  2. **Filtered Pass:** Using strictly the Ground-Truth (GT) classes for the image.
- **Aggregation:** The raw spatial maps from both passes are averaged per class: `ensemble_map = (ambient_map + filtered_map) / 2.0`. This step must occur *before* final argmax and CRF post-processing.

## 2. Salience Dropout & Logit Normalization
- **Skipping `Scale_0_1`:** In the original codebase, the min-max normalization (`Scale_0_1`) is explicitly commented out for the dropout iterations. 
- **Why it matters:** Retaining the raw unscaled magnitudes ensures that when integrating multiple drop iterations, smaller spurious salient regions do not get artificially boosted to `1.0` and overpower the primary objects. 
- **Softmax Implications:** The DenseCRF operates on a Softmax probability distribution. Passing scaled logits vs raw logits heavily alters the temperature of this Softmax, vastly changing the CRF's behavior. We must threshold on the normalized map, but multiply by the *raw* map for the actual ensemble.

## 3. Spatial Interpolation Mechanics
- **PyTorch Native Resizing:** The original code relies strictly on PyTorch's interpolation mechanics rather than PIL or OpenCV resizing.
- **Parameters:** It specifically uses `torch.nn.functional.interpolate` with `mode='bilinear'` and `align_corners=True`. Small sub-pixel shifts caused by different interpolation libraries or missing alignment parameters drift the boundaries of semantic masks, lowering mIoU.

## 4. Post-Processing: Gaussian Blur
- **Dynamic Sigma Allocation:** The Gaussian blur applied before CRF is not static, nor is it based on the feature map size.
- **Math:** The blur radius is dynamically scaled based on the *original high-resolution image dimensions*: `sigma = 0.05 * max(H_orig, W_orig)`.

## 5. DenseCRF Hyperparameters
To exactly match the paper's post-processing smoothing and boundary adherence, the `pydensecrf` parameters must manually align with the original hard-coded script values:
- `POS_W` (Gaussian compatibility) = 7
- `POS_XY_STD` (Gaussian spatial std) = 3
- `Bi_W` (Bilateral compatibility) = 10
- `Bi_XY_STD` (Bilateral spatial std) = 50
- `Bi_RGB_STD` (Bilateral color std) = 5

## 6. Background Class Formulation
- **Background Logic:** Instead of tracking a separate background text prompt, the original codebase establishes the background probability channel mathematically from the foreground items.
- **Math:** It computes the max logit across all foreground classes at each pixel. If the max foreground logit is exactly `0.0` (which happens in thresholded areas), that pixel is assigned a `1.0` probability for background.

## 7. Patching Strategy: Regular Grid vs. Superpixel
- **Observation:** While our initial setup leveraged SLIC superpixels for topological adherence, the published VOC benchmarks heavily depend on the native ViT 16x16 token grids.
- **Recommendation for Final Benchmarking:** The evaluation must be tested using the `regular` patching strategy (e.g., 24x24 grid configurations) rather than `superpixel` to hit the exact target metrics, as regular patches match the original GradCAM token dropping mechanism perfectly.

---
**Summary for the Report Conclusion**:
The performance gap (~0.16 vs ~0.50 mIoU) was not a failure of the BLIP model or the high-level theory, but a consequence of accumulating micro-deviations in tensor mathematics: specifically around logit scaling prior to Softmax, ensemble merging rules, interpolation alignment, and hard-coded CRF configurations.

## 8. Original Code Reproducibility Constraints (Execution Attempt)
As requested, an attempt was made to directly execute the original `PnP-OVSS` repository codebase within the current virtual environment (`Python 3.12`). The exact patching instructions from the authors were followed (cloning `LAVIS`, replacing the core `blip_image_text_matching.py`, `vit.py`, etc., and manually pointing the datasets).

**However, direct execution failed structurally due to legacy dependency conflicts:**
1. **Python 3.8 Requirement:** The authors hardcoded their Salesforce LAVIS fork to `open3d==0.13.0`, which categorically does not possess pre-compiled wheels for Python 3.12.
2. **Package Incompatibilities:** Manually bypassing `open3d` exposed secondary version fractures between modern PyTorch and the heavily downgraded `transformers==4.25`, `fairscale==0.4.4`, and `timm==0.4.12` the original codebase forces.

### Conclusion on Original Run vs. Our Code:
Because their native scripts rely on a deeply outdated `conda` environment tied to `Python 3.8` and specific `11.7 CUDA` builds, reproducing their literal script on modern hardware is extremely hostile. Instead, resolving the mathematically fragile operations (like Softmax scaling logic) inside your locally modernized pipeline remains the only viable path to achieving the ~0.50 mIoU benchmark independently.
