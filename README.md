# PnP-OVSS — Plug-and-Play Open-Vocabulary Semantic Segmentation

Training-free semantic segmentation using BLIP cross-attention, GradCAM salience, and Dense CRF post-processing. Targets ~0.50 mIoU on Pascal VOC 2012 without any task-specific fine-tuning.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Setup](#setup)
3. [Data](#data)
4. [Configuration](#configuration)
5. [Running the Pipeline](#running-the-pipeline)
6. [Output Structure](#output-structure)
7. [Project Structure](#project-structure)

---

## Project Overview

The pipeline processes an image and a set of class names through three steps:

1. **GradCAM salience** — extracts cross-attention gradients from a chosen BLIP layer/head to produce per-patch salience maps.
2. **Salience DropOut** — iteratively masks the highest-salience patches and re-runs the model to reveal less-salient object parts, accumulating scores across passes.
3. **Dense CRF post-processing** — refines segment boundaries using spatial and color consistency.

The pipeline supports two input modes: single-image inference (`main.py`) and batch evaluation on Pascal VOC 2012 (`scripts/evaluate.py`).

---

## Setup

### Prerequisites

- Python 3.12+
- CUDA (optional), MPS (Apple Silicon, optional), or CPU

### 1. Create the virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Dense CRF (optional but recommended)

Dense CRF significantly improves boundary quality. The PyPI build is broken on modern systems; install from source:

```bash
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

If `pydensecrf` is not installed, set `postprocess.use_crf: false` in your config file and the pipeline will skip that step.

---

## Data

### Pascal VOC 2012 (automatic download)

The dataset downloads automatically on first run. It will be saved to:

```
data/voc/VOCdevkit/VOC2012/
  JPEGImages/          — RGB images
  SegmentationClass/   — Semantic segmentation masks (0–20, 255 = ignore boundary)
  ImageSets/Main/      — Per-class train/val image lists
```

To trigger the download, just run any evaluation command (see [Running the Pipeline](#running-the-pipeline)). The download is controlled by `dataset.download: true` in the config.

### Custom folder dataset

For inference on your own images, organise them as follows:

```
data/folder/
  <class_name>/
    img001.jpg
    img001_mask.png    ← optional ground-truth mask (255 = foreground, 0 = background)
  <another_class>/
    ...
```

The class name is inferred from the parent directory name. No masks are required for inference-only use.

### Sample images (already present)

A few sample images are included in `images/` for quick testing with `main.py`.

---

## Configuration

All behaviour is controlled by a single YAML config file. The default is `config.yaml`. Several pre-configured variants are included at the project root.

### Key config sections

```yaml
model:
  name: Salesforce/blip-itm-base-coco   # or blip-itm-large-coco
  device: auto                           # auto | cuda | mps | cpu
  image_size: 336

patching:
  strategy: regular                      # regular | regular_free | superpixel
  grid_size: 14                          # for regular strategies
  n_segments: 200                        # for superpixel strategy

pipeline:
  layer_idx: 7                           # BLIP cross-attention layer (0-indexed)
  head_idx: 9                            # attention head index
  dropout_rounds: 3                      # salience dropout iterations
  patches_per_drop: 10                   # patches masked per iteration

postprocess:
  blur_sigma: 0.05                       # Gaussian blur (fraction of image side)
  threshold: 0.15                        # foreground binarisation threshold
  use_crf: true                          # requires pydensecrf

output:
  results_dir: results                   # where single-inference outputs go
  save_iterations: false                 # save per-pass visualisations

dataset:
  source: voc                            # voc | folder
  split: val
  download: true
  max_images: null                       # null = full dataset
  min_pixels: 500                        # skip very small ground-truth masks
  seed: 42
```

### Config variants included

| File | Description |
|------|-------------|
| `config.yaml` | Working default |
| `config_paper_baseline.yaml` | Validated baseline matching paper results |
| `config_paper_repro.yaml` | Full paper reproduction setup |
| `config_tune.yaml` | Starting point for hyperparameter search |
| `config_tune_result.yaml` | Best parameters found by tuning |
| `config_l7h9_filt_blur.yaml` | Layer 7, head 9 with filtering and blur |
| `config_superpixel_tune.yaml` | Superpixel strategy tuning |

---

## Running the Pipeline

All commands use the `run.sh` launcher. It activates `.venv` automatically and forwards extra CLI arguments to the underlying script.

```bash
./run.sh <mode> [config.yaml] [extra args...]
```

If no config is specified, `config.yaml` is used. You can override any config key from the command line with `--key value`.

### Modes

#### Single-image inference

Runs the pipeline on one image from `images/` and saves a visualisation.

```bash
./run.sh infer config.yaml
```

Output: `results/<run_slug>/overlay.png` and `results/<run_slug>/salience.npy`

#### Batch evaluation on Pascal VOC

Runs the full pipeline over the dataset split defined in the config and computes IoU metrics.

```bash
./run.sh eval config.yaml
# with a size limit for quick testing:
./run.sh eval config.yaml --max_images 50
```

Output: `experiments/eval_<run_slug>/results.csv` and `experiments/eval_<run_slug>/summary.txt`

#### Hyperparameter tuning — layer and head (annotation-free)

Grid-searches BLIP cross-attention layer/head combinations using a CLIP-based reward. Does not require segmentation ground truth.

```bash
./run.sh tune config_tune.yaml
```

Output: `experiments/<run_slug>/tune_results.csv`

#### Hyperparameter tuning — pipeline parameters

Searches over patching strategy, grid size, dropout rounds, and patches-per-drop using Dice/mIoU on Pascal VOC.

```bash
./run.sh tune_pipeline config_tune.yaml
```

Output: `experiments/config_tune/pipeline_results.csv`

---

## Output Structure

### Single inference

```
results/
  <run_slug>/
    overlay.png          — 3-panel image: original | salience map | overlay
    salience.npy         — raw accumulated salience scores [N patches]
    iterations/          — (if save_iterations: true)
      pass_01.png
      pass_02.png
      ...
```

### Batch evaluation

```
experiments/
  eval_<run_slug>/
    results.csv          — per-sample rows: image_id, class_name, iou, dice, ...
    summary.txt          — mIoU per class + overall statistics
    masks/               — (if --save_masks) overlay PNG per sample
```

### Tuning

```
experiments/
  <run_slug>/
    tune_results.csv          — (layer, head) → CLIP reward score
  config_tune/
    pipeline_results.csv      — (strategy, params) → Dice / mIoU
```

### Run slug format

Output directories are named by a slug that encodes the key hyperparameters:

```
<class>_L<layer>_H<head>_<patch_tag>_dr<rounds>_pd<per_drop>_sig<sigma>[_crf]

Example: elephant_L6_H7_reg_dr3_pd10_sig0.05_crf
```

---

## Project Structure

```
.
├── main.py                    — single-image inference entry point
├── run.sh                     — unified launcher for all modes
├── config.yaml                — default configuration
├── config_*.yaml              — experimental config variants
├── requirements.txt           — Python dependencies
│
├── core/
│   ├── salience_dropout.py    — main inference loop (dropout iterations)
│   ├── gradcam.py             — GradCAM salience extraction from BLIP cross-attention
│   ├── patch_strategy.py      — regular grid / superpixel segment management
│   └── clip_reward.py         — CLIP reward for annotation-free layer/head tuning
│
├── models/
│   └── blip_wrapper.py        — BLIP model wrapper (preprocessing, hooks, masking)
│
├── scripts/
│   ├── evaluate.py            — batch evaluation on Pascal VOC, outputs CSV + summary
│   ├── tune_hyperparams.py    — layer/head grid search via CLIP reward
│   └── tune_pipeline.py       — pipeline parameter search via Dice/mIoU
│
├── utils/
│   ├── config.py              — config loading, merging, slug generation
│   ├── data_loader.py         — image loading, prompt formatting, token indexing
│   ├── dataset.py             — PascalVOCDataset and FolderDataset wrappers
│   ├── postprocess.py         — Gaussian blur, thresholding, Dense CRF
│   └── visualize.py           — per-iteration 3-panel visualisations
│
├── data/                      — datasets (auto-downloaded or user-provided)
│   └── voc/VOCdevkit/VOC2012/
├── images/                    — sample images for single inference
├── results/                   — single-inference outputs (gitignored)
└── experiments/               — batch evaluation and tuning outputs (gitignored)
```
