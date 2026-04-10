"""
Hyperparameter tuning with CLIP reward (§3.4 of PnP-OVSS)
===========================================================
For each (layer, head) combination, runs GradCAM (no Salience DropOut,
matching the paper's efficiency note) on every image in the tuning set,
then scores each mask with the CLIP-based reward (equations 6–7).

The reward is annotation-free: it only needs image-level class labels
(which classes appear in each image), not pixel-level masks.

    Reward(layer, head) = Σ_{images} Σ_{k∈K(I)}
                          𝟙[ Pr_CLIP(masked_image, k) > Pr_CLIP(black, k) ]

The (layer, head) pair with the highest total reward is the winner.
Results are written progressively to CSV so an interrupted run is safe.

Usage
-----
    python3 scripts/tune_hyperparams.py                # uses config.yaml
    python3 scripts/tune_hyperparams.py --config config.yaml
    python3 scripts/tune_hyperparams.py --no_clip      # salience metrics only
    python3 scripts/tune_hyperparams.py --no_crf       # skip Dense CRF
    python3 scripts/tune_hyperparams.py --verbose
    ./run.sh tune
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.blip_wrapper import BLIPWrapper
from utils.config import load_config
from utils.data_loader import format_prompt, get_class_token_indices
from utils.dataset import build_dataset
from utils.postprocess import postprocess, save_mask_overlay
from core.gradcam import compute_gradcam_salience
from core.patch_strategy import build_strategy
from utils.visualize import save_patch_overview


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PnP-OVSS hyperparameter tuning with CLIP reward.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",   default="config.yaml")
    p.add_argument("--no_clip",  action="store_true",
                   help="Skip CLIP reward (salience metrics only — faster).")
    p.add_argument("--no_crf",   action="store_true",
                   help="Skip Dense CRF during mask generation.")
    p.add_argument("--verbose",  action="store_true",
                   help="Print full tracebacks on errors.")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    cfg      = load_config(args.config)
    tune_cfg = cfg["tuning"]
    pipe_cfg = cfg["pipeline"]
    pp_cfg   = cfg["postprocess"]
    ds_cfg   = cfg["dataset"]
    use_crf  = tune_cfg.get("use_crf", True) and not args.no_crf
    use_clip = not args.no_clip

    out_dir   = ROOT / tune_cfg["out_dir"]
    masks_dir = out_dir / "tune_masks"
    csv_path  = out_dir / "tune_results.csv"
    masks_dir.mkdir(parents=True, exist_ok=True)

    # ── BLIP model ───────────────────────────────────────────────────────
    import torch
    model_cfg = cfg["model"]
    device    = None if model_cfg["device"] == "auto" else torch.device(model_cfg["device"])
    wrapper   = BLIPWrapper(model_name=model_cfg["name"], device=device)
    num_layers, num_heads = wrapper.num_text_layers, wrapper.num_heads
    total     = num_layers * num_heads

    print(
        f"\nModel   : {num_layers} layers × {num_heads} heads | "
        f"{wrapper.image_size}px | {wrapper.patch_size}px patches"
    )
    print(f"Patching: {cfg['patching']['type']}")
    print(f"Combos  : {total}   CRF: {'on' if use_crf else 'off'}")

    # ── CLIP reward model ─────────────────────────────────────────────────
    clip_reward = None
    if use_clip:
        from core.clip_reward import CLIPReward
        clip_cfg    = cfg.get("clip", {})
        clip_reward = CLIPReward(
            model_name = clip_cfg.get("model", "openai/clip-vit-large-patch14-336"),
            device     = device,
        )

    # ── Tuning dataset ────────────────────────────────────────────────────
    # Use a small cap so tuning stays fast; override via config tuning.max_images
    tune_max    = tune_cfg.get("max_images", 50)
    tune_ds_cfg = {**ds_cfg, "max_images": tune_max}
    print(f"\nDataset : {ds_cfg['name']}  max_images={tune_max}")
    dataset  = build_dataset(tune_ds_cfg)
    samples  = list(dataset)           # load all into memory once
    print(f"  → {len(samples)} (image, class) samples")

    # Pre-compute per-sample: preprocess inputs + all classes per image
    # Group by image_id so we pass all K(I) classes together to CLIP
    from collections import defaultdict
    image_groups: dict[str, dict] = {}
    for s in samples:
        iid = s.image_id.rsplit("_", 1)[0]   # strip class suffix to get image id
        if iid not in image_groups:
            image_groups[iid] = {
                "image":   s.image,
                "classes": [],
                "samples": [],
            }
        image_groups[iid]["classes"].append(s.class_name)
        image_groups[iid]["samples"].append(s)

    print(f"  → {len(image_groups)} unique images\n")
    print("=" * 72)

    # ── CSV ───────────────────────────────────────────────────────────────
    fieldnames = [
        "layer", "head",
        "clip_reward", "clip_reward_pct",
        "salience_max_mean", "mask_coverage_mean",
        "elapsed_s", "status",
    ]
    csv_file = open(csv_path, "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()

    # ── Grid search ──────────────────────────────────────────────────────
    done    = 0
    results = []
    _overview_saved = False
    total_possible_reward = len(samples)   # one point per (image, class) pair
    save_overlay_ids = set(list(image_groups.keys())[:8])

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            done += 1
            tag = f"layer{layer_idx:02d}_head{head_idx:02d}"
            print(f"[{done:3d}/{total}]  {tag}", end="  ", flush=True)

            row  = {"layer": layer_idx, "head": head_idx}
            t0   = time.perf_counter()

            try:
                total_reward      = 0
                sal_maxes         = []
                coverages         = []

                for iid, group in image_groups.items():
                    image      = group["image"]
                    all_classes = group["classes"]

                    # Preprocess once per image per combo (BLIP)
                    inputs = wrapper.preprocess(image, format_prompt(all_classes[0]))
                    pixel_values   = inputs["pixel_values"]
                    input_ids_base = inputs["input_ids"]
                    attn_mask_base = inputs["attention_mask"]

                    # Build patch strategy (image-specific for superpixels)
                    strategy = build_strategy(cfg["patching"], wrapper, image)
                    P        = wrapper.num_patches_per_side

                    # Save one patch-layout overview for the whole run (first image only)
                    if not _overview_saved:
                        save_patch_overview(
                            image, strategy,
                            out_dir / "patch_overview.png",
                            title=f"{ds_cfg['name']} — {type(strategy).__name__} ({strategy.num_segments} segments)",
                        )
                        _overview_saved = True

                    # One GradCAM pass per class (paper: K passes per image)
                    class_masks: dict[str, np.ndarray] = {}
                    for s in group["samples"]:
                        prompt = format_prompt(s.class_name)
                        inp    = wrapper.preprocess(image, prompt)
                        cls_idx = get_class_token_indices(
                            wrapper.processor, prompt, s.class_name
                        )

                        attn, attn_grad = wrapper.forward_with_gradcam(
                            inp["pixel_values"],
                            inp["input_ids"],
                            inp["attention_mask"],
                            layer_idx,
                        )
                        if attn is None:
                            continue

                        flat_sal = compute_gradcam_salience(
                            attn, attn_grad, head_idx, cls_idx
                        )
                        seg_scores  = strategy.aggregate(flat_sal)
                        spatial_map = strategy.to_spatial(seg_scores)

                        mask = postprocess(
                            spatial_map    = spatial_map,
                            original_image = image,
                            gaussian_sigma = pp_cfg["gaussian_sigma"],
                            use_dense_crf  = use_crf,
                        )
                        class_masks[s.class_name] = mask
                        sal_maxes.append(float(seg_scores.max().item()))
                        coverages.append(float((mask > 0.5).mean() * 100))

                    # CLIP reward: batch all classes in this image at once
                    if clip_reward is not None and class_masks:
                        rewards = clip_reward.compute_batch_reward(
                            image       = image,
                            masks       = class_masks,
                            all_classes = all_classes,
                        )
                        total_reward += sum(rewards.values())

                    # Save mask overlays for first 8 images (for inspection)
                    if iid in save_overlay_ids:
                        for cls_name, mask in class_masks.items():
                            out_png = masks_dir / f"{tag}_{cls_name}.png"
                            save_mask_overlay(mask, image, str(out_png))

                elapsed     = time.perf_counter() - t0
                reward_pct  = (total_reward / total_possible_reward * 100) if total_possible_reward > 0 else 0.0
                sal_mean    = float(np.mean(sal_maxes)) if sal_maxes else 0.0
                cov_mean    = float(np.mean(coverages)) if coverages else 0.0

                row.update({
                    "clip_reward":        total_reward,
                    "clip_reward_pct":    round(reward_pct, 1),
                    "salience_max_mean":  round(sal_mean, 6),
                    "mask_coverage_mean": round(cov_mean, 2),
                    "elapsed_s":          round(elapsed, 1),
                    "status":             "ok",
                })
                clip_str = f"reward={total_reward}/{total_possible_reward} ({reward_pct:.0f}%)  " if use_clip else ""
                print(
                    f"{clip_str}"
                    f"cov={cov_mean:.1f}%  "
                    f"({elapsed:.1f}s)"
                )

            except KeyboardInterrupt:
                print("\nInterrupted.")
                results.append(row)
                writer.writerow({k: row.get(k, "") for k in fieldnames})
                csv_file.flush()
                break
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                print(f"ERROR ({elapsed:.1f}s): {exc}")
                if args.verbose:
                    traceback.print_exc()
                row.update({"elapsed_s": round(elapsed, 1), "status": f"error:{exc}"})

            results.append(row)
            writer.writerow({k: row.get(k, "") for k in fieldnames})
            csv_file.flush()

        else:
            continue
        break

    csv_file.close()

    # ── Print winner ─────────────────────────────────────────────────────
    ok_results = [r for r in results if r.get("status") == "ok"]
    print(f"\n{'='*72}")
    print(f"Tuning done.  {len(ok_results)} / {done} combinations succeeded.")

    if ok_results and use_clip:
        best = max(ok_results, key=lambda r: (r.get("clip_reward", 0), r.get("mask_coverage_mean", 0)))
        print(f"\n  ★  Best by CLIP reward:")
        print(f"       Layer = {best['layer']}   Head = {best['head']}")
        print(f"       CLIP reward = {best['clip_reward']} / {total_possible_reward} "
              f"({best['clip_reward_pct']}%)")
        print(f"\n  → Update config.yaml:  layer: {best['layer']}   head: {best['head']}")
        print(f"  → Or run directly: python3 main.py --layer {best['layer']} --head {best['head']}")
    elif ok_results:
        best = max(ok_results, key=lambda r: r.get("mask_coverage_mean", 0))
        print(f"\n  Best by coverage (no CLIP):  layer={best['layer']}  head={best['head']}")

    print(f"\n  CSV   : {csv_path.relative_to(ROOT)}")
    print(f"  Masks : {masks_dir.relative_to(ROOT)}/")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
