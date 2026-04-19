"""
Hyperparameter tuning with CLIP reward (§3.4 of PnP-OVSS)
===========================================================
For each (layer, head) combination, runs GradCAM on every image in the
tuning set, then scores each mask with the CLIP-based reward (equations 6–7).

The reward is annotation-free: it only needs image-level class labels
(which classes appear in each image), not pixel-level masks.

    Reward(layer, head) = Σ_{images} Σ_{k∈K(I)}
                          𝟙[ Pr_CLIP(masked_image, k) > Pr_CLIP(black, k) ]

The (layer, head) pair with the highest total reward is the winner.
Results are written progressively to CSV so an interrupted run is safe.

Mask generation (simplified):
  1. Compute GradCAM salience → flat [P*P] vector
  2. Normalise to [0, 1]
  3. Binarize at threshold 0.5  → {0, 1} mask at patch resolution
  CLIP reward is computed directly on this binary patch-resolution mask
  (clip_reward._apply_mask handles upsampling to image resolution).

Usage
-----
    python3 scripts/tune_hyperparams.py                # uses config.yaml
    python3 scripts/tune_hyperparams.py --config config.yaml
    python3 scripts/tune_hyperparams.py --no_clip      # salience metrics only
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
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.blip_wrapper import BLIPWrapper
from utils.config import load_config
from utils.data_loader import format_prompt, get_class_token_indices
from utils.dataset import build_dataset
from utils.postprocess import save_mask_overlay
from core.gradcam import compute_gradcam_salience
from core.clip_reward import _apply_mask


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PnP-OVSS hyperparameter tuning with CLIP reward.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",   default="config.yaml")
    p.add_argument("--no_clip",  action="store_true",
                   help="Skip CLIP reward (salience metrics only — faster).")
    p.add_argument("--verbose",  action="store_true",
                   help="Print full tracebacks on errors.")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    cfg      = load_config(args.config)
    tune_cfg = cfg["tuning"]
    ds_cfg   = cfg["dataset"]
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

    P = wrapper.num_patches_per_side
    print(
        f"\nModel   : {num_layers} layers × {num_heads} heads | "
        f"{wrapper.image_size}px | {wrapper.patch_size}px patches ({P}×{P} grid)"
    )
    print(f"Mask    : GradCAM → normalize → binarize @0.5 ({P}×{P})")
    print(f"Combos  : {total}")

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

    # ── Build mega-batch (done ONCE, reused for every combo) ──────────────
    # All N (image, class) pairs are stacked into a single tensor so each
    # layer needs exactly one BLIP forward+backward call instead of N.
    import torch
    print("Preprocessing mega-batch...")

    all_pairs: list[tuple[str, object]] = []   # (iid, sample) in stable order
    for iid, group in image_groups.items():
        for s in group["samples"]:
            all_pairs.append((iid, s))
    N = len(all_pairs)

    # Text: tokenize all prompts together → uniform padding across all N
    all_prompts = [format_prompt(s.class_name) for _, s in all_pairs]
    mega_text   = wrapper.preprocess_text(all_prompts)   # [N, max_text_len]

    # Image: preprocess each unique image once, then expand for its K classes
    pv_parts: list[torch.Tensor] = []
    for iid, group in image_groups.items():
        K  = len(group["samples"])
        pv = wrapper.preprocess_image(group["image"])    # [1, 3, H, W]
        pv_parts.append(pv.expand(K, -1, -1, -1).contiguous())
    mega_pv = torch.cat(pv_parts, dim=0)                 # [N, 3, H, W]

    # Class token indices per sample (computed on individual, unpadded prompts)
    all_cls_idx = [
        get_class_token_indices(wrapper.processor, format_prompt(s.class_name), s.class_name)
        for _, s in all_pairs
    ]

    blip_batch_size = tune_cfg.get("blip_batch_size", 4)
    n_chunks = (N + blip_batch_size - 1) // blip_batch_size
    print(f"  → {N} samples  |  BLIP batch_size={blip_batch_size}  ({n_chunks} chunks/layer)")

    # ── Pre-compute CLIP embeddings (constant across all combos) ─────────
    # Text embeddings: one vector per unique class name, never changes.
    # Black embeddings: one vector per unique image, never changes.
    # Per-combo we only need to encode the masked images (one batched call).
    text_embeds: dict = {}
    black_embeds: dict[str, torch.Tensor] = {}
    if clip_reward is not None:
        print("Precomputing CLIP text embeddings...")
        unique_classes = list({s.class_name for s in samples})
        text_embeds = clip_reward.precompute_text_embeddings(unique_classes)
        print(f"  → {len(text_embeds)} class embeddings")

        print("Precomputing CLIP black-image embeddings...")
        black_images = [Image.new("RGB", g["image"].size, (0, 0, 0))
                        for g in image_groups.values()]
        black_vecs   = clip_reward.encode_images(black_images)  # [n_images, D]
        black_embeds = {iid: black_vecs[i]
                        for i, iid in enumerate(image_groups)}
        print(f"  → {len(black_embeds)} black-image embeddings\n")

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
    # Outer loop: layer  → ONE BLIP forward+backward for all N samples.
    # Inner loop: head   → CPU-only slicing of the cached attn/grad tensors.
    # This gives 12 BLIP calls total instead of 12×12×50 = 7 200.
    done    = 0
    results = []
    total_possible_reward = len(samples)
    save_overlay_ids      = set(list(image_groups.keys())[:8])

    for layer_idx in range(num_layers):

        # ── BLIP forward+backward in sub-batches, cache attn/grad ────────
        print(f"\nLayer {layer_idx:02d} — forward+backward ({n_chunks} chunks)...",
              end="  ", flush=True)
        t_fwd = time.perf_counter()
        attn_parts: list = []
        grad_parts: list = []
        fwd_error: Exception | None = None
        try:
            for start in range(0, N, blip_batch_size):
                end = min(start + blip_batch_size, N)
                a, g = wrapper.forward_with_gradcam(
                    mega_pv[start:end],
                    mega_text["input_ids"][start:end],
                    mega_text["attention_mask"][start:end],
                    layer_idx,
                )
                if a is None:
                    raise RuntimeError("GradCAM hook did not fire.")
                attn_parts.append(a)
                grad_parts.append(g)
        except Exception as exc:
            fwd_error = exc

        fwd_s = time.perf_counter() - t_fwd

        if fwd_error is not None:
            print(f"ERROR ({fwd_s:.1f}s): {fwd_error}")
            if args.verbose:
                traceback.print_exc()
            for head_idx in range(num_heads):
                done += 1
                row = {"layer": layer_idx, "head": head_idx,
                       "elapsed_s": round(fwd_s / num_heads, 1),
                       "status": f"fwd_error:{fwd_error}"}
                results.append(row)
                writer.writerow({k: row.get(k, "") for k in fieldnames})
            csv_file.flush()
            continue

        import torch as _torch
        attn_mega = _torch.cat(attn_parts, dim=0)   # [N, heads, text_len, img_len]
        grad_mega = _torch.cat(grad_parts, dim=0)
        del attn_parts, grad_parts
        print(f"done ({fwd_s:.1f}s)")

        # ── All heads share the cached attn_mega / grad_mega ─────────────
        for head_idx in range(num_heads):
            done += 1
            tag = f"layer{layer_idx:02d}_head{head_idx:02d}"
            print(f"[{done:3d}/{total}]  {tag}", end="  ", flush=True)

            row = {"layer": layer_idx, "head": head_idx}
            t0  = time.perf_counter()

            try:
                sal_maxes: list[float] = []
                coverages: list[float] = []

                # Build masks for every (image, class) pair — CPU only
                group_masks: dict[str, dict[str, np.ndarray]] = {
                    iid: {} for iid in image_groups
                }
                for idx, (iid, s) in enumerate(all_pairs):
                    flat_sal = compute_gradcam_salience(
                        attn_mega[idx:idx+1], grad_mega[idx:idx+1],
                        head_idx, all_cls_idx[idx],
                    )
                    sal = flat_sal.cpu().float()
                    mn, mx = sal.min(), sal.max()
                    sal  = (sal - mn) / (mx - mn) if (mx - mn) > 1e-8 else sal.zero_()
                    mask = (sal.reshape(P, P).numpy() > 0.25).astype(np.float32)
                    group_masks[iid][s.class_name] = mask
                    sal_maxes.append(float(mx.item()))
                    coverages.append(float(mask.mean() * 100))

                # CLIP reward — one batched encode for all masked images
                total_reward = 0
                if clip_reward is not None:
                    masked_imgs:  list = []
                    masked_meta:  list = []   # (iid, class_name)
                    for iid, class_masks in group_masks.items():
                        img = image_groups[iid]["image"]
                        for cls_name, mask in class_masks.items():
                            masked_imgs.append(_apply_mask(img, mask))
                            masked_meta.append((iid, cls_name))

                    # One CLIP encode call for all N masked images
                    masked_embeds = clip_reward.encode_images(masked_imgs)  # [N, D]

                    for i, (iid, cls_name) in enumerate(masked_meta):
                        total_reward += clip_reward.compute_reward_from_embeds(
                            masked_embed = masked_embeds[i],
                            black_embed  = black_embeds[iid],
                            class_name   = cls_name,
                            all_classes  = image_groups[iid]["classes"],
                            text_embeds  = text_embeds,
                        )

                # Save overlays (no CLIP dependency)
                for iid in save_overlay_ids:
                    for cls_name, mask in group_masks.get(iid, {}).items():
                        out_png = masks_dir / f"{tag}_{cls_name}.png"
                        save_mask_overlay(mask, image_groups[iid]["image"], str(out_png))

                elapsed    = time.perf_counter() - t0
                reward_pct = (total_reward / total_possible_reward * 100) if total_possible_reward > 0 else 0.0
                sal_mean   = float(np.mean(sal_maxes)) if sal_maxes else 0.0
                cov_mean   = float(np.mean(coverages)) if coverages else 0.0

                row.update({
                    "clip_reward":        total_reward,
                    "clip_reward_pct":    round(reward_pct, 1),
                    "salience_max_mean":  round(sal_mean, 6),
                    "mask_coverage_mean": round(cov_mean, 2),
                    "elapsed_s":          round(elapsed, 2),
                    "status":             "ok",
                })
                clip_str = f"reward={total_reward}/{total_possible_reward} ({reward_pct:.0f}%)  " if use_clip else ""
                print(f"{clip_str}cov={cov_mean:.1f}%  ({elapsed:.2f}s)")

            except KeyboardInterrupt:
                print("\nInterrupted.")
                results.append(row)
                writer.writerow({k: row.get(k, "") for k in fieldnames})
                csv_file.flush()
                csv_file.close()
                _print_winner(results, ok_results=[], use_clip=use_clip,
                              total_possible_reward=total_possible_reward,
                              done=done, csv_path=csv_path, masks_dir=masks_dir, ROOT=ROOT)
                return
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                print(f"ERROR ({elapsed:.2f}s): {exc}")
                if args.verbose:
                    traceback.print_exc()
                row.update({"elapsed_s": round(elapsed, 2), "status": f"error:{exc}"})

            results.append(row)
            writer.writerow({k: row.get(k, "") for k in fieldnames})
            csv_file.flush()

    csv_file.close()

    ok_results = [r for r in results if r.get("status") == "ok"]
    _print_winner(results, ok_results, use_clip, total_possible_reward,
                  done, csv_path, masks_dir, ROOT)


def _print_winner(results, ok_results, use_clip, total_possible_reward,
                  done, csv_path, masks_dir, ROOT) -> None:
    if not ok_results:
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
