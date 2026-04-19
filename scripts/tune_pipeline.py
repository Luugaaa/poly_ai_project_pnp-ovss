"""
Pipeline hyperparameter tuning with Dice/IoU on Pascal VOC
==========================================================
Searches over Salience DropOut parameters for both patching strategies,
keeping layer and head fixed (set in config.yaml pipeline section).

Search space
------------
    strategy       : regular | regular_free | superpixel
    grid_size      : (regular, regular_free) coarse grid G×G
    n_segments     : (superpixel only) SLIC target segment count
  patches_per_drop: segments removed per dropout round
  dropout_rounds : 0 (GradCAM only) … dropout_rounds_max

Key efficiency gains
--------------------
  * Vision encoder (ViT-L, the dominant cost) runs with no_grad on every pass —
    its activations are never stored for backward, eliminating the ViT backward.
    Gradients flow only through the text encoder and ITM head (~2-3× speedup).
  * Vision embeddings for unmasked images are cached ONCE before the outer loop
    and reused across all (strategy, segment_param) combos and across all class
    queries on the same image.
  * Pass-0 (no dropout) reads from the cache entirely — zero ViT calls.
    * Dice or mIoU on Pascal VOC is used as the tuning metric; no CLIP model is loaded.

Usage
-----
    python3 scripts/tune_pipeline.py                # uses config.yaml
    python3 scripts/tune_pipeline.py --config config.yaml
    python3 scripts/tune_pipeline.py --verbose
    ./run.sh tune_pipeline
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
from core.patch_strategy import RegularPatchStrategy, RegularFreePatchStrategy, SuperpixelPatchStrategy


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PnP-OVSS pipeline hyperparameter tuning (Dice/IoU on VOC).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",  default="config.yaml")
    p.add_argument("--verbose", action="store_true",
                   help="Print full tracebacks on errors.")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _blip_pass(
    wrapper,
    pixel_values_batch,           # [N, 3, H, W]
    text_inputs: dict,
    layer_idx: int,
    head_idx: int,
    all_pairs: list,
    all_cls_idx: list,
    strategies_per_image: dict,
    blip_batch_size: int,
    embed_cache: Optional[dict] = None,  # iid → Tensor [1, img_len, D]
) -> dict:
    """
    Run one BLIP pass (chunked) and return per-sample segment scores.

    If ``embed_cache`` is provided, the vision encoder is skipped and
    pre-computed embeddings are used (pass-0, where all image copies are
    identical).  Otherwise the vision encoder runs on ``pixel_values_batch``
    with no_grad before each text-encoder chunk.

    Gradients flow only through the text encoder + ITM head in both cases.

    Returns
    -------
    seg_scores : dict  (iid, class_name) → Tensor [N_segs]
    """
    import torch
    N = pixel_values_batch.shape[0]
    seg_scores: dict = {}

    for start in range(0, N, blip_batch_size):
        end = min(start + blip_batch_size, N)
        B   = end - start

        # Vision: use cache or encode fresh (always no_grad)
        if embed_cache is not None:
            emb_parts = [embed_cache[all_pairs[start + i][0]] for i in range(B)]
            image_embeds = torch.cat(emb_parts, dim=0)   # [B, img_len, D]
        else:
            image_embeds = wrapper.encode_image_embeds(
                pixel_values_batch[start:end]
            )

        # Text encoder + ITM head (grad flows here only)
        attn, grad = wrapper.forward_with_gradcam_from_embeds(
            image_embeds,
            text_inputs["input_ids"][start:end],
            text_inputs["attention_mask"][start:end],
            layer_idx,
        )
        if attn is None:
            continue

        for i in range(B):
            idx      = start + i
            iid, s   = all_pairs[idx]
            strategy = strategies_per_image[iid]
            flat_sal = compute_gradcam_salience(
                attn[i:i+1], grad[i:i+1], head_idx, all_cls_idx[idx]
            )
            seg_scores[(iid, s.class_name)] = strategy.aggregate(flat_sal)

    return seg_scores


def _build_masks(
    accumulated: dict,
    strategies_per_image: dict,
    threshold: float,
    image_groups: dict,
    gaussian_sigma: float,
    use_blur: bool,
    use_dense_crf: bool,
) -> dict:
    """Convert accumulated segment scores to binary masks using full pipeline postprocessing."""
    from utils.postprocess import postprocess
    group_masks: dict = {}
    for (iid, cls_name), accum in accumulated.items():
        strategy = strategies_per_image[iid]
        spatial  = strategy.to_spatial(accum)
        spatial  = spatial.cpu().numpy() if hasattr(spatial, "cpu") else spatial
        
        # Normalize and apply the binarization threshold as described in the paper
        mn, mx = float(spatial.min()), float(spatial.max())
        sal = (spatial - mn) / (mx - mn) if mx - mn > 1e-8 else np.zeros_like(spatial)
        sal_bin = (sal > threshold).astype(np.float32)
        
        mask = postprocess(
            spatial_map=sal_bin,
            original_image=image_groups[iid]["image"],
            gaussian_sigma=gaussian_sigma,
            use_blur=use_blur,
            use_dense_crf=use_dense_crf,
        )
        
        # Binarize output in case dense crf was inactive
        group_masks.setdefault(iid, {})[cls_name] = (mask > 0.5).astype(np.float32)
    return group_masks


def _mask_scores(
    group_masks: dict,
    image_groups: dict,
) -> Tuple[float, float, float]:
    """
    Compute mean IoU, mean Dice and mean mask coverage across all pairs.

    Returns
    -------
    (miou, mdice, coverage_mean_pct)
    """
    ious: List[float] = []
    dices: List[float] = []
    coverages: List[float] = []

    for iid, class_masks in group_masks.items():
        gt_masks = image_groups[iid].get("gt_masks", {})
        for cls_name, mask in class_masks.items():
            coverages.append(float(mask.mean() * 100))
            gt = gt_masks.get(cls_name)
            if gt is None:
                continue

            # Resize pred to GT resolution if needed
            if mask.shape != gt.shape:
                mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
                mask_rs = np.array(
                    mask_pil.resize((gt.shape[1], gt.shape[0]), Image.BILINEAR),
                    dtype=np.float32,
                ) / 255.0
            else:
                mask_rs = mask

            pred_bin = mask_rs > 0.5
            inter = np.logical_and(pred_bin, gt).sum()
            union = np.logical_or(pred_bin, gt).sum()
            iou = float(inter) / float(union) if union > 0 else (1.0 if inter == 0 else 0.0)
            denom = pred_bin.sum() + gt.sum()
            dice = float(2 * inter) / float(denom) if denom > 0 else 1.0
            ious.append(iou)
            dices.append(dice)

    miou = float(np.mean(ious)) if ious else 0.0
    mdice = float(np.mean(dices)) if dices else 0.0
    cov = float(np.mean(coverages)) if coverages else 0.0
    return miou, mdice, cov


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    cfg    = load_config(args.config)
    tp_cfg = cfg.get("tune_pipeline", {})
    ds_cfg = cfg["dataset"]

    out_dir   = ROOT / tp_cfg.get("out_dir", "experiments")
    masks_dir = out_dir / "pipeline_masks"
    csv_path  = out_dir / "tune_pipeline_results.csv"
    masks_dir.mkdir(parents=True, exist_ok=True)

    # ── BLIP model ───────────────────────────────────────────────────────
    import torch
    model_cfg = cfg["model"]
    device    = None if model_cfg["device"] == "auto" else torch.device(model_cfg["device"])
    wrapper   = BLIPWrapper(
        model_name=model_cfg["name"],
        device=device,
        input_size=model_cfg.get("image_size"),
    )

    layer_idx = cfg["pipeline"]["layer"]
    head_idx  = cfg["pipeline"]["head"]
    P         = wrapper.num_patches_per_side

    print(
        f"\nModel   : {wrapper.num_text_layers} layers × {wrapper.num_heads} heads | "
        f"{wrapper.image_size}px | {wrapper.patch_size}px patches ({P}×{P})"
    )
    print(f"Fixed   : layer={layer_idx}  head={head_idx}")

    # ── Dataset ───────────────────────────────────────────────────────────
    tune_max    = tp_cfg.get("max_images", 50)
    tune_ds_cfg = {**ds_cfg, "max_images": tune_max}
    print(f"\nDataset : {ds_cfg['name']}  max_images={tune_max}")
    dataset = build_dataset(tune_ds_cfg)
    samples = list(dataset)
    print(f"  → {len(samples)} (image, class) samples")

    # Group samples by image; store GT masks
    image_groups: dict = {}
    for s in samples:
        iid = s.image_id.rsplit("_", 1)[0]
        if iid not in image_groups:
            image_groups[iid] = {
                "image":    s.image,
                "classes":  [],
                "samples":  [],
                "gt_masks": {},
            }
        image_groups[iid]["classes"].append(s.class_name)
        image_groups[iid]["samples"].append(s)
        if s.gt_mask is not None:
            image_groups[iid]["gt_masks"][s.class_name] = s.gt_mask

    total_pairs = len(samples)
    n_with_gt   = sum(
        1 for g in image_groups.values()
        for cls in g["classes"] if cls in g["gt_masks"]
    )
    print(
        f"  → {len(image_groups)} unique images  |  "
        f"{n_with_gt}/{total_pairs} pairs have GT masks\n"
    )
    print("=" * 72)

    # ── Mega-batch inputs ─────────────────────────────────────────────────
    print("Preprocessing mega-batch...")
    all_pairs: list = []
    for iid, group in image_groups.items():
        for s in group["samples"]:
            all_pairs.append((iid, s))
    N = len(all_pairs)

    all_prompts = [format_prompt(s.class_name) for _, s in all_pairs]
    mega_text   = wrapper.preprocess_text(all_prompts)

    pv_parts = []
    for iid, group in image_groups.items():
        K  = len(group["samples"])
        pv = wrapper.preprocess_image(group["image"])
        pv_parts.append(pv.expand(K, -1, -1, -1).contiguous())
    mega_pv = torch.cat(pv_parts, dim=0)   # [N, 3, H, W]

    all_cls_idx = []
    for _, s in all_pairs:
        cls_idx_map = get_class_token_indices(
            wrapper.processor, format_prompt(s.class_name), s.class_name
        )
        # GradCAM expects a flat list[int] for the active class tokens.
        cls_idx = cls_idx_map.get(s.class_name)
        if cls_idx is None and cls_idx_map:
            cls_idx = next(iter(cls_idx_map.values()))
        all_cls_idx.append(cls_idx or [])

    blip_batch_size = tp_cfg.get("blip_batch_size", 4)
    n_chunks        = (N + blip_batch_size - 1) // blip_batch_size
    print(f"  → {N} samples  |  blip_batch_size={blip_batch_size}  ({n_chunks} text-chunks/pass)")

    # ── Pre-cache vision embeddings (once for ALL outer combos) ──────────
    # Unique images only — K class queries on the same image all share one
    # ViT call.  This cache is valid for pass-0 of every outer combo.
    vis_batch = tp_cfg.get("vision_batch_size", 16)
    img_ids   = list(image_groups.keys())
    M         = len(img_ids)
    print(f"\nCaching vision embeddings for {M} unique images (vis_batch={vis_batch})...")
    t_cache = time.perf_counter()
    all_pv_unique = torch.cat(
        [wrapper.preprocess_image(image_groups[iid]["image"]) for iid in img_ids],
        dim=0,
    )   # [M, 3, H, W]
    embed_cache: dict = {}
    for start in range(0, M, vis_batch):
        end  = min(start + vis_batch, M)
        embs = wrapper.encode_image_embeds(all_pv_unique[start:end])  # [B, img_len, D]
        for i in range(end - start):
            embed_cache[img_ids[start + i]] = embs[i : i + 1]   # [1, img_len, D]
    del all_pv_unique
    print(f"  → Done in {time.perf_counter()-t_cache:.1f}s\n")

    # ── Search space ──────────────────────────────────────────────────────
    R_max            = tp_cfg.get("dropout_rounds_max", 3)
    patches_per_drop = tp_cfg.get("patches_per_drop", [10, 25, 50])
    threshold        = tp_cfg.get("mask_threshold", 0.25)
    metric_name      = str(tp_cfg.get("metric", "dice")).lower()
    if metric_name not in {"dice", "iou"}:
        raise ValueError(f"Unsupported tune_pipeline.metric '{metric_name}'. Use 'dice' or 'iou'.")

    reg_cfg    = tp_cfg.get("regular", {})
    regf_cfg   = tp_cfg.get("regular_free", {})
    sp_cfg_t   = tp_cfg.get("superpixel", {})
    regular_grid_sizes = reg_cfg.get("grid_sizes",  [8, 12, 24])
    regular_free_grid_sizes = regf_cfg.get("grid_sizes", [])
    sp_n_segments_list = sp_cfg_t.get("n_segments", [100, 200, 400])
    sp_compactness     = sp_cfg_t.get("compactness", 10.0)
    sp_sigma           = sp_cfg_t.get("sigma", 1.0)

    outer_combos: list = (
        [("regular",    f"grid{g}", g) for g in regular_grid_sizes] +
        [("regular_free", f"grid{g}", g) for g in regular_free_grid_sizes] +
        [("superpixel", f"sp{n}",   n) for n in sp_n_segments_list]
    )

    rows_per_outer = 1 + len(patches_per_drop) * R_max
    total_rows     = len(outer_combos) * rows_per_outer
    print(f"Search space : {len(outer_combos)} outer combos × {rows_per_outer} rows = {total_rows} total rows")
    print(
        f"Strategies   : regular {regular_grid_sizes}  |  "
        f"regular_free {regular_free_grid_sizes}  |  superpixel {sp_n_segments_list}"
    )
    print(f"patches/drop : {patches_per_drop}   R_max={R_max}   threshold={threshold}")
    print(f"Metric       : {metric_name}")
    print("=" * 72)

    # ── CSV ───────────────────────────────────────────────────────────────
    fieldnames = [
        "strategy", "segment_param",
        "dropout_rounds", "patches_per_drop",
        "miou", "miou_pct", "mdice", "mdice_pct", "score", "score_pct", "metric",
        "mask_coverage_mean", "elapsed_s", "status",
    ]
    csv_file = open(csv_path, "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()

    results: list        = []
    save_overlay_ids     = set(list(image_groups.keys())[:4])

    # ── Outer loop: (strategy, segment_param) ────────────────────────────
    for strategy_type, seg_label, seg_val in outer_combos:
        print(f"\n{'─'*72}")
        print(f"  {strategy_type.upper()}  segment_param={seg_label}")
        print(f"{'─'*72}")

        # Build one strategy object per image
        strategies_per_image: dict = {}
        try:
            for iid, group in image_groups.items():
                if strategy_type == "regular":
                    strategies_per_image[iid] = RegularPatchStrategy(
                        num_patches_per_side=P,
                        patch_size=wrapper.patch_size,
                        grid_size=seg_val,
                    )
                elif strategy_type == "regular_free":
                    strategies_per_image[iid] = RegularFreePatchStrategy(
                        model_img_size=wrapper.image_size,
                        model_patch_sz=wrapper.patch_size,
                        grid_size=seg_val,
                        device=wrapper.device,
                    )
                else:
                    strategies_per_image[iid] = SuperpixelPatchStrategy(
                        image=group["image"],
                        model_img_size=wrapper.image_size,
                        model_patch_sz=wrapper.patch_size,
                        n_segments=seg_val,
                        compactness=sp_compactness,
                        sigma=sp_sigma,
                        device=wrapper.device,
                    )
        except Exception as exc:
            print(f"  Strategy build failed: {exc}")
            if args.verbose:
                traceback.print_exc()
            continue

        # Pass 0: vision embeddings come from cache — zero ViT calls
        print(f"  Pass 0 (cached vision, {n_chunks} text-chunks)...", end="  ", flush=True)
        t0 = time.perf_counter()
        try:
            seg_scores_0 = _blip_pass(
                wrapper, mega_pv, mega_text, layer_idx, head_idx,
                all_pairs, all_cls_idx, strategies_per_image, blip_batch_size,
                embed_cache=embed_cache,
            )
        except Exception as exc:
            print(f"ERROR: {exc}")
            if args.verbose:
                traceback.print_exc()
            continue
        pass0_s = time.perf_counter() - t0
        print(f"done ({pass0_s:.1f}s)")

        # R=0 baseline (patches_per_drop irrelevant)
        t0 = time.perf_counter()
        try:
            masks_0       = _build_masks(dict(seg_scores_0), strategies_per_image, threshold, image_groups, cfg["postprocess"]["gaussian_sigma"], cfg["postprocess"].get("use_blur", False), cfg["postprocess"]["use_dense_crf"])
            miou_0, mdice_0, cov_0 = _mask_scores(masks_0, image_groups)
            score_0 = mdice_0 if metric_name == "dice" else miou_0
            elapsed       = time.perf_counter() - t0 + pass0_s
            row = dict(
                strategy=strategy_type, segment_param=seg_label,
                dropout_rounds=0, patches_per_drop="n/a",
                miou=round(miou_0, 4), miou_pct=round(miou_0 * 100, 1),
                mdice=round(mdice_0, 4), mdice_pct=round(mdice_0 * 100, 1),
                score=round(score_0, 4), score_pct=round(score_0 * 100, 1), metric=metric_name,
                mask_coverage_mean=round(cov_0, 2),
                elapsed_s=round(elapsed, 1), status="ok",
            )
            print(
                f"  [R=0          ]  mIoU={miou_0:.4f} ({miou_0*100:.1f}%)  "
                f"mDice={mdice_0:.4f} ({mdice_0*100:.1f}%)  "
                f"cov={cov_0:.1f}%  ({elapsed:.1f}s)"
            )
            if save_overlay_ids:
                for iid in save_overlay_ids:
                    for cls_name, mask in masks_0.get(iid, {}).items():
                        save_mask_overlay(
                            mask, image_groups[iid]["image"],
                            str(masks_dir / f"{seg_label}_R0_{cls_name}.png"),
                            gt_mask=image_groups[iid].get("gt_masks", {}).get(cls_name),
                        )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"  [R=0] ERROR: {exc}")
            if args.verbose:
                traceback.print_exc()
            row = dict(
                strategy=strategy_type, segment_param=seg_label,
                dropout_rounds=0, patches_per_drop="n/a",
                elapsed_s=round(elapsed, 1), status=f"error:{exc}",
            )

        results.append(row)
        writer.writerow({k: row.get(k, "") for k in fieldnames})
        csv_file.flush()

        # Inner loop: patches_per_drop → R_max rounds (amortized)
        for pdrop in patches_per_drop:

            accumulated   = {k: v.clone() for k, v in seg_scores_0.items()}
            dropped: dict = {k: set() for k in seg_scores_0}
            remaining: dict = {
                (iid, s.class_name): set(range(strategies_per_image[iid].num_segments))
                for iid, s in all_pairs
                if (iid, s.class_name) in seg_scores_0
            }
            current_scores = {k: v.clone() for k, v in seg_scores_0.items()}

            for r in range(1, R_max + 1):
                t_r = time.perf_counter()
                try:
                    # Build masked pixel values for every sample
                    masked_pv_parts = []
                    for idx, (iid, s) in enumerate(all_pairs):
                        key      = (iid, s.class_name)
                        strategy = strategies_per_image[iid]
                        eff_k    = min(pdrop, max(1, int(0.7 * strategy.num_segments)))
                        new_drops = strategy.top_k(
                            current_scores[key], remaining[key], k=eff_k
                        )
                        dropped[key]   |= new_drops
                        remaining[key] -= new_drops
                        masked_pv_parts.append(
                            strategy.mask_segments(mega_pv[idx : idx + 1], dropped[key])
                        )
                    masked_mega_pv = torch.cat(masked_pv_parts, dim=0)

                    # BLIP pass: ViT runs no_grad on masked images, text with grad
                    new_scores = _blip_pass(
                        wrapper, masked_mega_pv, mega_text, layer_idx, head_idx,
                        all_pairs, all_cls_idx, strategies_per_image, blip_batch_size,
                        embed_cache=None,   # masked images differ → re-encode
                    )

                    # Zero dropped segments, accumulate
                    for key in accumulated:
                        ns = new_scores.get(key)
                        if ns is None:
                            continue
                        ns_clean = ns.clone()
                        for d_idx in dropped[key]:
                            if d_idx < len(ns_clean):
                                ns_clean[d_idx] = 0.0
                        accumulated[key] = accumulated[key] + ns_clean
                    current_scores = new_scores

                    # Evaluate
                    masks_r       = _build_masks(accumulated, strategies_per_image, threshold, image_groups, cfg["postprocess"]["gaussian_sigma"], cfg["postprocess"].get("use_blur", False), cfg["postprocess"]["use_dense_crf"])
                    miou_r, mdice_r, cov_r = _mask_scores(masks_r, image_groups)
                    score_r = mdice_r if metric_name == "dice" else miou_r
                    elapsed = time.perf_counter() - t_r
                    row = dict(
                        strategy=strategy_type, segment_param=seg_label,
                        dropout_rounds=r, patches_per_drop=pdrop,
                        miou=round(miou_r, 4), miou_pct=round(miou_r * 100, 1),
                        mdice=round(mdice_r, 4), mdice_pct=round(mdice_r * 100, 1),
                        score=round(score_r, 4), score_pct=round(score_r * 100, 1), metric=metric_name,
                        mask_coverage_mean=round(cov_r, 2),
                        elapsed_s=round(elapsed, 1), status="ok",
                    )
                    print(
                        f"  [R={r} pdrop={pdrop:3d}]  "
                        f"mIoU={miou_r:.4f} ({miou_r*100:.1f}%)  "
                        f"mDice={mdice_r:.4f} ({mdice_r*100:.1f}%)  "
                        f"cov={cov_r:.1f}%  ({elapsed:.1f}s)"
                    )
                    if r == R_max and save_overlay_ids:
                        for iid in save_overlay_ids:
                            for cls_name, mask in masks_r.get(iid, {}).items():
                                save_mask_overlay(
                                    mask, image_groups[iid]["image"],
                                    str(masks_dir / f"{seg_label}_R{r}_p{pdrop}_{cls_name}.png"),
                                    gt_mask=image_groups[iid].get("gt_masks", {}).get(cls_name),
                                )

                except KeyboardInterrupt:
                    print("\nInterrupted.")
                    csv_file.close()
                    _print_winner(results, csv_path, masks_dir, ROOT, metric_name)
                    return
                except Exception as exc:
                    elapsed = time.perf_counter() - t_r
                    print(f"  [R={r} pdrop={pdrop}] ERROR ({elapsed:.1f}s): {exc}")
                    if args.verbose:
                        traceback.print_exc()
                    row = dict(
                        strategy=strategy_type, segment_param=seg_label,
                        dropout_rounds=r, patches_per_drop=pdrop,
                        elapsed_s=round(elapsed, 1), status=f"error:{exc}",
                    )

                results.append(row)
                writer.writerow({k: row.get(k, "") for k in fieldnames})
                csv_file.flush()

    csv_file.close()
    _print_winner(results, csv_path, masks_dir, ROOT, metric_name)


# ── Winner summary ────────────────────────────────────────────────────────────

def _print_winner(results, csv_path, masks_dir, ROOT, metric_name: str) -> None:
    ok = [r for r in results if r.get("status") == "ok"]
    print(f"\n{'='*72}")
    print(f"Tuning done.  {len(ok)} / {len(results)} rows succeeded.")

    if ok:
        best = max(ok, key=lambda r: (r.get("score", 0), -r.get("mask_coverage_mean", 100)))
        metric_label = "mDice" if metric_name == "dice" else "mIoU"
        print(f"\n  ★  Best by {metric_label}:")
        print(f"       strategy         = {best['strategy']}")
        print(f"       segment_param    = {best['segment_param']}")
        print(f"       dropout_rounds   = {best['dropout_rounds']}")
        print(f"       patches_per_drop = {best['patches_per_drop']}")
        print(f"       mIoU             = {best['miou']} ({best.get('miou_pct', '')}%)")
        print(f"       mDice            = {best.get('mdice', '')} ({best.get('mdice_pct', '')}%)")
        print(f"       selected metric  = {metric_name} ({best.get('score', '')})")
        print(f"\n  → Update config.yaml:")
        print(f"       patching.type: {best['strategy']}")
        
        # Save best config
        import json
        best_cfg = {
            "patching": {
                "type": best["strategy"],
                "regular": {},
                "superpixel": {}
            },
            "pipeline": {
                "dropout_rounds": best["dropout_rounds"],
                "patches_per_drop": best["patches_per_drop"] if best["patches_per_drop"] != "n/a" else 10
            }
        }
        
        if best["strategy"] in {"regular", "regular_free"}:
            g = int(best["segment_param"].replace("grid", ""))
            if best["strategy"] == "regular":
                print(f"       patching.regular.grid_size: {g}")
                best_cfg["patching"]["regular"]["grid_size"] = g
            else:
                print(f"       patching.regular_free.grid_size: {g}")
                best_cfg["patching"]["regular_free"] = {"grid_size": g}
        else:
            n = int(best["segment_param"].replace("sp", ""))
            print(f"       patching.superpixel.n_segments: {n}")
            best_cfg["patching"]["superpixel"]["n_segments"] = n
            
        print(f"       pipeline.dropout_rounds: {best['dropout_rounds']}")
        print(f"       pipeline.patches_per_drop: {best['patches_per_drop']}")
        
        out_json = csv_path.parent / "best_pipeline.json"
        with open(out_json, "w") as f:
            json.dump(best_cfg, f, indent=4)
        print(f"\n  JSON  : {out_json.relative_to(ROOT)}")

    print(f"\n  CSV   : {csv_path.relative_to(ROOT)}")
    print(f"  Masks : {masks_dir.relative_to(ROOT)}/")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
