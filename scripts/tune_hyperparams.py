"""
Layer/head tuning with configurable metric.

Supports two metrics via tuning.metric:
- clip: weakly supervised CLIP contrastive reward.
- dice: supervised mean Dice against GT masks.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.experiment_runtime import ExperimentRunner
from scripts.metrics_engine import MetricsEngine


PROMPT_PREFIX = "A picture of"


def _format_prompt(classes: list[str] | str) -> str:
    if isinstance(classes, str):
        classes = [classes]
    return f"{PROMPT_PREFIX} " + " ".join(classes)


def _get_class_token_indices(processor, prompt: str, classes: list[str] | str) -> dict[str, list[int]]:
    if isinstance(classes, str):
        classes = [classes]
    tokenizer = processor.tokenizer
    prefix_ids = tokenizer.encode(PROMPT_PREFIX, add_special_tokens=False)
    start_idx = 1 + len(prefix_ids)

    class_indices = {}
    curr_idx = start_idx
    for cls in classes:
        cls_ids = tokenizer(cls, add_special_tokens=False).input_ids
        n = len(cls_ids)
        class_indices[cls] = list(range(curr_idx, curr_idx + n))
        curr_idx += n
    return class_indices


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PnP-OVSS layer/head tuning with clip or dice objective.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--metric", choices=["clip", "dice"], default=None,
                   help="Override tuning.metric from config.")
    p.add_argument("--verbose", action="store_true",
                   help="Print full tracebacks on errors.")
    return p.parse_args()


def _resize_mask_to_gt(mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    if mask.shape == gt_mask.shape:
        return mask
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    resized = np.array(
        mask_pil.resize((gt_mask.shape[1], gt_mask.shape[0]), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    return resized


def _apply_mask_local(image: Image.Image, mask: np.ndarray) -> Image.Image:
    img_np = np.array(image.convert("RGB"), dtype=np.float32)
    h, w = img_np.shape[:2]
    mh, mw = mask.shape
    if (mh, mw) != (h, w):
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
        mask = np.array(mask_pil, dtype=np.float32) / 255.0
    masked = img_np * mask[..., np.newaxis]
    return Image.fromarray(masked.astype(np.uint8))


def _print_winner(metric: str, score_key: str, ok_results: list[dict], run_dir: Path) -> None:
    print("\n" + "=" * 72)
    if not ok_results:
        print("No successful tuning combinations.")
        print(f"Run dir: {run_dir}")
        print("=" * 72)
        return

    best = max(ok_results, key=lambda r: r.get(score_key, 0.0))
    print(f"Best by tuning.metric='{metric}':")
    print(f"  Layer = {best['layer']}   Head = {best['head']}")
    print(f"  Score = {best.get(score_key, 0.0):.4f}")
    if metric == "clip":
        print(
            f"  CLIP reward = {best.get('clip_reward', 0)} / {best.get('total_possible_reward', 0)} "
            f"({best.get('clip_reward_pct', 0.0):.1f}%)"
        )
    else:
        print(f"  mDice = {best.get('mdice', 0.0):.4f}")

    print(f"Run dir: {run_dir}")
    print("=" * 72)


def main() -> None:
    args = parse_args()

    # Load once to derive run slug and overrides, then let ExperimentRunner own lifecycle.
    from utils.config import load_config

    cfg = load_config(args.config)
    metric = (args.metric or cfg.get("tuning", {}).get("metric", "clip")).lower()
    if metric not in {"clip", "dice"}:
        raise ValueError(f"Unsupported tuning.metric '{metric}'. Use 'clip' or 'dice'.")

    tune_cfg = cfg["tuning"]
    ds_cfg = cfg["dataset"]
    tune_max = tune_cfg.get("max_images", 50)

    slug = f"tune_layer_head_{metric}_{ds_cfg.get('name', 'dataset')}"
    with ExperimentRunner(
        config_path=args.config,
        root=ROOT,
        slug=slug,
        output_root="outputs",
        dataset_override={"max_images": tune_max},
    ) as runner:
        runner.cfg["tuning"]["metric"] = metric
        writer = runner.writer
        assert writer is not None

        print(f"Run dir : {writer.run_dir.relative_to(ROOT)}")
        print(f"Metric  : {metric}")

        dataset = runner.dataset
        dataset_spec = runner.dataset_spec
        inf = runner.inference_engine
        assert dataset is not None and inf is not None and dataset_spec is not None

        wrapper = inf.wrapper
        num_layers, num_heads = wrapper.num_text_layers, wrapper.num_heads
        total = num_layers * num_heads
        patches_per_side = wrapper.num_patches_per_side

        print(
            f"Model   : {num_layers} layers x {num_heads} heads | "
            f"{wrapper.image_size}px | patch grid {patches_per_side}x{patches_per_side}"
        )

        samples = list(dataset)
        print(f"Dataset : {len(samples)} (image, class) samples")
        if not samples:
            raise RuntimeError("No samples found for tuning dataset.")

        image_groups: dict[str, dict] = {}
        for s in samples:
            iid = str(s["image_uid"])
            if iid not in image_groups:
                image_groups[iid] = {
                    "image": s["image"],
                    "classes": [],
                    "samples": [],
                }
            image_groups[iid]["classes"].append(str(s["class_name"]))
            image_groups[iid]["samples"].append(s)

        print(f"Images  : {len(image_groups)} unique images")

        import torch

        all_pairs: list[tuple[str, object]] = []
        for iid, group in image_groups.items():
            for s in group["samples"]:
                all_pairs.append((iid, s))

        all_prompts = [_format_prompt(str(s["class_name"])) for _, s in all_pairs]
        mega_text = wrapper.preprocess_text(all_prompts)

        pv_parts: list[torch.Tensor] = []
        for iid, group in image_groups.items():
            k = len(group["samples"])
            pv = wrapper.preprocess_image(group["image"])
            pv_parts.append(pv.expand(k, -1, -1, -1).contiguous())
        mega_pv = torch.cat(pv_parts, dim=0)

        all_cls_idx = []
        for _, s in all_pairs:
            cls_name = str(s["class_name"])
            cls_idx_map = _get_class_token_indices(wrapper.processor, _format_prompt(cls_name), cls_name)
            cls_idx = cls_idx_map.get(cls_name)
            if cls_idx is None and cls_idx_map:
                cls_idx = next(iter(cls_idx_map.values()))
            all_cls_idx.append(cls_idx or [])

        blip_batch_size = tune_cfg.get("blip_batch_size", 4)
        n_chunks = (len(all_pairs) + blip_batch_size - 1) // blip_batch_size
        print(f"BLIP bs : {blip_batch_size} ({n_chunks} chunks/layer)")

        clip_reward = None
        text_embeds = {}
        black_embeds = {}
        if metric == "clip":
            from core.clip_reward import CLIPReward

            clip_cfg = runner.cfg.get("clip", {})
            clip_reward = CLIPReward(
                model_name=clip_cfg.get("model", "openai/clip-vit-large-patch14-336"),
                device=wrapper.device,
            )
            unique_classes = list({str(s["class_name"]) for s in samples})
            text_embeds = clip_reward.precompute_text_embeddings(unique_classes)

            black_images = [Image.new("RGB", g["image"].size, (0, 0, 0)) for g in image_groups.values()]
            black_vecs = clip_reward.encode_images(black_images)
            black_embeds = {iid: black_vecs[i] for i, iid in enumerate(image_groups)}

        metric_score_key = "clip_reward" if metric == "clip" else "mdice"
        fieldnames = [
            "layer",
            "head",
            "metric",
            "clip_reward",
            "clip_reward_pct",
            "mdice",
            "mask_coverage_mean",
            "salience_max_mean",
            "score",
            "elapsed_s",
            "status",
            "total_possible_reward",
        ]
        csv_name = "tune_results.csv"
        writer.open_csv(csv_name, fieldnames)

        save_overlay_ids = set(list(image_groups.keys())[:8])
        done = 0
        results: list[dict] = []
        total_possible_reward = len(samples)
        base_threshold = float(tune_cfg.get("mask_threshold", 0.25))

        for layer_idx in range(num_layers):
            print(f"\nLayer {layer_idx:02d}: forward/backward across all samples...", end=" ", flush=True)
            t_fwd = time.perf_counter()
            attn_parts = []
            grad_parts = []
            fwd_error: Exception | None = None

            try:
                for start in range(0, len(all_pairs), blip_batch_size):
                    end = min(start + blip_batch_size, len(all_pairs))
                    attn, grad = wrapper.forward_with_gradcam(
                        mega_pv[start:end],
                        mega_text["input_ids"][start:end],
                        mega_text["attention_mask"][start:end],
                        layer_idx,
                    )
                    if attn is None or grad is None:
                        raise RuntimeError("GradCAM hook did not fire.")
                    attn_parts.append(attn)
                    grad_parts.append(grad)
            except Exception as exc:
                fwd_error = exc

            fwd_elapsed = time.perf_counter() - t_fwd
            if fwd_error is not None:
                print(f"ERROR ({fwd_elapsed:.1f}s): {fwd_error}")
                if args.verbose:
                    traceback.print_exc()
                for head_idx in range(num_heads):
                    done += 1
                    row = {
                        "layer": layer_idx,
                        "head": head_idx,
                        "metric": metric,
                        "elapsed_s": round(fwd_elapsed / max(num_heads, 1), 2),
                        "status": f"fwd_error:{fwd_error}",
                    }
                    results.append(row)
                    writer.write_csv_row(csv_name, row, fieldnames)
                continue

            attn_mega = torch.cat(attn_parts, dim=0)
            grad_mega = torch.cat(grad_parts, dim=0)
            print(f"done ({fwd_elapsed:.1f}s)")

            for head_idx in range(num_heads):
                done += 1
                tag = f"layer{layer_idx:02d}_head{head_idx:02d}"
                print(f"[{done:3d}/{total}] {tag}", end=" ", flush=True)

                t0 = time.perf_counter()
                row = {
                    "layer": layer_idx,
                    "head": head_idx,
                    "metric": metric,
                    "total_possible_reward": total_possible_reward,
                }

                try:
                    group_masks: dict[str, dict[str, np.ndarray]] = {iid: {} for iid in image_groups}
                    sal_maxes: list[float] = []
                    coverages: list[float] = []

                    for idx, (iid, s) in enumerate(all_pairs):
                        cls_name = str(s["class_name"])
                        mask, sal_max = inf.extract_gradcam_mask(
                            attn=attn_mega[idx:idx+1],
                            attn_grad=grad_mega[idx:idx+1],
                            head_idx=head_idx,
                            class_token_indices=all_cls_idx[idx],
                            patches_per_side=patches_per_side,
                            threshold=base_threshold,
                        )
                        group_masks[iid][cls_name] = mask
                        sal_maxes.append(sal_max)
                        coverages.append(float(mask.mean() * 100))

                    clip_total_reward = 0
                    mdice = 0.0

                    if metric == "clip":
                        assert clip_reward is not None
                        masked_imgs = []
                        masked_meta = []
                        for iid, class_masks in group_masks.items():
                            image = image_groups[iid]["image"]
                            for cls_name, mask in class_masks.items():
                                masked_imgs.append(_apply_mask_local(image, mask))
                                masked_meta.append((iid, cls_name))

                        masked_embeds = clip_reward.encode_images(masked_imgs)
                        for i, (iid, cls_name) in enumerate(masked_meta):
                            clip_total_reward += clip_reward.compute_reward_from_embeds(
                                masked_embed=masked_embeds[i],
                                black_embed=black_embeds[iid],
                                class_name=cls_name,
                                all_classes=image_groups[iid]["classes"],
                                text_embeds=text_embeds,
                            )
                    else:
                        combo_metrics = MetricsEngine(dataset_spec)
                        for iid, s in all_pairs:
                            cls_name = str(s["class_name"])
                            pred_mask = group_masks[iid].get(cls_name)
                            if pred_mask is None:
                                continue
                            gt_mask = s.get("gt_mask")
                            if gt_mask is None:
                                combo_metrics.record_sample(cls_name, pred_mask, None)
                                continue
                            pred_rs = _resize_mask_to_gt(pred_mask, gt_mask)
                            combo_metrics.record_sample(cls_name, pred_rs, gt_mask)
                        _, mdice = combo_metrics.running_means()

                    for iid in save_overlay_ids:
                        image = image_groups[iid]["image"]
                        for cls_name, mask in group_masks.get(iid, {}).items():
                            writer.save_mask_overlay(
                                rel_path=f"masks/{tag}_{cls_name}.png",
                                mask=mask,
                                image=image,
                            )

                    elapsed = time.perf_counter() - t0
                    clip_pct = (clip_total_reward / total_possible_reward * 100) if total_possible_reward > 0 else 0.0
                    cov_mean = float(np.mean(coverages)) if coverages else 0.0
                    sal_mean = float(np.mean(sal_maxes)) if sal_maxes else 0.0
                    score = float(clip_total_reward) if metric == "clip" else float(mdice)

                    row.update({
                        "clip_reward": int(clip_total_reward),
                        "clip_reward_pct": round(clip_pct, 2),
                        "mdice": round(float(mdice), 6),
                        "mask_coverage_mean": round(cov_mean, 2),
                        "salience_max_mean": round(sal_mean, 6),
                        "score": round(score, 6),
                        "elapsed_s": round(elapsed, 2),
                        "status": "ok",
                    })

                    if metric == "clip":
                        print(f"reward={clip_total_reward}/{total_possible_reward} ({clip_pct:.1f}%) ({elapsed:.2f}s)")
                    else:
                        print(f"mDice={mdice:.4f} ({elapsed:.2f}s)")

                except Exception as exc:
                    elapsed = time.perf_counter() - t0
                    print(f"ERROR ({elapsed:.2f}s): {exc}")
                    if args.verbose:
                        traceback.print_exc()
                    row.update({
                        "elapsed_s": round(elapsed, 2),
                        "status": f"error:{exc}",
                    })

                results.append(row)
                writer.write_csv_row(csv_name, row, fieldnames)

        ok_results = [r for r in results if r.get("status") == "ok"]
        _print_winner(metric, metric_score_key, ok_results, writer.run_dir)

        summary_lines = [
            f"metric: {metric}",
            f"total_combinations: {total}",
            f"successful_combinations: {len(ok_results)}",
        ]
        if ok_results:
            best = max(ok_results, key=lambda r: r.get(metric_score_key, 0.0))
            summary_lines.extend([
                f"best_layer: {best['layer']}",
                f"best_head: {best['head']}",
                f"best_score: {best.get(metric_score_key, 0.0)}",
                f"best_clip_reward: {best.get('clip_reward', 0)}",
                f"best_mdice: {best.get('mdice', 0.0)}",
            ])
        writer.save_text("summary.txt", summary_lines)


if __name__ == "__main__":
    main()
