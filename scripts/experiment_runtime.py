from __future__ import annotations

import copy
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
import torch
import yaml

from datasets.specs import resolve_dataset_spec
from utils.config import load_config
from utils.dataset import build_dataset
from utils.postprocess import postprocess_multiclass, save_mask_overlay

if TYPE_CHECKING:
    from models.blip_wrapper import BLIPWrapper


PROMPT_CLASS_ALIAS = {
    "diningtable": "table",
}


def _infer_transformer(cfg: dict) -> str:
    name = str(cfg.get("model", {}).get("name", "")).lower()
    if "blip" in name:
        return "blip"
    if "bridgetower" in name:
        return "bridgetower"
    return "unknown"

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


class InferenceEngine:
    """Encapsulates prompting, GradCAM extraction, patching, and Salience DropOut."""

    def __init__(self, cfg: dict, wrapper: Optional["BLIPWrapper"] = None) -> None:
        # Deep-copy so each engine instance owns its config snapshot.
        # This prevents config mutations in the orchestrator (e.g. granularity
        # sweeps) from silently changing the behaviour of already-built engines.
        self.cfg = copy.deepcopy(cfg)
        self.pipeline_cfg = self.cfg["pipeline"]
        self.patching_cfg = self.cfg["patching"]
        self.postprocess_cfg = self.cfg["postprocess"]

        if wrapper is None:
            from models.blip_wrapper import BLIPWrapper

            model_cfg = cfg["model"]
            device = None if model_cfg["device"] == "auto" else torch.device(model_cfg["device"])
            self.wrapper = BLIPWrapper(
                model_name=model_cfg["name"],
                device=device,
                input_size=model_cfg.get("image_size"),
            )
        else:
            self.wrapper = wrapper

    def map_prompt_classes(self, class_names: list[str]) -> list[str]:
        return [PROMPT_CLASS_ALIAS.get(c, c) for c in class_names]

    def build_patch_strategy(self, image):
        from core.patch_strategy import build_strategy

        return build_strategy(self.patching_cfg, self.wrapper, image)

    def build_prompt_inputs(self, image, class_names: list[str]) -> tuple[dict, dict[str, list[int]], list[str]]:
        prompt_classes = self.map_prompt_classes(class_names)
        prompt = _format_prompt(prompt_classes)
        inputs = self.wrapper.preprocess(image, prompt)
        token_indices = _get_class_token_indices(self.wrapper.processor, prompt, prompt_classes)
        return inputs, token_indices, prompt_classes

    def infer_multiclass_masks(
        self,
        image,
        selected_classes: list[str],
        full_ensemble_classes: Optional[list[str]] = None,
    ) -> tuple[dict[str, np.ndarray], object]:
        layer_idx = self.pipeline_cfg["layer"]
        head_idx = self.pipeline_cfg["head"]
        dropout_rounds = self.pipeline_cfg["dropout_rounds"]
        patches_per_drop = self.pipeline_cfg["patches_per_drop"]
        use_full_ensemble = self.pipeline_cfg.get("use_full_ensemble", True)

        strategy = self.build_patch_strategy(image)

        acc_full = None
        if use_full_ensemble:
            from core.salience_dropout import salience_dropout

            full_classes = full_ensemble_classes if full_ensemble_classes else selected_classes
            full_inputs, full_indices, full_prompt_classes = self.build_prompt_inputs(image, full_classes)
            acc_full = salience_dropout(
                wrapper=self.wrapper,
                pixel_values=full_inputs["pixel_values"],
                input_ids=full_inputs["input_ids"],
                attention_mask=full_inputs["attention_mask"],
                layer_idx=layer_idx,
                head_idx=head_idx,
                class_token_indices=full_indices,
                strategy=strategy,
                dropout_rounds=dropout_rounds,
                patches_per_drop=patches_per_drop,
                verbose=False,
            )

        filt_inputs, filt_indices, filt_prompt_classes = self.build_prompt_inputs(image, selected_classes)
        from core.salience_dropout import salience_dropout

        acc_filt = salience_dropout(
            wrapper=self.wrapper,
            pixel_values=filt_inputs["pixel_values"],
            input_ids=filt_inputs["input_ids"],
            attention_mask=filt_inputs["attention_mask"],
            layer_idx=layer_idx,
            head_idx=head_idx,
            class_token_indices=filt_indices,
            strategy=strategy,
            dropout_rounds=dropout_rounds,
            patches_per_drop=patches_per_drop,
            verbose=False,
        )

        spatial_maps = {}
        for class_name in selected_classes:
            prompt_name = PROMPT_CLASS_ALIAS.get(class_name, class_name)
            filt_spatial = strategy.to_spatial(acc_filt[prompt_name])
            if use_full_ensemble and acc_full is not None and prompt_name in acc_full:
                full_spatial = strategy.to_spatial(acc_full[prompt_name])
                spatial_maps[class_name] = (full_spatial + filt_spatial) / 2.0
            else:
                spatial_maps[class_name] = filt_spatial

        masks = postprocess_multiclass(
            spatial_maps=spatial_maps,
            original_image=image,
            threshold=self.postprocess_cfg.get("threshold", 0.15),
            gaussian_sigma=self.postprocess_cfg["gaussian_sigma"],
            use_blur=self.postprocess_cfg.get("use_blur", False),
            use_dense_crf=self.postprocess_cfg["use_dense_crf"],
        )
        # Release any lingering CUDA tensors from the GradCAM passes.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return masks, strategy

    @staticmethod
    def extract_gradcam_mask(
        attn,
        attn_grad,
        head_idx: int,
        class_token_indices: list[int],
        patches_per_side: int,
        threshold: float,
    ) -> tuple[np.ndarray, float]:
        from core.gradcam import compute_gradcam_salience

        flat_sal = compute_gradcam_salience(attn, attn_grad, head_idx, class_token_indices)
        sal = flat_sal.cpu().float()
        mn, mx = sal.min(), sal.max()
        if (mx - mn) > 1e-8:
            sal = (sal - mn) / (mx - mn)
        else:
            sal.zero_()
        mask = (sal.reshape(patches_per_side, patches_per_side).numpy() > threshold).astype(np.float32)
        return mask, float(mx.item())


class ReportWriter:
    """Writes config snapshots, CSV streams, and visual outputs to one isolated run directory.

    Output layout: root_dir / dataset_name / transformer_name / run_YYYYMMDD_HHMMSS_slug /
    """

    def __init__(
        self,
        root_dir: Path,
        slug: str,
        cfg: dict,
        dataset_name: str = "",
        transformer_name: str = "",
        run_dir: Optional[Path] = None,
    ) -> None:
        if run_dir is not None:
            self.run_dir = run_dir
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_slug = slug.replace("/", "_").replace(" ", "_")
            ds = (dataset_name.strip().lower() or str(cfg.get("dataset", {}).get("name", "unknown"))).lower()
            tr = transformer_name.strip().lower() or _infer_transformer(cfg)
            self.run_dir = root_dir / ds / tr / f"run_{ts}_{safe_slug}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.cfg = cfg
        self.csv_files: dict[str, object] = {}
        self.csv_writers: dict[str, csv.DictWriter] = {}

    def save_config(self, file_name: str = "config_snapshot.yaml") -> Path:
        out = self.run_dir / file_name
        out.write_text(yaml.safe_dump(self.cfg, sort_keys=False))
        return out

    def open_csv(self, file_name: str, fieldnames: list[str]) -> Path:
        csv_path = self.run_dir / file_name
        fh = open(csv_path, "w", newline="")
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        fh.flush()
        self.csv_files[file_name] = fh
        self.csv_writers[file_name] = writer
        return csv_path

    def write_csv_row(self, file_name: str, row: dict, fieldnames: list[str]) -> None:
        writer = self.csv_writers[file_name]
        fh = self.csv_files[file_name]
        writer.writerow({k: row.get(k, "") for k in fieldnames})
        fh.flush()

    def save_mask_overlay(self, rel_path: str, mask: np.ndarray, image) -> Path:
        out = self.run_dir / rel_path
        out.parent.mkdir(parents=True, exist_ok=True)
        save_mask_overlay(mask, image, str(out))
        return out

    def save_text(self, file_name: str, lines: list[str]) -> Path:
        out = self.run_dir / file_name
        out.write_text("\n".join(lines))
        return out

    def close(self) -> None:
        for fh in self.csv_files.values():
            fh.close()
        self.csv_files.clear()
        self.csv_writers.clear()


class ExperimentRunner:
    """Context manager owning config, dataset/spec setup, inference engine, and report lifecycle."""

    def __init__(
        self,
        config_path: str,
        root: Path,
        slug: str,
        output_root: str = "outputs",
        dataset_override: Optional[dict] = None,
        run_dir_override: Optional[Path] = None,
    ) -> None:
        self.root = root
        self.config_path = config_path
        self.cfg = load_config(config_path)
        self.slug = slug
        self.output_root = output_root
        self.dataset_override = dataset_override or {}
        self.run_dir_override = run_dir_override

        self.writer: Optional[ReportWriter] = None
        self.dataset = None
        self.dataset_spec = None
        self.inference_engine: Optional[InferenceEngine] = None

    def __enter__(self) -> "ExperimentRunner":
        ds_name = str(self.cfg.get("dataset", {}).get("name", "unknown")).lower()
        tr_name = _infer_transformer(self.cfg)
        self.writer = ReportWriter(
            self.root / self.output_root,
            self.slug,
            self.cfg,
            dataset_name=ds_name,
            transformer_name=tr_name,
            run_dir=self.run_dir_override,
        )
        self.writer.save_config()

        ds_cfg = {**self.cfg["dataset"], **self.dataset_override}
        self.dataset = build_dataset(ds_cfg)
        self.dataset_spec = getattr(self.dataset, "dataset_spec", None)
        if self.dataset_spec is None:
            self.dataset_spec = resolve_dataset_spec(ds_cfg.get("name", "voc"))

        self.inference_engine = InferenceEngine(self.cfg)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.writer is not None:
            self.writer.close()
        # Explicitly release the BLIP model from GPU memory so the next phase
        # (or a subprocess) doesn't load a second copy alongside this one.
        if self.inference_engine is not None:
            del self.inference_engine.wrapper
            self.inference_engine = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
