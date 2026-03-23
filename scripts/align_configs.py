#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Align all foundation model probing YAML configs.

This script performs three operations:
1. Normalize dataset folder names to the canonical set.
2. Normalize YAML file names to: {model_variant}_64f_{dataset_lower}.yaml
3. Ensure every model variant covers all 9 datasets; generate missing configs
   from existing templates.

Usage:
    python scripts/align_configs.py                # dry-run (default)
    python scripts/align_configs.py --apply        # actually perform changes
    python scripts/align_configs.py --apply --verbose
"""

from __future__ import annotations

import argparse
import re
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# Canonical dataset definitions
# ---------------------------------------------------------------------------
CANONICAL_DATASETS: Dict[str, dict] = {
    "AutoLaparo": {
        "num_classes": 7,
        "resolution": 224,
        "dataset_train": "data/Surge_Frames/AutoLaparo/clips_64f/train_dense_64f_detailed.csv",
        "dataset_val": "data/Surge_Frames/AutoLaparo/clips_64f/test_dense_64f_detailed.csv",
        "metric_aggregation": None,
        "data_list_format": False,
    },
    "Cholec80": {
        "num_classes": 7,
        "resolution": 256,
        "dataset_train": "data/Surge_Frames/Cholec80/clips_64f/train_dense_64f_detailed.csv",
        "dataset_val": "data/Surge_Frames/Cholec80/clips_64f/test_dense_64f_detailed.csv",
        "metric_aggregation": None,
        "data_list_format": False,
    },
    "EgoSurgery": {
        "num_classes": 9,
        "resolution": 224,
        "dataset_train": "data/Surge_Frames/EgoSurgery/clips_64f/train_dense_64f_detailed.csv",
        "dataset_val": "data/Surge_Frames/EgoSurgery/clips_64f/test_dense_64f_detailed.csv",
        "metric_aggregation": None,
        "data_list_format": False,
    },
    "M2CAI16": {
        "num_classes": 8,
        "resolution": 224,
        "dataset_train": "data/Surge_Frames/M2CAI16/clips_64f/train_dense_64f_detailed.csv",
        "dataset_val": "data/Surge_Frames/M2CAI16/clips_64f/test_dense_64f_detailed.csv",
        "metric_aggregation": None,
        "data_list_format": False,
    },
    "OphNet": {
        "num_classes": 52,
        "resolution": 224,
        "dataset_train": "data/Surge_Frames/OphNet2024_phase/clips_64f/train_dense_64f_detailed.csv",
        "dataset_val": "data/Surge_Frames/OphNet2024_phase/clips_64f/test_dense_64f_detailed.csv",
        "metric_aggregation": None,
        "data_list_format": False,
    },
    "PitVis": {
        "num_classes": 15,
        "resolution": 224,
        "dataset_train": "data/Surge_Frames/PitVis/clips_64f/train_dense_64f_detailed.csv",
        "dataset_val": "data/Surge_Frames/PitVis/clips_64f/test_dense_64f_detailed.csv",
        "metric_aggregation": None,
        "data_list_format": False,
    },
    "PmLR50": {
        "num_classes": 7,
        "resolution": 224,
        "dataset_train": "data/Surge_Frames/PmLR50/clips_64f/train_dense_64f_detailed.csv",
        "dataset_val": "data/Surge_Frames/PmLR50/clips_64f/test_dense_64f_detailed.csv",
        "metric_aggregation": None,
        "data_list_format": False,
    },
    "PolypDiag": {
        "num_classes": 2,
        "resolution": 224,
        "dataset_train": "data/Surge_Frames/PolypDiag_v1/train_metadata.csv",
        "dataset_val": "data/Surge_Frames/PolypDiag_v1/val_metadata.csv",
        "metric_aggregation": "global",
        "data_list_format": True,
    },
    "Surgical-Action-160": {
        "num_classes": 16,
        "resolution": 224,
        "dataset_train": "data/Surge_Frames/SurgicalActions160_v1/train_metadata_fold0_fps25.csv",
        "dataset_val": "data/Surge_Frames/SurgicalActions160_v1/test_metadata_fold0_fps25.csv",
        "metric_aggregation": "global",
        "data_list_format": True,
    },
}

DATASET_LOWER_TO_CANONICAL = {k.lower().replace("-", ""): k for k in CANONICAL_DATASETS}
DATASET_LOWER_TO_CANONICAL.update({
    "cholec80": "Cholec80",
    "probing_cholec80": "Cholec80",
    "polypdiag": "PolypDiag",
    "surgical-action-160": "Surgical-Action-160",
    "surgicalaction160": "Surgical-Action-160",
    "surgical_action_160": "Surgical-Action-160",
    "surgicalactions160": "Surgical-Action-160",
})

# Model variants: model_dir -> list of (file_prefix, model_type, encoder.model_name, checkpoint)
MODEL_VARIANTS: Dict[str, List[Tuple[str, str, str, Optional[str]]]] = {
    "dinov3": [
        ("dinov3_vitl", "dinov3", "dinov3_vitl16", None),
        ("dinov3_vith", "dinov3", "dinov3_vith16plus", None),
    ],
    "endofm": [
        ("endofm_vitb", "endofm", "endofm", None),
    ],
    "endomamba": [
        ("endomamba_small", "endomamba", "endomamba_small",
         "ckpts_foundation/endomamba_checkpoint-best.pth"),
    ],
    "endossl": [
        ("endossl_vitl_laparo", "endossl", "endossl_laparo", None),
        ("endossl_vitl_colono", "endossl", "endossl_colono", None),
    ],
    "endovit": [
        ("endovit_vitl", "endovit", "endovit", None),
    ],
    "gastronet": [
        ("gastronet_vits", "gastronet", "gastronet", None),
    ],
    "gsvit": [
        ("gsvit_vit", "gsvit", "gsvit", None),
    ],
    "selfsupsurg": [
        ("selfsupsurg_res50", "selfsupsurg", "selfsupsurg", None),
    ],
    "surgenet": [
        ("surgenet_convnextv2", "surgenet", "surgenet_convnextv2", None),
        ("surgenetxl_caformer", "surgenet", "surgenetxl", None),
    ],
    "surgvlp": [
        ("surgvlp_res50", "surgvlp", "surgvlp", None),
    ],
    "videomaev2": [
        ("videomaev2_large", "videomae", "videomaev2_large", None),
        ("videomaev2_huge", "videomae", "videomaev2_huge", None),
        ("videomaev2_giant", "videomae", "videomaev2_giant", None),
    ],
}

MULTIHEAD_KWARGS = []
for wd in [0.01, 0.1, 0.4, 0.8]:
    for lr in [0.005, 0.003, 0.001, 0.0003, 0.0001]:
        MULTIHEAD_KWARGS.append({
            "final_lr": 0.0,
            "final_weight_decay": wd,
            "lr": lr,
            "start_lr": lr,
            "warmup": 0.0,
            "weight_decay": wd,
        })


def _normalize_folder_name(name: str) -> Optional[str]:
    """Map a possibly-wrong dataset folder name to the canonical form."""
    key = name.lower().replace("-", "").replace("_", "")
    key = re.sub(r"^probing_?", "", key)
    for pattern, canonical in DATASET_LOWER_TO_CANONICAL.items():
        if key == pattern.replace("-", "").replace("_", ""):
            return canonical
    return None


def _dataset_lower(ds: str) -> str:
    """Canonical dataset name -> lowercase for filenames."""
    return ds.lower()


def _build_yaml_content(
    variant_prefix: str,
    model_type: str,
    encoder_model_name: str,
    checkpoint: Optional[str],
    dataset_name: str,
) -> str:
    """Generate a complete YAML config string."""
    ds = CANONICAL_DATASETS[dataset_name]
    ds_lower = _dataset_lower(dataset_name)
    tag = f"{variant_prefix}_64f_{ds_lower}"
    folder = f"logs/foundation/{variant_prefix}_{ds_lower}"

    lines = []
    lines.append(f"cpus_per_task: 4")
    lines.append(f"eval_name: foundation_phase_probing")
    lines.append(f"folder: {folder}")
    lines.append(f"mem_per_gpu: 220G")
    lines.append(f"nodes: 1")
    lines.append(f"max_workers: 4")
    lines.append(f"resume_checkpoint: true")
    lines.append(f"tag: {tag}")
    lines.append(f"tasks_per_node: 1")

    if ds["metric_aggregation"]:
        lines.append(f'metric_aggregation: "{ds["metric_aggregation"]}"')

    lines.append("")
    lines.append("# wandb configuration")
    lines.append("wandb:")
    lines.append("  project: foundation-model-probing")
    lines.append("  entity: nsjepa")
    lines.append(f"  name: {tag}")
    lines.append("  tags:")
    lines.append(f"    - {dataset_name}")
    lines.append("    - phase-probing")
    lines.append(f"    - {variant_prefix}_64f")
    lines.append("  group: null")
    lines.append(f'  notes: "Multi-head probing on {dataset_name} dataset"')

    lines.append("")
    lines.append("experiment:")
    lines.append("  classifier:")
    lines.append("    num_heads: 16")
    lines.append("    num_probe_blocks: 4")
    lines.append("  data:")
    lines.append("    dataset_type: surgical_videodataset")

    if ds["data_list_format"]:
        lines.append("    dataset_train:")
        lines.append(f'    - {ds["dataset_train"]}')
        lines.append("")
        lines.append("    dataset_val:")
        lines.append(f'    - {ds["dataset_val"]}')
        lines.append("")
    else:
        lines.append(f'    dataset_train: {ds["dataset_train"]}')
        lines.append(f'    dataset_val: {ds["dataset_val"]}')

    lines.append("    frames_per_clip: 64")
    lines.append(f'    num_classes: {ds["num_classes"]}')
    lines.append("    num_segments: 1")
    lines.append("    num_views_per_segment: 1")
    lines.append(f'    resolution: {ds["resolution"]}')

    lines.append("")
    lines.append("  optimization:")
    lines.append("    batch_size: 4")
    lines.append("    multihead_kwargs:")

    for kw in MULTIHEAD_KWARGS:
        lines.append(f"    - final_lr: {kw['final_lr']}")
        lines.append(f"      final_weight_decay: {kw['final_weight_decay']}")
        lines.append(f"      lr: {kw['lr']}")
        lines.append(f"      start_lr: {kw['start_lr']}")
        lines.append(f"      warmup: {kw['warmup']}")
        lines.append(f"      weight_decay: {kw['weight_decay']}")

    lines.append("    num_epochs: 1")
    lines.append("    use_bfloat16: true")
    lines.append("    use_pos_embed: false")

    lines.append("")
    lines.append("model_kwargs:")
    lines.append(f"  checkpoint: {checkpoint if checkpoint else 'null'}")
    lines.append("  module_name: evals.foundation_phase_probing.modelcustom.foundation_model_wrapper")
    lines.append(f"  model_type: {model_type}")
    lines.append("  encoder:")
    lines.append(f"    model_name: {encoder_model_name}")
    lines.append("  wrapper_kwargs:")
    lines.append("    tubelet_size: 2")
    lines.append("    max_frames: 128")
    lines.append("    use_pos_embed: true")

    return "\n".join(lines) + "\n"


def scan_existing(configs_root: Path) -> Dict[str, Dict[str, List[Path]]]:
    """
    Scan existing configs and return:
    {model_dir: {canonical_dataset: [yaml_paths]}}
    """
    result: Dict[str, Dict[str, List[Path]]] = {}
    for model_dir in sorted(configs_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        result[model_name] = {}
        for ds_dir in sorted(model_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            canonical = _normalize_folder_name(ds_dir.name)
            if canonical is None:
                canonical = ds_dir.name
            yamls = sorted(ds_dir.glob("*.yaml"))
            if canonical not in result[model_name]:
                result[model_name][canonical] = []
            result[model_name][canonical].extend(yamls)
    return result


def _expected_filename(variant_prefix: str, dataset_name: str) -> str:
    ds_lower = _dataset_lower(dataset_name)
    return f"{variant_prefix}_64f_{ds_lower}.yaml"


def main():
    parser = argparse.ArgumentParser(description="Align foundation model probing configs")
    parser.add_argument("--configs_root", type=str,
                        default="configs/foundation_model_probing",
                        help="Root directory of foundation model probing configs")
    parser.add_argument("--apply", action="store_true",
                        help="Actually perform changes (default: dry-run)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed info")
    args = parser.parse_args()

    configs_root = Path(args.configs_root)
    if not configs_root.is_dir():
        print(f"Error: {configs_root} is not a directory")
        return

    dry_run = not args.apply
    if dry_run:
        print("=" * 70)
        print("DRY RUN — no changes will be made. Use --apply to execute.")
        print("=" * 70)
    print()

    # Track statistics
    folders_renamed = 0
    files_renamed = 0
    files_rewritten = 0
    files_created = 0
    skipped_models = []
    skipped_extras = []

    # ---------------------------------------------------------------
    # Phase 1: Rename dataset folders to canonical names
    # ---------------------------------------------------------------
    print("=" * 70)
    print("Phase 1: Normalize dataset folder names")
    print("=" * 70)

    for model_dir in sorted(configs_root.iterdir()):
        if not model_dir.is_dir():
            continue
        for ds_dir in sorted(model_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            canonical = _normalize_folder_name(ds_dir.name)
            if canonical is None:
                print(f"  [SKIP] Unknown dataset folder: {ds_dir} (not in canonical list)")
                skipped_extras.append(str(ds_dir))
                continue
            if ds_dir.name != canonical:
                new_dir = ds_dir.parent / canonical
                print(f"  [RENAME] {ds_dir.relative_to(configs_root)} -> {canonical}")
                if not dry_run:
                    if new_dir.exists():
                        for f in ds_dir.iterdir():
                            shutil.move(str(f), str(new_dir / f.name))
                        ds_dir.rmdir()
                    else:
                        ds_dir.rename(new_dir)
                folders_renamed += 1

    print(f"\n  Folders to rename: {folders_renamed}\n")

    # ---------------------------------------------------------------
    # Phase 2: Rename YAML files and rewrite content
    # ---------------------------------------------------------------
    print("=" * 70)
    print("Phase 2: Normalize YAML filenames & content")
    print("=" * 70)

    for model_dir in sorted(configs_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        if model_name not in MODEL_VARIANTS:
            if args.verbose:
                print(f"  [SKIP] Model '{model_name}' not in MODEL_VARIANTS definition")
            skipped_models.append(model_name)
            continue

        variants = MODEL_VARIANTS[model_name]

        for ds_dir in sorted(model_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            canonical = _normalize_folder_name(ds_dir.name)
            if canonical is None:
                continue
            actual_dir = ds_dir.parent / canonical if not dry_run else ds_dir

            for yaml_path in sorted(actual_dir.glob("*.yaml") if actual_dir.exists() else ds_dir.glob("*.yaml")):
                fname = yaml_path.name
                fname_lower = fname.lower()

                # Skip special eval files (pumch, pwh, tss)
                if any(tag in fname_lower for tag in ["_eval_pumch", "_eval_pwh", "_eval_tss"]):
                    if args.verbose:
                        print(f"  [KEEP] {yaml_path.relative_to(configs_root)} (eval variant)")
                    continue

                # Determine which variant this file belongs to
                matched_variant = None
                for vprefix, vtype, vencoder, vckpt in variants:
                    vprefix_norm = vprefix.lower()
                    if fname_lower.startswith(vprefix_norm):
                        matched_variant = (vprefix, vtype, vencoder, vckpt)
                        break

                if matched_variant is None:
                    print(f"  [WARN] Cannot match variant for: {yaml_path.relative_to(configs_root)}")
                    continue

                vprefix, vtype, vencoder, vckpt = matched_variant
                expected_name = _expected_filename(vprefix, canonical)
                ds_lower = _dataset_lower(canonical)

                new_content = _build_yaml_content(vprefix, vtype, vencoder, vckpt, canonical)

                # Rewrite content
                target_dir = ds_dir.parent / canonical
                target_path = target_dir / expected_name

                if fname != expected_name:
                    print(f"  [RENAME+REWRITE] {yaml_path.relative_to(configs_root)} -> {canonical}/{expected_name}")
                    files_renamed += 1
                else:
                    print(f"  [REWRITE] {yaml_path.relative_to(configs_root)}")

                files_rewritten += 1

                if not dry_run:
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target_path.write_text(new_content, encoding="utf-8")
                    if yaml_path != target_path and yaml_path.exists():
                        yaml_path.unlink()

    print(f"\n  Files to rename: {files_renamed}")
    print(f"  Files to rewrite: {files_rewritten}\n")

    # ---------------------------------------------------------------
    # Phase 3: Generate missing configs
    # ---------------------------------------------------------------
    print("=" * 70)
    print("Phase 3: Generate missing dataset configs")
    print("=" * 70)

    for model_name, variants in sorted(MODEL_VARIANTS.items()):
        model_dir = configs_root / model_name
        if not model_dir.exists():
            print(f"  [SKIP] Model directory does not exist: {model_name}")
            continue

        for ds_name in CANONICAL_DATASETS:
            ds_dir = model_dir / ds_name
            for vprefix, vtype, vencoder, vckpt in variants:
                expected_name = _expected_filename(vprefix, ds_name)
                expected_path = ds_dir / expected_name

                if expected_path.exists() and not dry_run:
                    continue

                # In dry-run, check both old and new locations
                if dry_run:
                    found = False
                    for candidate_dir in model_dir.iterdir():
                        if not candidate_dir.is_dir():
                            continue
                        canonical_check = _normalize_folder_name(candidate_dir.name)
                        if canonical_check == ds_name:
                            for yf in candidate_dir.glob("*.yaml"):
                                if yf.name.lower().startswith(vprefix.lower()):
                                    if not any(t in yf.name.lower()
                                               for t in ["_eval_pumch", "_eval_pwh", "_eval_tss"]):
                                        found = True
                                        break
                        if found:
                            break
                    if found:
                        continue

                print(f"  [CREATE] {model_name}/{ds_name}/{expected_name}")
                files_created += 1

                if not dry_run:
                    ds_dir.mkdir(parents=True, exist_ok=True)
                    content = _build_yaml_content(vprefix, vtype, vencoder, vckpt, ds_name)
                    expected_path.write_text(content, encoding="utf-8")

    print(f"\n  Configs to create: {files_created}\n")

    # ---------------------------------------------------------------
    # Phase 4: Clean up empty directories
    # ---------------------------------------------------------------
    if not dry_run:
        for model_dir in sorted(configs_root.iterdir()):
            if not model_dir.is_dir():
                continue
            for ds_dir in sorted(model_dir.iterdir()):
                if ds_dir.is_dir() and not any(ds_dir.iterdir()):
                    print(f"  [RMDIR] Empty directory: {ds_dir.relative_to(configs_root)}")
                    ds_dir.rmdir()

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Folders renamed:    {folders_renamed}")
    print(f"  Files renamed:      {files_renamed}")
    print(f"  Files rewritten:    {files_rewritten}")
    print(f"  Files created:      {files_created}")
    if skipped_models:
        print(f"  Skipped models (no variant def): {', '.join(skipped_models)}")
    if skipped_extras:
        print(f"  Skipped non-standard dirs: {len(skipped_extras)}")
        for s in skipped_extras:
            print(f"    - {s}")

    # Coverage matrix
    print()
    print("=" * 70)
    print("Coverage Matrix (after alignment)")
    print("=" * 70)

    header = f"{'Model':<20}"
    for ds in CANONICAL_DATASETS:
        short = ds[:8]
        header += f" {short:>8}"
    print(header)
    print("-" * len(header))

    for model_name, variants in sorted(MODEL_VARIANTS.items()):
        for vprefix, _, _, _ in variants:
            row = f"{vprefix:<20}"
            for ds_name in CANONICAL_DATASETS:
                expected = _expected_filename(vprefix, ds_name)
                path = configs_root / model_name / ds_name / expected
                # Check both existing and planned
                exists = path.exists()
                if not exists and dry_run:
                    # Check old locations
                    for candidate_dir in (configs_root / model_name).iterdir():
                        if not candidate_dir.is_dir():
                            continue
                        canonical_check = _normalize_folder_name(candidate_dir.name)
                        if canonical_check == ds_name:
                            for yf in candidate_dir.glob("*.yaml"):
                                if yf.name.lower().startswith(vprefix.lower()):
                                    if not any(t in yf.name.lower()
                                               for t in ["_eval_pumch", "_eval_pwh", "_eval_tss"]):
                                        exists = True
                                        break
                        if exists:
                            break
                row += f" {'  OK' if exists else 'MISS':>8}"
            print(row)

    if dry_run:
        print()
        print("This was a DRY RUN. Re-run with --apply to execute changes.")


if __name__ == "__main__":
    main()
