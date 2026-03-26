#!/usr/bin/env python3
"""
Generate ViT-Giant configs from existing ViT-Large configs

Reads all surgmotion_vitl_64f_*.yaml configs and generates corresponding
surgmotion_vitg_64f_*.yaml configs with the following changes:
- vitl -> vitg in names and paths
- model_name: vit_large -> vit_giant_xformers
- checkpoint: ckpts/SurgMotion-vitl.pt -> ckpts/SurgMotion-vitg.pt

Usage:
    python scripts/generate_vitg_configs.py
    python scripts/generate_vitg_configs.py --dry-run
"""

import argparse
import re
from pathlib import Path


def generate_vitg_config(vitl_content: str) -> str:
    """Convert vitl config content to vitg config."""
    
    content = vitl_content
    
    # Replace folder path
    content = re.sub(
        r'folder: logs/foundation/surgmotion_vitl_',
        'folder: logs/foundation/surgmotion_vitg_',
        content
    )
    
    # Replace tag
    content = re.sub(
        r'tag: surgmotion_vitl_64f_',
        'tag: surgmotion_vitg_64f_',
        content
    )
    
    # Replace wandb name
    content = re.sub(
        r'name: surgmotion_vitl_64f_',
        'name: surgmotion_vitg_64f_',
        content
    )
    
    # Replace wandb tags
    content = re.sub(
        r'- surgmotion_vitl_64f',
        '- surgmotion_vitg_64f',
        content
    )
    
    # Replace checkpoint path
    content = re.sub(
        r'checkpoint: ckpts/SurgMotion-vitl\.pt',
        'checkpoint: ckpts/SurgMotion-vitg.pt',
        content
    )
    
    # Replace model name (exact match to avoid partial replacement)
    content = re.sub(
        r'model_name: vit_large',
        'model_name: vit_giant_xformers',
        content
    )
    
    return content


def main():
    parser = argparse.ArgumentParser(description="Generate ViT-Giant configs")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Dry run mode (print what would be done)")
    parser.add_argument("--base-dir", type=str, 
                        default="configs/foundation_model_probing/surgmotion",
                        help="Base directory for surgmotion configs")
    args = parser.parse_args()
    
    # Find all vitl configs
    script_dir = Path(__file__).parent.parent
    base_dir = script_dir / args.base_dir
    
    if not base_dir.exists():
        print(f"Error: Base directory not found: {base_dir}")
        return
    
    vitl_configs = list(base_dir.rglob("surgmotion_vitl_64f_*.yaml"))
    
    print("=" * 70)
    print("GENERATE VIT-GIANT CONFIGS")
    print("=" * 70)
    print(f"Base directory: {base_dir}")
    print(f"Found {len(vitl_configs)} ViT-Large configs")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'PRODUCTION'}")
    print("=" * 70)
    
    generated = 0
    skipped = 0
    
    for vitl_path in sorted(vitl_configs):
        # Determine output path
        vitg_filename = vitl_path.name.replace("surgmotion_vitl_64f_", "surgmotion_vitg_64f_")
        vitg_path = vitl_path.parent / vitg_filename
        
        # Check if already exists
        if vitg_path.exists():
            print(f"[SKIP] {vitg_path.relative_to(script_dir)} (already exists)")
            skipped += 1
            continue
        
        # Read vitl config
        vitl_content = vitl_path.read_text()
        
        # Generate vitg config
        vitg_content = generate_vitg_config(vitl_content)
        
        # Write or print
        if args.dry_run:
            print(f"\n[WOULD CREATE] {vitg_path.relative_to(script_dir)}")
            print("-" * 50)
            # Show diff summary
            print(f"  folder: ...surgmotion_vitl_* -> ...surgmotion_vitg_*")
            print(f"  tag: surgmotion_vitl_64f_* -> surgmotion_vitg_64f_*")
            print(f"  checkpoint: ckpts/SurgMotion-vitl.pt -> ckpts/SurgMotion-vitg.pt")
            print(f"  model_name: vit_large -> vit_giant_xformers")
        else:
            vitg_path.write_text(vitg_content)
            print(f"[CREATED] {vitg_path.relative_to(script_dir)}")
            generated += 1
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Generated: {generated}")
    print(f"Skipped (already exist): {skipped}")
    print(f"Total ViT-Large configs: {len(vitl_configs)}")
    
    if args.dry_run and generated == 0 and skipped < len(vitl_configs):
        print(f"\nRun without --dry-run to create {len(vitl_configs) - skipped} config files")


if __name__ == "__main__":
    main()
