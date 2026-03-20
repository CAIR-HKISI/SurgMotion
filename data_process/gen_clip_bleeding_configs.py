#!/usr/bin/env python3
"""
从每个模型的 *_64f_bleeding.yaml 自动生成 *_clip_bleeding.yaml,
将数据集替换为 Bleeding V2 whole-video clips (60帧/视频)。

用法:
    python data_process/gen_clip_bleeding_configs.py              # dry-run 预览
    python data_process/gen_clip_bleeding_configs.py --write      # 实际写入
"""

import argparse
import os
import re
from pathlib import Path

CONFIG_ROOT = "configs/foundation_model_probing"

OLD_TRAIN = "data/Surge_Frames/Bleeding_Dataset_70_30/clips_64f/train_dense_64f_detailed.csv"
OLD_VAL = "data/Surge_Frames/Bleeding_Dataset_70_30/clips_64f/test_dense_64f_detailed.csv"
NEW_TRAIN = "data/Surge_Frames/Bleeding_V2/clips_whole/train_clips.csv"
NEW_VAL = "data/Surge_Frames/Bleeding_V2/clips_whole/val_clips.csv"


def transform_yaml(content: str) -> str:
    """Apply all substitutions to produce a clip config."""
    out = content

    # 1. dataset paths
    out = out.replace(OLD_TRAIN, NEW_TRAIN)
    out = out.replace(OLD_VAL, NEW_VAL)

    # 2. frames_per_clip: 64 → 60
    out = re.sub(r"frames_per_clip:\s*64", "frames_per_clip: 64", out)

    # 3. tag / wandb.name: _64f_bleeding → _clip_bleeding, _64f → _clip
    out = out.replace("_64f_bleeding", "_clip_bleeding")
    out = out.replace("_64f", "_clip")

    # 4. folder: append _clip before _bleeding (if not already there)
    out = re.sub(
        r"(folder:\s*logs/foundation/\S+?)(_bleeding)",
        r"\1_clip\2",
        out,
    )

    # 5. remove wandb.id line to avoid colliding with existing runs
    out = re.sub(r"\n\s*id:\s*\S+", "", out)

    # 6. notes
    out = out.replace(
        "Multi-head probing on Autobleeding dataset",
        "Multi-head probing on Bleeding V2 (per-video, 20fps, 60f)",
    )

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="Actually write files (default: dry-run)")
    args = parser.parse_args()

    root = Path(CONFIG_ROOT)
    sources = sorted(root.rglob("*_64f_bleeding.yaml"))

    if not sources:
        print(f"No *_64f_bleeding.yaml found under {CONFIG_ROOT}")
        return

    print(f"Found {len(sources)} source configs:\n")

    for src in sources:
        dst_name = src.name.replace("_64f_bleeding", "_clip_bleeding")
        dst = src.with_name(dst_name)

        content = src.read_text()
        new_content = transform_yaml(content)

        status = "OVERWRITE" if dst.exists() else "CREATE"
        print(f"  [{status}] {src.relative_to('.')} -> {dst.name}")

        if args.write:
            dst.write_text(new_content)
            print(f"           Written: {dst}")

    if not args.write:
        print(f"\nDry-run complete. Re-run with --write to apply.")
    else:
        print(f"\nDone. {len(sources)} clip configs generated.")


if __name__ == "__main__":
    main()
