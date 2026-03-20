#!/usr/bin/env python3
"""
Bleeding V2 Clip 生成器
每个视频 case（~61 帧 @20fps）直接作为一个独立 clip，
无需滑动窗口。产出与 SurgicalVideoDataset 兼容的 CSV + txt。

用法:
    python data_process/gen_clips_v2.py
    python data_process/gen_clips_v2.py --base_data_path data/Surge_Frames/Bleeding_V2
"""

import os
import argparse
from pathlib import Path

import pandas as pd


def generate_whole_video_clips(
    input_csv_path: str,
    output_csv_path: str,
    clip_info_dir: str,
):
    """
    将每个 Case（视频）的全部帧打包成 1 个 clip txt，
    并生成与 SurgicalVideoDataset 兼容的汇总 CSV。
    """
    df = pd.read_csv(input_csv_path)
    video_groups = df.groupby("Case_ID")
    all_clips = []

    os.makedirs(clip_info_dir, exist_ok=True)

    for case_id, video_df in video_groups:
        video_df = video_df.sort_values("Frame_Path").reset_index(drop=True)
        case_name = video_df.iloc[0]["Case_Name"]
        label = int(video_df.iloc[0]["Phase_GT"])
        label_name = video_df.iloc[0]["Phase_Name"]
        total_frames = len(video_df)

        txt_filename = f"{case_name}.txt"
        txt_path = os.path.join(clip_info_dir, txt_filename)

        with open(txt_path, "w") as f:
            for frame_path in video_df["Frame_Path"]:
                f.write(f"{frame_path}\n")

        all_clips.append({
            "clip_path": txt_path,
            "label": label,
            "label_name": label_name,
            "case_id": case_id,
            "case_name": case_name,
            "clip_idx": 0,
            "start_frame": 0,
            "end_frame": total_frames,
            "actual_frames": total_frames,
            "padded_frames": 0,
            "is_padded": False,
        })

    output_df = pd.DataFrame(all_clips)
    output_df.index.name = "Index"
    output_df.to_csv(output_csv_path, index=True)

    print(f"  生成 {len(all_clips)} 个 clip -> {output_csv_path}")
    pos = sum(1 for c in all_clips if c["label"] == 1)
    neg = len(all_clips) - pos
    print(f"  类别分布: pos(bleeding)={pos}, neg(non_bleeding)={neg}")

    return output_df


def process_all_splits(base_data_path: str, output_base_path: str):
    """处理 train / val / test 三个子集。"""
    subsets = ["train", "val", "test"]
    for subset in subsets:
        input_csv = os.path.join(base_data_path, f"{subset}_metadata.csv")
        if not os.path.exists(input_csv):
            print(f"\n=== 跳过 {subset} (未找到 {input_csv}) ===")
            continue

        print(f"\n=== 处理 {subset} 集 ===")
        output_csv = os.path.join(output_base_path, f"{subset}_clips.csv")
        clip_dir = os.path.join(output_base_path, f"clip_info/{subset}")

        generate_whole_video_clips(
            input_csv_path=input_csv,
            output_csv_path=output_csv,
            clip_info_dir=clip_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate whole-video clips for Bleeding V2 dataset"
    )
    parser.add_argument(
        "--base_data_path", type=str,
        default="data/Surge_Frames/Bleeding_V2",
        help="Base path containing train/val/test_metadata.csv",
    )
    args = parser.parse_args()

    output_dir = os.path.join(args.base_data_path, "clips_whole")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=== Bleeding V2 Clip 生成 (whole-video mode) ===")
    print(f"数据路径: {args.base_data_path}")
    print(f"输出路径: {output_dir}")

    process_all_splits(
        base_data_path=args.base_data_path,
        output_base_path=output_dir,
    )

    print("\n=== 完成 ===")
