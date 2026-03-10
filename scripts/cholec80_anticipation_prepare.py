#!/usr/bin/env python3
"""
Cholec80 手术流程预测 (Phase Anticipation) 纯回归数据处理脚本。

从视频帧和 anticipation 标注生成 64 帧 clip 列表，以及对应的 7 维连续回归目标。

输入:
  - 视频帧: data/Surge_Frames/Cholec80/frames
  - 标注: evals/surgical_phase_anticipation/cholec80_anticipation_annotations

输出:
  - clips_64f_anticipation/clip_dense_64f_info/{train,val,test}/*.txt  (每行一个帧路径)
  - clips_64f_anticipation/{train,val,test}_dense_64f_detailed.csv

CSV 仅保留训练所需元信息与 7 个连续目标 `ant_reg_*`。
标注值保持 benchmark 原始语义，不做归一化:
  - 0.0: 当前阶段
  - 0<v<1: 目标阶段位于 anticipation horizon 内
  - 5.0: 目标阶段位于 anticipation horizon 外
"""

import argparse
import os
import pandas as pd
from datetime import timedelta
from tqdm import tqdm

CHOLEC80_PHASES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]

def load_annotation(ann_path):
    """加载标注文件，返回 DataFrame"""
    if not os.path.exists(ann_path):
        return None
    df = pd.read_csv(ann_path, sep="\t")
    if "Frame" not in df.columns:
        return None
    df = df.rename(columns={"Frame": "frame_idx"})
    available = [c for c in CHOLEC80_PHASES if c in df.columns]
    if not available:
        return None
    return df


def annotation_to_regression_targets(ann_row):
    """
    将单帧标注转换为 7 维回归目标。

    Returns:
        target_reg: list[float], 长度 7
    """
    target_reg = []

    for phase_name in CHOLEC80_PHASES:
        val = float(ann_row.get(phase_name, 5.0))
        target_reg.append(val)

    return target_reg


def process_video(
    video_name,
    case_id,
    frames_dir,
    ann_dir,
    clip_info_dir,
    window_size=64,
    stride=1,
):
    """处理单个视频，生成 clips 和对应标签"""
    video_frames_dir = os.path.join(frames_dir, video_name)
    if not os.path.isdir(video_frames_dir):
        return []

    frame_files = sorted(
        f for f in os.listdir(video_frames_dir) if f.endswith((".jpg", ".jpeg", ".png"))
    )
    num_frames = len(frame_files)
    if num_frames == 0:
        return []

    ann_path = os.path.join(ann_dir, f"{video_name}-phase.txt")
    ann_df = load_annotation(ann_path)

    if ann_df is None:
        print(f"  [WARN] 无标注文件: {ann_path}，跳过")
        return []

    ann_dict = {}
    for _, row in ann_df.iterrows():
        ann_dict[int(row["frame_idx"])] = row

    all_clips = []
    clip_count = 0
    start_idx = 0

    while start_idx < num_frames:
        end_idx = min(start_idx + window_size, num_frames)
        start_for_window = max(0, end_idx - window_size)
        actual_frames = end_idx - start_for_window

        frame_indices = list(range(start_for_window, end_idx))
        is_padded = actual_frames < window_size
        if is_padded:
            pad_count = window_size - actual_frames
            last_idx = end_idx - 1
            frame_indices = frame_indices + [last_idx] * pad_count

        frame_paths = []
        for idx in frame_indices:
            fname = f"{video_name}_{idx + 1:08d}.jpg"
            rel_path = os.path.join(frames_dir, video_name, fname)
            frame_paths.append(rel_path)

        # 取窗口最后一帧的标注
        last_frame_idx = end_idx - 1
        if last_frame_idx in ann_dict:
            ann_row = ann_dict[last_frame_idx]
            target_reg = annotation_to_regression_targets(ann_row)
        else:
            target_reg = [5.0] * 7

        clip_id = f"case{case_id}_c{clip_count:03d}_f{start_for_window:06d}-{end_idx:06d}"
        if is_padded:
            clip_id += "_padded"

        os.makedirs(clip_info_dir, exist_ok=True)
        clip_txt_path = os.path.join(clip_info_dir, f"{clip_id}.txt")
        with open(clip_txt_path, "w") as f:
            for p in frame_paths:
                f.write(p + "\n")

        clip_info = {
            "clip_path": clip_txt_path,
            "video_name": video_name,
            "case_id": case_id,
            "clip_idx": clip_count,
            "start_frame": start_for_window,
            "end_frame": end_idx,
            "actual_frames": actual_frames,
            "padded_frames": window_size - actual_frames if is_padded else 0,
            "start_time": str(timedelta(seconds=start_for_window)),
            "end_time": str(timedelta(seconds=end_idx)),
            "duration_seconds": window_size,
            "is_padded": is_padded,
        }
        for i, pname in enumerate(CHOLEC80_PHASES):
            clip_info[f"ant_reg_{pname}"] = target_reg[i]

        all_clips.append(clip_info)

        start_idx += stride
        clip_count += 1
        if start_idx >= num_frames:
            break

    return all_clips


def build_video_to_split(metadata_dir):
    """从 metadata CSVs 构建 video_name -> split 映射"""
    video_to_split = {}

    train_path = os.path.join(metadata_dir, "train_metadata.csv")
    if os.path.exists(train_path):
        df = pd.read_csv(train_path)
        if "Case_Name" in df.columns:
            for vn in df["Case_Name"].unique():
                video_to_split[vn] = "train"

    val_path = os.path.join(metadata_dir, "val_metadata.csv")
    test_path = os.path.join(metadata_dir, "test_metadata.csv")
    extra_videos = set()
    for p in [val_path, test_path]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "Case_Name" in df.columns:
                extra_videos.update(df["Case_Name"].unique())

    for vn in extra_videos:
        if vn in video_to_split:
            continue
        try:
            vid_num = int(vn.replace("video", ""))
        except ValueError:
            vid_num = 50
        if 41 <= vid_num <= 48:
            video_to_split[vn] = "val"
        else:
            video_to_split[vn] = "test"

    if not video_to_split:
        for i in range(1, 81):
            vn = f"video{i:02d}"
            if i <= 40:
                video_to_split[vn] = "train"
            elif i <= 48:
                video_to_split[vn] = "val"
            else:
                video_to_split[vn] = "test"

    return video_to_split


def main():
    parser = argparse.ArgumentParser(
        description="Cholec80 手术流程预测数据处理：生成 clips_64f 格式（含 anticipation 标签）"
    )
    parser.add_argument("--frames_dir", type=str,
                        default="data/Surge_Frames/Cholec80/frames")
    parser.add_argument("--ann_dir", type=str,
                        default="evals/surgical_phase_anticipation/cholec80_anticipation_annotations")
    parser.add_argument("--output_dir", type=str,
                        default="data/Surge_Frames/Cholec80/clips_64f_anticipation")
    parser.add_argument("--metadata_dir", type=str,
                        default="data/Surge_Frames/Cholec80")
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    video_to_split = build_video_to_split(args.metadata_dir)

    subsets = ["train", "val", "test"]
    all_data = {s: [] for s in subsets}

    for video_name in tqdm(sorted(video_to_split.keys()), desc="Processing videos"):
        split = video_to_split[video_name]
        case_id_str = video_name.replace("video", "")
        try:
            case_id = int(case_id_str)
        except ValueError:
            case_id = hash(video_name) % (2 ** 31)

        clip_info_dir = os.path.join(
            args.output_dir, f"clip_dense_{args.window_size}f_info", split
        )
        clips = process_video(
            video_name=video_name,
            case_id=case_id,
            frames_dir=args.frames_dir,
            ann_dir=args.ann_dir,
            clip_info_dir=clip_info_dir,
            window_size=args.window_size,
            stride=args.stride,
        )
        all_data[split].extend(clips)

    for subset in subsets:
        rows = all_data[subset]
        if not rows:
            print(f"[WARN] {subset} 集无数据")
            continue
        df = pd.DataFrame(rows)
        csv_path = os.path.join(
            args.output_dir, f"{subset}_dense_{args.window_size}f_detailed.csv"
        )
        df.to_csv(csv_path, index=True, index_label="Index")
        print(f"[OK] {subset}: {len(df)} clips -> {csv_path}")

        # 打印回归目标统计
        for pname in CHOLEC80_PHASES:
            reg_col = f"ant_reg_{pname}"
            print(
                f"  {pname}: mean={df[reg_col].mean():.4f}, "
                f"min={df[reg_col].min():.4f}, max={df[reg_col].max():.4f}"
            )

    print(f"\n=== 完成 ===")
    print(f"输出目录: {args.output_dir}")
    print(f"\nCSV 中新增列:")
    print(f"  ant_reg_{{PhaseName}}: 纯回归目标 (0.0=present, 0~1=inside, 1.0=outside)")


if __name__ == "__main__":
    main()
