#!/usr/bin/env python3
"""
基于已有 64 帧 clip 元信息，生成 Cholec80 器械 anticipation 纯回归数据集。

输入:
  - clip 元信息 CSV: data/Surge_Frames/Cholec80/clips_64f_anticipation/{train,val,test}_dense_64f_detailed.csv
  - 标注: evals/surgical_phase_anticipation/cholec80_anticipation_annotations/videoXX-phase.txt

输出:
  - data/Surge_Frames/Cholec80/clips_64f_instrument_anticipation/{train,val,test}_dense_64f_detailed.csv

说明:
  - 复用已有 clip_path，不重复生成 clip txt 文件
  - 目标为 5 维器械 anticipation 回归值，保持原始标注值，不做归一化
"""

import argparse
from pathlib import Path

import pandas as pd


INSTRUMENTS = [
    "Bipolar",
    "Scissors",
    "Clipper",
    "Irrigator",
    "SpecimenBag",
]


def load_annotation(ann_path):
    if not ann_path.exists():
        return None
    df = pd.read_csv(ann_path, sep="\t")
    if "Frame" not in df.columns:
        return None
    df = df.rename(columns={"Frame": "frame_idx"})
    if not all(col in df.columns for col in INSTRUMENTS):
        return None
    return df


def keep_raw_regression_value(val):
    return float(val)


def infer_video_name(row):
    if "video_name" in row and pd.notna(row["video_name"]):
        return str(row["video_name"])
    return f"video{int(row['case_id']):02d}"


def build_instrument_dataset(split, source_dir, ann_dir, output_dir):
    source_csv = source_dir / f"{split}_dense_64f_detailed.csv"
    df = pd.read_csv(source_csv)

    out_df = df.copy()
    if "video_name" not in out_df.columns:
        out_df.insert(2, "video_name", out_df["case_id"].map(lambda x: f"video{int(x):02d}"))

    ann_cache = {}
    for instrument in INSTRUMENTS:
        out_df[f"ant_reg_{instrument}"] = 5.0

    for idx, row in out_df.iterrows():
        video_name = infer_video_name(row)
        if video_name not in ann_cache:
            ann_cache[video_name] = load_annotation(ann_dir / f"{video_name}-phase.txt")

        ann_df = ann_cache[video_name]
        if ann_df is None:
            continue

        last_frame_idx = int(row["end_frame"]) - 1
        matched = ann_df.loc[ann_df["frame_idx"] == last_frame_idx]
        if matched.empty:
            continue

        ann_row = matched.iloc[0]
        for instrument in INSTRUMENTS:
            out_df.at[idx, f"ant_reg_{instrument}"] = keep_raw_regression_value(ann_row[instrument])

    keep_cols = [
        "Index",
        "clip_path",
        "video_name",
        "case_id",
        "clip_idx",
        "start_frame",
        "end_frame",
        "actual_frames",
        "padded_frames",
        "start_time",
        "end_time",
        "duration_seconds",
        "is_padded",
    ] + [f"ant_reg_{instrument}" for instrument in INSTRUMENTS]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{split}_dense_64f_detailed.csv"
    out_df[keep_cols].to_csv(output_csv, index=False)
    print(f"[OK] {split}: {len(out_df)} clips -> {output_csv}")
    for instrument in INSTRUMENTS:
        col = f"ant_reg_{instrument}"
        print(
            f"  {instrument}: mean={out_df[col].mean():.4f}, "
            f"min={out_df[col].min():.4f}, max={out_df[col].max():.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="生成 Cholec80 器械 anticipation 纯回归数据集")
    parser.add_argument(
        "--source_dir",
        type=str,
        default="data/Surge_Frames/Cholec80/clips_64f_anticipation",
        help="已有 phase anticipation clip 元信息目录",
    )
    parser.add_argument(
        "--ann_dir",
        type=str,
        default="evals/surgical_phase_anticipation/cholec80_anticipation_annotations",
        help="benchmark 标注目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/Surge_Frames/Cholec80/clips_64f_instrument_anticipation",
        help="器械 anticipation 回归数据集输出目录",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    ann_dir = Path(args.ann_dir)
    output_dir = Path(args.output_dir)

    for split in ["train", "val", "test"]:
        build_instrument_dataset(split, source_dir, ann_dir, output_dir)


if __name__ == "__main__":
    main()
