#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dense Clip Generation Module
-----------------------------
Generate sliding-window dense clips from frame-level metadata CSVs.

This module implements the dense sampling strategy for surgical workflow recognition:
- Sliding window with configurable window size and stride
- Causal window: last frame is the target frame, preceding frames are context
- Padding at video start when window is incomplete (replicates last frame of current window)
- Clip label is determined by the last frame's label (supports online prediction)

Input CSV format (frame-level metadata):
    Case_ID,Frame_Path,Phase_GT,Phase_Name
    1,data/Surge_Frames/Dataset/frames/01/frame_00001.jpg,0,PhaseA
    1,data/Surge_Frames/Dataset/frames/01/frame_00002.jpg,0,PhaseA
    ...

Output:
    - clips_{window_size}f/{split}_dense_{window_size}f_detailed.csv
    - clips_{window_size}f/clip_dense_{window_size}f_info/{split}/*.txt

Usage:
    python gen_clips.py --base_data_path data/Surge_Frames/AutoLaparo --window_size 64

    Or import as module:
        from gen_clips import generate_dense_clips
        generate_dense_clips(base_path, window_size=64, stride=1)
"""

import argparse
import os
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd


def process_video_csv_dense_sampling(
    input_csv_path: str,
    output_csv_path: str,
    clip_info_dir: str,
    window_size: int = 64,
    stride: int = 1,
    fps: int = 1,
    enable_padding: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate dense clips using sliding window sampling.

    For each video/case, slides a temporal window across all frames:
    - Window end = current target frame
    - Window start = end - window_size
    - When window_start < 0, pad with the last frame of current window

    Args:
        input_csv_path: Path to frame-level metadata CSV
        output_csv_path: Path for output dense CSV
        clip_info_dir: Directory to save clip frame-list txt files
        window_size: Number of frames per window (default 64)
        stride: Step size between consecutive windows (default 1)
        fps: Frames per second of the sampled data (default 1)
        enable_padding: Whether to pad incomplete windows at video start
        verbose: Print progress information

    Returns:
        DataFrame with all generated clip metadata
    """
    if not os.path.exists(input_csv_path):
        print(f"[WARN] Input CSV not found: {input_csv_path}, skipping...")
        return pd.DataFrame()

    df = pd.read_csv(input_csv_path)

    required_cols = {"Case_ID", "Frame_Path", "Phase_GT", "Phase_Name"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV missing required columns: {missing}. "
            f"Expected: {required_cols}, Got: {list(df.columns)}"
        )

    video_groups = df.groupby("Case_ID")
    all_clips_data: List[dict] = []

    frames_per_window = window_size * fps
    frames_per_stride = stride * fps

    for case_id, video_df in video_groups:
        video_df = video_df.sort_values("Frame_Path").reset_index(drop=True)
        total_frames = len(video_df)

        if verbose:
            print(f"\nProcessing case {case_id}:")
            print(f"  - Total frames: {total_frames}")
            print(f"  - Window size: {window_size}s ({frames_per_window} frames)")
            print(f"  - Stride: {stride}s ({frames_per_stride} frames)")
            print(f"  - Padding enabled: {enable_padding}")

        clip_count = 0
        start_idx = 0

        while start_idx < total_frames:
            end_idx = start_idx + 1
            start_idx_for_window = end_idx - frames_per_window

            if start_idx_for_window < 0:
                valid_frames = video_df.iloc[0:end_idx].copy()
                actual_frames = len(valid_frames)

                if enable_padding:
                    last_row = valid_frames.iloc[-1]
                    padding_frames = frames_per_window - actual_frames
                    padding_data = [last_row.to_dict()] * padding_frames
                    padding_df = pd.DataFrame(padding_data)
                    clip_frames = pd.concat(
                        [valid_frames, padding_df], ignore_index=True
                    )
                    is_padded = True
                else:
                    if actual_frames < frames_per_window:
                        start_idx += frames_per_stride
                        continue
                    clip_frames = valid_frames
                    padding_frames = 0
                    is_padded = False
            else:
                clip_frames = video_df.iloc[start_idx_for_window:end_idx].copy()
                actual_frames = len(clip_frames)
                padding_frames = 0
                is_padded = False

            last_frame = clip_frames.iloc[-1]
            clip_label = int(last_frame["Phase_GT"])
            clip_phase_name = str(last_frame["Phase_Name"])

            clip_start_time = max(start_idx_for_window, 0) / fps
            clip_end_time = end_idx / fps
            clip_start_time_str = str(timedelta(seconds=int(clip_start_time)))
            clip_end_time_str = str(timedelta(seconds=int(clip_end_time)))

            clip_identifier = (
                f"case{case_id}_c{clip_count:03d}_f{start_idx_for_window:06d}-{end_idx:06d}"
            )
            if is_padded:
                clip_identifier += "_padded"

            os.makedirs(clip_info_dir, exist_ok=True)
            clip_frames_file = os.path.join(clip_info_dir, f"{clip_identifier}.txt")

            with open(clip_frames_file, "w") as f:
                for _, row in clip_frames.iterrows():
                    f.write(f"{row['Frame_Path']}\n")

            clip_info = {
                "clip_path": str(clip_frames_file),
                "label": clip_label,
                "label_name": clip_phase_name,
                "case_id": case_id,
                "clip_idx": clip_count,
                "start_frame": max(start_idx_for_window, 0),
                "end_frame": end_idx,
                "actual_frames": actual_frames,
                "padded_frames": padding_frames,
                "start_time": clip_start_time_str,
                "end_time": clip_end_time_str,
                "duration_seconds": frames_per_window / fps,
                "is_padded": is_padded,
            }
            all_clips_data.append(clip_info)

            start_idx += frames_per_stride
            clip_count += 1

        if verbose:
            print(f"  - Generated {clip_count} clips")

    output_df = pd.DataFrame(all_clips_data)

    if len(output_df) > 0:
        detailed_path = output_csv_path.replace(".csv", "_detailed.csv")
        os.makedirs(os.path.dirname(detailed_path) or ".", exist_ok=True)
        output_df.to_csv(detailed_path, index=True, index_label="Index")

        if verbose:
            print(f"\n=== Processing complete ===")
            print(f"Total clips generated: {len(all_clips_data)}")
            print(f"Output file: {detailed_path}")
    else:
        print(f"[WARN] No clips generated from {input_csv_path}")

    return output_df


def generate_dense_clips(
    base_data_path: str,
    window_size: int = 64,
    stride: int = 1,
    fps: int = 1,
    enable_padding: bool = True,
    subsets: Optional[List[str]] = None,
    verbose: bool = True,
) -> None:
    """
    Generate dense clips for train/val/test subsets.

    Args:
        base_data_path: Root directory containing {train,val,test}_metadata.csv
        window_size: Number of frames per temporal window
        stride: Step size between consecutive windows
        fps: Frames per second of sampled data
        enable_padding: Pad incomplete windows at video start
        subsets: List of subsets to process (default: ["train", "val", "test"])
        verbose: Print progress
    """
    if subsets is None:
        subsets = ["train", "val", "test"]

    output_base_path = os.path.join(base_data_path, f"clips_{window_size}f")
    Path(output_base_path).mkdir(parents=True, exist_ok=True)

    stats = {}

    for subset in subsets:
        input_csv = os.path.join(base_data_path, f"{subset}_metadata.csv")
        if not os.path.exists(input_csv):
            if verbose:
                print(f"\n[SKIP] {subset}_metadata.csv not found in {base_data_path}")
            continue

        if verbose:
            print(f"\n{'='*50}")
            print(f"Processing {subset} set")
            print(f"{'='*50}")

        output_csv = os.path.join(
            output_base_path, f"{subset}_dense_{window_size}f.csv"
        )
        clip_dir = os.path.join(
            output_base_path, f"clip_dense_{window_size}f_info/{subset}"
        )

        result_df = process_video_csv_dense_sampling(
            input_csv_path=input_csv,
            output_csv_path=output_csv,
            clip_info_dir=clip_dir,
            window_size=window_size,
            stride=stride,
            fps=fps,
            enable_padding=enable_padding,
            verbose=verbose,
        )

        stats[subset] = len(result_df)

    if verbose and stats:
        print(f"\n{'='*50}")
        print("Summary")
        print(f"{'='*50}")
        for subset, count in stats.items():
            print(f"  {subset}: {count} clips")
        print(f"  Output directory: {output_base_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate dense clips with sliding window sampling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate 64-frame clips with stride 1
    python gen_clips.py --base_data_path data/Surge_Frames/AutoLaparo

    # Generate 16-frame clips for triplet recognition
    python gen_clips.py --base_data_path data/Surge_Frames/Cholec80 --window_size 16

    # Disable padding (skip incomplete windows)
    python gen_clips.py --base_data_path data/Surge_Frames/PitVis --no_padding
        """,
    )
    parser.add_argument(
        "--base_data_path",
        type=str,
        required=True,
        help="Base path containing {train,val,test}_metadata.csv",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=64,
        help="Temporal window size in frames (default: 64)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride between consecutive windows (default: 1)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="Frames per second of sampled data (default: 1)",
    )
    parser.add_argument(
        "--no_padding",
        action="store_true",
        help="Disable padding (skip incomplete windows at video start)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    print("=" * 50)
    print("Dense Clip Generation")
    print("=" * 50)
    print(f"Data path: {args.base_data_path}")
    print(f"Window size: {args.window_size} frames")
    print(f"Stride: {args.stride} frames")
    print(f"FPS: {args.fps}")
    print(f"Padding: {not args.no_padding}")
    print("=" * 50)

    generate_dense_clips(
        base_data_path=args.base_data_path,
        window_size=args.window_size,
        stride=args.stride,
        fps=args.fps,
        enable_padding=not args.no_padding,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
