#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SurgicalActions160 Preprocessing Pipeline
-------------------------------------------
Aligned with NSJepa_jinlin/data_process/surgicalactions160_prepare.py.

Data organization:
  - Original videos: data/Landscopy/SurgicalActions160/<action>/*.mp4
  - Renumbered videos (optional): data/Landscopy/SurgicalActions160_renumbered/
  - Frames: data/Surge_Frames/SurgicalActions160_v1/frames/fps{fps}/...

NOTE: Unlike most other datasets, SurgicalActions160 DOES require frame extraction
from videos. The --step all will run the full pipeline including extraction.

Pipeline Steps:
  --step all (default): rename + frames + metadata + clips
  --step rename:        Copy and rename videos to consecutive numbers
  --step frames:        Extract frames from videos
  --step metadata:      Build frame-level metadata CSV
  --step clips:         Generate dense sliding-window clips

Output structure:
  <output_dir>/   (default: data/Surge_Frames/SurgicalActions160_v1)
    clip_infos_{fps}/              # One txt per video
    train_metadata.csv             # Frame-level metadata
    val_metadata.csv               # Same as fold 0 test
    test_metadata.csv              # Same as fold 0 test
    train_metadata_fold{i}_{fps}.csv
    test_metadata_fold{i}_{fps}.csv
    clips_64f/                     # Dense clips
      train_dense_64f_detailed.csv
      ...

Usage:
    python surgicalactions160_prepare.py --step all
    python surgicalactions160_prepare.py --step metadata
    python surgicalactions160_prepare.py --step clips --window_size 64
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from gen_clips import generate_dense_clips

BASE_DIR = Path("data/Surge_Frames/SurgicalActions160_v1")


def clean_videos(
    src_root: str,
    dst_root: str,
) -> Path:
    """Copy and rename videos by number, preserving subdirectory structure."""
    src_root_path = Path(src_root)
    dst_root_path = Path(dst_root)
    dst_root_path.mkdir(parents=True, exist_ok=True)

    video_files = list(src_root_path.rglob("*.mp4"))
    video_files.sort()

    print(f"[INFO] Found {len(video_files)} videos, copying and renaming...")

    for folder in sorted({v.parent for v in video_files}):
        rel = folder.relative_to(src_root_path)
        out_subdir = dst_root_path / rel
        out_subdir.mkdir(parents=True, exist_ok=True)

        vids = sorted(folder.glob("*.mp4"))
        for idx, vid in enumerate(vids, start=1):
            new_name = f"{idx:05d}.mp4"
            new_path = out_subdir / new_name
            shutil.copy2(vid, new_path)
        print(f"[INFO] {rel}: {len(vids)} videos copied")

    print("[INFO] Video renaming completed")
    return dst_root_path


def videos_to_frames(
    input_path: Path,
    output_path: Path,
    fps: int = 30,
    pattern: str = "*.mp4",
    debug: bool = False,
) -> None:
    """
    Extract frames from all *.mp4 under input_path into output_path.
    Preserves subdirectory structure.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    video_files = list(input_path.rglob(pattern))
    video_files.sort()

    if not video_files:
        print(f"[WARN] No videos found matching {pattern} in {input_path}.")
        return

    print(f"\n[INFO] Found {len(video_files)} videos, extracting frames at {fps} fps...")
    failed_videos: List[str] = []

    for vid_path in tqdm(video_files, desc="Extracting frames"):
        rel_path = vid_path.relative_to(input_path).parent
        out_folder = output_path / rel_path / vid_path.stem
        out_folder.mkdir(parents=True, exist_ok=True)
        output_pattern = out_folder / f"{vid_path.stem}_%08d.jpg"

        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", str(vid_path.resolve()),
            "-vf", f"fps={fps},scale=512:-1:flags=bicubic",
            "-vsync", "2", "-qscale:v", "2",
            str(output_pattern),
        ]

        if debug:
            print(f"[DEBUG] FFmpeg: {' '.join(ffmpeg_cmd)}")

        try:
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Frame extraction failed: {vid_path}")
            if debug and e.stderr:
                print(e.stderr.decode("utf-8", errors="ignore")[:500])
            failed_videos.append(str(vid_path))

    print("[INFO] Frame extraction finished")
    if failed_videos:
        fail_log = output_path / "failed_videos.txt"
        fail_log.write_text("\n".join(failed_videos), encoding="utf-8")
        print(f"[WARN] {len(failed_videos)} videos failed; see {fail_log}")


def generate_clip_txt(video_dir: Path, txt_path: Path) -> List[str]:
    """
    Generate txt listing all frame paths for a video.
    Returns list of frame paths.
    """
    frame_files = sorted(
        [p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")],
        key=lambda p: p.name,
    )
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    frame_paths = [str(fp).replace("\\", "/") for fp in frame_files]
    with txt_path.open("w", encoding="utf-8") as f:
        for fp in frame_paths:
            f.write(fp + "\n")
    return frame_paths


def build_frame_level_metadata(
    frames_root: Path,
    clip_infos_dir: Path,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Build frame-level metadata with columns: Case_ID, Frame_Path, Phase_GT, Phase_Name.
    Label is assigned by action subdirectory (sorted alphabetically).
    """
    all_rows: List[dict] = []
    case_id_counter = 0

    action_dirs = sorted(
        [d for d in frames_root.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )

    for label_id, action_dir in enumerate(action_dirs):
        label_name = action_dir.name
        video_dirs = sorted(
            [d for d in action_dir.iterdir() if d.is_dir()],
            key=lambda p: p.name,
        )

        for video_dir in tqdm(video_dirs, desc=f"Processing {label_name}"):
            rel_video = video_dir.relative_to(frames_root)
            txt_parent = clip_infos_dir / rel_video.parent
            txt_parent.mkdir(parents=True, exist_ok=True)
            txt_path = txt_parent / f"{video_dir.name}.txt"

            frame_paths = generate_clip_txt(video_dir, txt_path)
            if not frame_paths:
                if debug:
                    print(f"[DEBUG] No frames in {video_dir}")
                continue

            for fp in frame_paths:
                all_rows.append({
                    "Case_ID": case_id_counter,
                    "Frame_Path": fp,
                    "Phase_GT": label_id,
                    "Phase_Name": label_name,
                })

            case_id_counter += 1

    if all_rows:
        df = pd.DataFrame(all_rows)
        df = df.sort_values(["Case_ID", "Frame_Path"]).reset_index(drop=True)
        return df
    else:
        return pd.DataFrame(columns=["Case_ID", "Frame_Path", "Phase_GT", "Phase_Name"])


def make_4_folds(df: pd.DataFrame, seed: int = 42) -> List[List[int]]:
    """
    Perform 4-fold splitting based on Case_ID, distributing evenly by label.
    Returns list of lists, each containing Case_IDs for that fold.
    """
    rng = random.Random(seed)

    case_to_label = df.groupby("Case_ID")["Phase_GT"].first().to_dict()
    label_to_cases: Dict[int, List[int]] = defaultdict(list)

    for case_id, label in case_to_label.items():
        label_to_cases[label].append(case_id)

    folds: List[List[int]] = [[] for _ in range(4)]

    for _, cases in label_to_cases.items():
        rng.shuffle(cases)
        for i, case_id in enumerate(cases):
            folds[i % 4].append(case_id)

    return folds


def save_metadata_csvs(
    output_dir: Path,
    df: pd.DataFrame,
    fps_tag: str,
    seed: int = 42,
) -> None:
    """Save frame-level metadata CSVs and 4-fold splits."""
    if len(df) == 0:
        print("[WARN] No metadata rows to save")
        return

    all_csv = output_dir / f"metadata_{fps_tag}.csv"
    df.to_csv(all_csv, index=False)
    print(f"[INFO] Saved total metadata ({len(df)} rows) to {all_csv}")

    folds = make_4_folds(df, seed=seed)

    for i in range(4):
        test_cases = set(folds[i])
        train_cases = set(c for j, fold in enumerate(folds) if j != i for c in fold)

        train_df = df[df["Case_ID"].isin(train_cases)].copy()
        test_df = df[df["Case_ID"].isin(test_cases)].copy()

        train_csv = output_dir / f"train_metadata_fold{i}_{fps_tag}.csv"
        test_csv = output_dir / f"test_metadata_fold{i}_{fps_tag}.csv"

        train_df.to_csv(train_csv, index=False)
        test_df.to_csv(test_csv, index=False)

        print(f"[INFO] Fold {i}: train={len(train_df)} frames, test={len(test_df)} frames")

    train_df_fold0 = df[df["Case_ID"].isin(set(c for j, fold in enumerate(folds) if j != 0 for c in fold))]
    test_df_fold0 = df[df["Case_ID"].isin(set(folds[0]))]

    train_csv = output_dir / "train_metadata.csv"
    val_csv = output_dir / "val_metadata.csv"
    test_csv = output_dir / "test_metadata.csv"

    train_df_fold0.to_csv(train_csv, index=False)
    test_df_fold0.to_csv(val_csv, index=False)
    test_df_fold0.to_csv(test_csv, index=False)

    print(f"[INFO] Saved standard splits: train={len(train_df_fold0)}, val/test={len(test_df_fold0)} frames")


def main():
    parser = argparse.ArgumentParser(
        description="SurgicalActions160: End-to-end preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python surgicalactions160_prepare.py --step all
    python surgicalactions160_prepare.py --step metadata
    python surgicalactions160_prepare.py --step clips --window_size 64
        """,
    )
    parser.add_argument(
        "--step",
        choices=["all", "rename", "frames", "metadata", "clips"],
        default="all",
        help="Pipeline step to run (default: all)",
    )
    parser.add_argument(
        "--src_root",
        type=str,
        default="data/Landscopy/SurgicalActions160",
        help="Original video root directory",
    )
    parser.add_argument(
        "--dst_root",
        type=str,
        default="data/Landscopy/SurgicalActions160_renumbered",
        help="Renamed video save directory",
    )
    parser.add_argument(
        "--frames_root",
        type=str,
        default=str(BASE_DIR / "frames"),
        help="Frame extraction save directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(BASE_DIR),
        help="Output directory for metadata and clips",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="FPS for frame extraction (default: 1)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=64,
        help="Window size for dense clip generation (default: 64)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride for dense clip generation (default: 1)",
    )
    parser.add_argument(
        "--no_padding",
        action="store_true",
        help="Disable padding for incomplete windows",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="4-fold split random seed",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug output",
    )
    args = parser.parse_args()

    fps_tag = f"fps{args.fps}"
    frames_root = Path(args.frames_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SurgicalActions160 Preprocessing Pipeline")
    print("=" * 60)

    if args.step in ("all", "rename"):
        print("\n[STEP 0] Renaming videos...")
        src_root = Path(args.src_root)
        if src_root.exists():
            clean_videos(args.src_root, args.dst_root)
        else:
            print(f"[SKIP] Source directory not found: {src_root}")

    if args.step in ("all", "frames"):
        print("\n[STEP 1] Extracting frames from videos...")
        dst_root = Path(args.dst_root)
        src_root = Path(args.src_root)
        if dst_root.exists():
            videos_to_frames(dst_root, frames_root / fps_tag, fps=args.fps, debug=args.debug)
        elif src_root.exists():
            videos_to_frames(src_root, frames_root / fps_tag, fps=args.fps, debug=args.debug)
        else:
            print(f"[SKIP] No videos directory found")
            print("[INFO] Assuming frames are already extracted.")

    if args.step in ("all", "metadata"):
        print("\n[STEP 2] Building frame-level metadata...")
        frames_dir = frames_root / fps_tag
        if not frames_dir.exists():
            frames_dir = frames_root
        clip_infos_dir = output_dir / f"clip_infos_{fps_tag}"
        clip_infos_dir.mkdir(parents=True, exist_ok=True)

        df = build_frame_level_metadata(
            frames_root=frames_dir,
            clip_infos_dir=clip_infos_dir,
            debug=args.debug,
        )
        save_metadata_csvs(output_dir, df, fps_tag, seed=args.seed)

    if args.step in ("all", "clips"):
        print(f"\n[STEP 3] Generating dense clips (window_size={args.window_size})...")
        generate_dense_clips(
            base_data_path=str(output_dir),
            window_size=args.window_size,
            stride=args.stride,
            fps=args.fps,
            enable_padding=not args.no_padding,
        )

    print("\n" + "=" * 60)
    print("SurgicalActions160 preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
