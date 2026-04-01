#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PolypDiag End-to-End Preprocessing Pipeline
--------------------------------------------
Data organization:
  - Raw videos: data/GI_Videos/PolypDiag/videos/*.mp4
  - Splits: data/GI_Videos/PolypDiag/splits/train.txt, val.txt
    Each line format: <video_filename>.mp4,<label>
    label is 0/1 (normal=0, abnormal=1)

Pipeline Steps:
  1. [Optional] Rename videos to consecutive numbers (for cleaner organization)
  2. [Optional] Extract frames from videos (--step frames or --step all)
  3. Build frame-level metadata CSV with columns:
     Case_ID, Frame_Path, Phase_GT, Phase_Name
  4. Generate dense sliding-window clips (--step clips or --step all)

Output structure:
  <output_dir>/
    clip_infos/                    # One txt per video
    train_metadata.csv             # Frame-level metadata
    val_metadata.csv
    test_metadata.csv              # Copy of val for compatibility
    clips_64f/                     # Dense clips
      train_dense_64f_detailed.csv
      ...

Usage:
    python polypdiag_prepare.py --step all
    python polypdiag_prepare.py --step metadata
    python polypdiag_prepare.py --step clips --window_size 64
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from gen_clips import generate_dense_clips

# Label mapping for binary classification
LABEL_TO_NAME: Dict[int, str] = {0: "normal", 1: "abnormal"}


def renumber_videos(
    src_videos_dir: Path,
    dst_videos_dir: Path,
) -> Dict[str, str]:
    """
    Copy and rename all mp4 in src_videos_dir to dst_videos_dir.
    Returns dictionary: {original file name: new file name}.
    """
    dst_videos_dir.mkdir(parents=True, exist_ok=True)
    video_files = sorted(src_videos_dir.glob("*.mp4"))

    mapping: Dict[str, str] = {}
    print(f"[INFO] Found {len(video_files)} videos, renaming and copying...")

    for idx, vid in enumerate(tqdm(video_files, desc="Copy & rename"), start=1):
        new_name = f"{idx:05d}.mp4"
        new_path = dst_videos_dir / new_name
        shutil.copy2(vid, new_path)
        mapping[vid.name] = new_name

    print(f"[INFO] Renaming completed, output directory: {dst_videos_dir}")
    return mapping


def save_mapping_csv(mapping: Dict[str, str], out_path: Path):
    """Save the mapping to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["orig_name", "new_name"])
        for orig, new in sorted(mapping.items()):
            writer.writerow([orig, new])
    print(f"[INFO] Saved renaming mapping to: {out_path}")


def load_or_build_name_mapping(src_videos_dir: Path, dst_videos_dir: Path) -> Dict[str, str]:
    """
    Load or build the mapping of orig_name -> new_name.
    If polypdiag_name_mapping.csv exists, load from it;
    Otherwise, use identity mapping (orig_name -> orig_name).
    """
    mapping_csv_path = dst_videos_dir.parent / "polypdiag_name_mapping.csv"
    mapping: Dict[str, str] = {}

    if mapping_csv_path.exists():
        print(f"[INFO] Loading mapping from existing file: {mapping_csv_path}")
        with mapping_csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                orig = row.get("orig_name")
                new = row.get("new_name")
                if orig and new:
                    mapping[orig] = new
        print(f"[INFO] Loaded {len(mapping)} mappings")
    else:
        print(f"[INFO] No mapping file found, using identity mapping (orig_name -> orig_name)")
        for vid in src_videos_dir.glob("*.mp4"):
            mapping[vid.name] = vid.name
        print(f"[INFO] Built {len(mapping)} identity mappings")

    return mapping


def videos_to_frames(
    input_path: Path,
    output_path: Path,
    fps: int = 1,
    debug: bool = False,
) -> None:
    """
    Extract frames from all *.mp4 under input_path into output_path.
    Output: output_path/<video_stem>/<video_stem>_%08d.jpg
    """
    output_path.mkdir(parents=True, exist_ok=True)
    video_files = sorted(input_path.glob("*.mp4"))

    if not video_files:
        print(f"[WARN] No mp4 videos found in {input_path}.")
        return

    print(f"\n[INFO] Found {len(video_files)} videos, extracting frames at {fps} fps...")
    failed_videos: List[str] = []

    for vid_path in tqdm(video_files, desc="Extracting frames"):
        vid_id = vid_path.stem
        out_folder = output_path / vid_id
        out_folder.mkdir(parents=True, exist_ok=True)
        output_pattern = out_folder / f"{vid_id}_%08d.jpg"

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

    print(f"[INFO] Frame extraction finished.")
    if failed_videos:
        fail_log = output_path / "failed_videos.txt"
        fail_log.write_text("\n".join(failed_videos), encoding="utf-8")
        print(f"[WARN] {len(failed_videos)} videos failed; see {fail_log}")


def generate_clip_txt(video_frames_dir: Path, txt_path: Path) -> List[str]:
    """
    Generate txt listing all frame paths for a video.
    Returns list of frame paths.
    """
    frame_files = sorted(
        [p for p in video_frames_dir.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")],
        key=lambda p: p.name,
    )
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    frame_paths = [str(fp).replace("\\", "/") for fp in frame_files]
    with txt_path.open("w", encoding="utf-8") as f:
        for fp in frame_paths:
            f.write(fp + "\n")
    return frame_paths


def load_split_file(split_path: Path) -> Dict[str, int]:
    """Load train.txt / val.txt, return {video_filename: label_int}."""
    mapping: Dict[str, int] = {}
    if not split_path.exists():
        print(f"[WARN] Split file not found: {split_path}")
        return mapping

    with split_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                name, label_str = parts[0], parts[1]
                mapping[name] = int(label_str)
    return mapping


def build_frame_level_metadata(
    frames_root: Path,
    splits_dir: Path,
    output_dir: Path,
    src_videos_dir: Path,
    dst_videos_dir: Path,
    debug: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Build frame-level metadata with columns: Case_ID, Frame_Path, Phase_GT, Phase_Name.
    For PolypDiag, all frames in a video have the same label (video-level label).
    """
    name_mapping = load_or_build_name_mapping(src_videos_dir, dst_videos_dir)
    train_split = load_split_file(splits_dir / "train.txt")
    val_split = load_split_file(splits_dir / "val.txt")

    clip_infos_dir = output_dir / "clip_infos"
    clip_infos_dir.mkdir(parents=True, exist_ok=True)

    by_split: Dict[str, List[dict]] = {"train": [], "val": []}
    case_id_counter = 0

    for split_name, split_dict in [("train", train_split), ("val", val_split)]:
        missing = []
        for orig_name, label in tqdm(split_dict.items(), desc=f"Processing {split_name}"):
            if orig_name not in name_mapping:
                missing.append(orig_name)
                continue

            new_name = name_mapping[orig_name]
            vid_id = Path(new_name).stem
            video_frames_dir = frames_root / vid_id

            if not video_frames_dir.exists():
                missing.append(orig_name)
                continue

            txt_path = clip_infos_dir / f"{vid_id}.txt"
            frame_paths = generate_clip_txt(video_frames_dir, txt_path)

            if not frame_paths:
                if debug:
                    print(f"[DEBUG] No frames for {vid_id}")
                continue

            label_name = LABEL_TO_NAME.get(label, str(label))

            for fp in frame_paths:
                by_split[split_name].append({
                    "Case_ID": case_id_counter,
                    "Frame_Path": fp,
                    "Phase_GT": label,
                    "Phase_Name": label_name,
                })

            case_id_counter += 1

        if missing:
            print(f"[WARN] In {split_name} split, {len(missing)} files not found. Example: {missing[:3]}")

    result = {}
    for split_name, rows in by_split.items():
        if rows:
            df = pd.DataFrame(rows)
            df = df.sort_values(["Case_ID", "Frame_Path"]).reset_index(drop=True)
            result[split_name] = df
        else:
            result[split_name] = pd.DataFrame(
                columns=["Case_ID", "Frame_Path", "Phase_GT", "Phase_Name"]
            )

    return result


def save_metadata_csvs(output_dir: Path, metadata_by_split: Dict[str, pd.DataFrame]) -> None:
    """Save frame-level metadata CSVs."""
    for split_name, df in metadata_by_split.items():
        path = output_dir / f"{split_name}_metadata.csv"
        if len(df) == 0:
            print(f"[WARN] No rows for split '{split_name}', skip writing {path.name}.")
            continue
        df.to_csv(path, index=False)
        print(f"[INFO] Saved {len(df)} frame rows to {path}")

    val_path = output_dir / "val_metadata.csv"
    test_path = output_dir / "test_metadata.csv"
    if val_path.exists() and not test_path.exists():
        shutil.copy(val_path, test_path)
        print(f"[INFO] Copied val_metadata.csv to test_metadata.csv for compatibility")


def main():
    parser = argparse.ArgumentParser(
        description="PolypDiag: End-to-end preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python polypdiag_prepare.py --step all
    python polypdiag_prepare.py --step metadata
    python polypdiag_prepare.py --step clips --window_size 64
        """,
    )
    parser.add_argument(
        "--step",
        choices=["all", "rename", "frames", "metadata", "clips"],
        default="all",
        help="Pipeline step to run (default: all)",
    )
    parser.add_argument(
        "--src_videos",
        type=str,
        default="data/GI_Videos/PolypDiag/videos",
        help="Original video directory",
    )
    parser.add_argument(
        "--dst_videos",
        type=str,
        default="data/GI_Videos/PolypDiag/videos_renumbered",
        help="Renamed video directory",
    )
    parser.add_argument(
        "--frames_root",
        type=str,
        default="data/Surge_Frames/PolypDiag/frames",
        help="Frame extraction output directory",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="data/GI_Videos/PolypDiag/splits",
        help="train/val split directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/Surge_Frames/PolypDiag",
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
        "--debug",
        action="store_true",
        help="Enable verbose debug output",
    )
    args = parser.parse_args()

    src_videos_dir = Path(args.src_videos)
    dst_videos_dir = Path(args.dst_videos)
    frames_root = Path(args.frames_root)
    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PolypDiag Preprocessing Pipeline")
    print("=" * 60)

    if args.step in ("all", "rename"):
        print("\n[STEP 0] Renaming videos...")
        if src_videos_dir.exists():
            mapping = renumber_videos(src_videos_dir, dst_videos_dir)
            mapping_csv = dst_videos_dir.parent / "polypdiag_name_mapping.csv"
            save_mapping_csv(mapping, mapping_csv)
        else:
            print(f"[SKIP] Source videos directory not found: {src_videos_dir}")

    if args.step in ("all", "frames"):
        print("\n[STEP 1] Extracting frames from videos...")
        if dst_videos_dir.exists():
            videos_to_frames(dst_videos_dir, frames_root, fps=args.fps, debug=args.debug)
        elif src_videos_dir.exists():
            videos_to_frames(src_videos_dir, frames_root, fps=args.fps, debug=args.debug)
        else:
            print(f"[SKIP] No videos directory found")
            print("[INFO] Assuming frames are already extracted.")

    if args.step in ("all", "metadata"):
        print("\n[STEP 2] Building frame-level metadata...")
        metadata_by_split = build_frame_level_metadata(
            frames_root=frames_root,
            splits_dir=splits_dir,
            output_dir=output_dir,
            src_videos_dir=src_videos_dir,
            dst_videos_dir=dst_videos_dir,
            debug=args.debug,
        )
        save_metadata_csvs(output_dir, metadata_by_split)

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
    print("PolypDiag preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
