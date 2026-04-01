#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EgoSurgery Preprocessing Pipeline
-----------------------------------
Aligned with NSJepa_jinlin/data_process/egosurgery_csv.py.

Data organization:
  - Phase annotations:
    data/Open_surgery/EgoSurgery/annotations/phase/XX_Y.csv
    XX = video id (zero-padded), Y = view id.
    Columns: Frame, Phase (phase name string).
  - Frames (pre-extracted):
    data/Surge_Frames/EgoSurgery/frames/XX/<Frame>.jpg
    OR you can point --frames_root to data/Open_surgery/EgoSurgery/images
    if the directory layout matches {video_id}/{Frame}.jpg.

Official splits (by integer video id):
  - train: [1, 2, 3, 4, 8, 9, 11, 13, 14, 15, 17, 20, 21]
  - val:   [5, 19]
  - test:  [6, 7, 10, 12, 18]

NOTE: This pipeline assumes frames already exist. Use --step frames only if
you have raw mp4 files.

Pipeline Steps:
  --step all (default): metadata + clips (does NOT extract frames)
  --step frames:        Extract frames from videos (requires --videos_dir with mp4s)
  --step metadata:      Build frame-level metadata CSV
  --step clips:         Generate dense sliding-window clips

Output structure:
  <output_dir>/
    clip_infos/                    # One txt per (video, view)
    train_metadata.csv             # Frame-level metadata
    val_metadata.csv
    test_metadata.csv
    missing_frames_report.csv
    clips_64f/                     # Dense clips
      train_dense_64f_detailed.csv
      ...

Usage:
    python egosurgery_prepare.py --step all
    python egosurgery_prepare.py --step metadata
    python egosurgery_prepare.py --step frames --videos_dir /path/to/videos
    python egosurgery_prepare.py --step clips --window_size 64
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from gen_clips import generate_dense_clips

# Integer video id -> split name
TRAIN_VIDEO_IDS = [1, 2, 3, 4, 8, 9, 11, 13, 14, 15, 17, 20, 21]
VAL_VIDEO_IDS = [5, 19]
TEST_VIDEO_IDS = [6, 7, 10, 12, 18]

# Phase name (lowercase) -> class id
PHASE_TO_ID: Dict[str, int] = {
    "disinfection": 0,
    "design": 1,
    "anesthesia": 2,
    "incision": 3,
    "dissection": 4,
    "hemostasis": 5,
    "irrigation": 6,
    "closure": 7,
    "dressing": 8,
}

ID_TO_PHASE: Dict[int, str] = {v: k for k, v in PHASE_TO_ID.items()}


def split_for_video_id(video_id: int) -> Optional[str]:
    """Return split name for a given video id."""
    if video_id in TRAIN_VIDEO_IDS:
        return "train"
    if video_id in VAL_VIDEO_IDS:
        return "val"
    if video_id in TEST_VIDEO_IDS:
        return "test"
    return None


def videos_to_frames(
    input_path: Path,
    output_path: Path,
    fps: int = 30,
    debug: bool = False,
) -> None:
    """
    Extract frames from all *.mp4 under input_path into output_path.
    Output: output_path/<video_stem>/<video_stem>_%08d.jpg
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    video_files = sorted(input_path.glob("*.mp4"))

    if not video_files:
        print(f"[WARN] No mp4 videos found under {input_path}.")
        return

    print(f"\n[INFO] Found {len(video_files)} videos, extracting frames at {fps} fps...")
    failed: List[str] = []

    for vid_path in tqdm(video_files, desc="Extracting frames"):
        vid_id = vid_path.stem
        out_folder = output_path / vid_id
        out_folder.mkdir(parents=True, exist_ok=True)
        pattern = out_folder / f"{vid_id}_%08d.jpg"

        cmd = [
            "ffmpeg", "-y", "-i", str(vid_path.resolve()),
            "-vf", f"fps={fps},scale=512:-1:flags=bicubic",
            "-vsync", "2", "-qscale:v", "2",
            str(pattern),
        ]

        if debug:
            print(f"[DEBUG] FFmpeg: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed: {vid_path}")
            if debug and e.stderr:
                print(e.stderr.decode("utf-8", errors="ignore")[:500])
            failed.append(str(vid_path))

    print(f"[INFO] Frame extraction finished.")
    if failed:
        log = output_path / "failed_videos.txt"
        log.write_text("\n".join(failed), encoding="utf-8")
        print(f"[WARN] {len(failed)} videos failed; see {log}")


def generate_clip_txt(video_frames_dir: Path, txt_path: Path) -> List[str]:
    """
    Write one line per frame file under video_frames_dir (sorted by filename).
    Returns list of frame paths.
    """
    if not video_frames_dir.is_dir():
        return []

    frame_files = sorted(
        [p for p in video_frames_dir.iterdir() if p.is_file()],
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
    annot_dir: Path,
    output_dir: Path,
    debug: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Build frame-level metadata with columns: Case_ID, Frame_Path, Phase_GT, Phase_Name.
    Returns dict of DataFrames keyed by split name.
    """
    clip_infos_dir = output_dir / "clip_infos"
    clip_infos_dir.mkdir(parents=True, exist_ok=True)

    by_split: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    missing_report: List[dict] = []

    annot_files = sorted(annot_dir.glob("*.csv"))
    if not annot_files:
        print(f"[WARN] No CSV annotations found under {annot_dir}")

    for annot_path in tqdm(annot_files, desc="Building frame-level metadata"):
        stem = annot_path.stem
        parts = stem.split("_")
        if len(parts) != 2:
            if debug:
                print(f"[DEBUG] Skipping unexpected annotation name: {annot_path.name}")
            continue

        video_id_str, view_id_str = parts[0], parts[1]
        try:
            case_id = int(video_id_str)
            view_id = int(view_id_str)
        except ValueError:
            if debug:
                print(f"[DEBUG] Skipping non-integer id in name: {annot_path.name}")
            continue

        split = split_for_video_id(case_id)
        if split is None:
            if debug:
                print(f"[DEBUG] Skipping {annot_path.name}: video id {case_id} not in splits")
            continue

        try:
            df_phase = pd.read_csv(annot_path)
        except Exception as exc:
            print(f"[WARN] Failed to read {annot_path}: {exc}")
            continue

        if "Frame" not in df_phase.columns or "Phase" not in df_phase.columns:
            print(f"[WARN] Missing Frame/Phase columns in {annot_path.name}")
            continue

        video_frames_dir = frames_root / video_id_str
        clip_txt_path = clip_infos_dir / f"{stem}.txt"
        frame_paths = generate_clip_txt(video_frames_dir, clip_txt_path)

        if not frame_paths and debug:
            print(f"[DEBUG] No frames for {video_frames_dir}")

        frame_path_map = {}
        for fp in frame_paths:
            fname = Path(fp).stem
            frame_path_map[fname] = fp

        for _, row in df_phase.iterrows():
            frame_name = str(row["Frame"])
            frame_path = frame_path_map.get(frame_name)

            if frame_path is None:
                missing_report.append({
                    "video_id": video_id_str,
                    "view_id": view_id_str,
                    "missing_frame": frame_name,
                    "annotation_file": annot_path.name,
                })
                continue

            phase_raw = str(row["Phase"]).strip()
            phase_name = phase_raw.lower()
            label = PHASE_TO_ID.get(phase_name, -1)

            by_split[split].append({
                "Case_ID": case_id,
                "Frame_Path": frame_path,
                "Phase_GT": label,
                "Phase_Name": phase_name if phase_name else phase_raw,
            })

    if missing_report:
        miss_path = output_dir / "missing_frames_report.csv"
        pd.DataFrame(missing_report).to_csv(miss_path, index=False)
        print(f"[WARN] Missing frames: {len(missing_report)} entries (see {miss_path})")

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EgoSurgery: End-to-end preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python egosurgery_prepare.py --step all
    python egosurgery_prepare.py --step metadata
    python egosurgery_prepare.py --step clips --window_size 64
        """,
    )
    parser.add_argument(
        "--step",
        choices=["all", "frames", "metadata", "clips"],
        default="all",
        help="Pipeline step to run (default: all)",
    )
    parser.add_argument(
        "--videos_dir",
        type=str,
        default="",
        help="Directory containing raw mp4 videos (only for --step frames; not part of standard workflow)",
    )
    parser.add_argument(
        "--frames_root",
        type=str,
        default="data/Surge_Frames/EgoSurgery/frames",
        help="Root directory containing per-video frame folders",
    )
    parser.add_argument(
        "--annot_dir",
        type=str,
        default="data/Open_surgery/EgoSurgery/annotations/phase",
        help="Directory with XX_Y.csv annotation files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/Surge_Frames/EgoSurgery",
        help="Output directory for metadata and clips",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS for frame extraction (default: 30)",
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

    frames_root = Path(args.frames_root)
    annot_dir = Path(args.annot_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EgoSurgery Preprocessing Pipeline")
    print("=" * 60)

    # Frame extraction is NOT part of --step all; run only with --step frames
    if args.step == "frames":
        print("\n[STEP] Extracting frames from videos...")
        videos_dir = Path(args.videos_dir)
        if videos_dir.exists() and any(videos_dir.glob("*.mp4")):
            videos_to_frames(videos_dir, frames_root, fps=args.fps, debug=args.debug)
        else:
            print(f"[ERROR] No mp4 videos found in: {videos_dir}")
            print("[INFO] Provide --videos_dir pointing to a directory with mp4 files.")
            return

    if args.step in ("all", "metadata"):
        print("\n[STEP 2] Building frame-level metadata...")
        metadata_by_split = build_frame_level_metadata(
            frames_root=frames_root,
            annot_dir=annot_dir,
            output_dir=output_dir,
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
    print("EgoSurgery preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
