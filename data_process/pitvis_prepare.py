#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PitVis Preprocessing Pipeline
------------------------------
Aligned with NSJepa_jinlin/data_process/pitvis_csv.py.

Expected layout:
  - Annotations: data/NeuroSurgery/pitvits/26531686/annotations_XX.csv
  - Frames (pre-extracted, 1 fps): data/Surge_Frames/PitVis/frames/video_XX/video_XX_XXXXXXXX.jpg
    Frame index (1-based): int_time + 1

NOTE: This pipeline assumes frames already exist. Use --step frames only if you
have raw mp4 files, which are not distributed with the standard PitVis dataset.

Pipeline Steps:
  --step all (default): metadata + clips (does NOT extract frames)
  --step frames:        Extract frames from videos (requires --videos_dir with mp4s)
  --step metadata:      Build frame-level metadata CSV
  --step clips:         Generate dense sliding-window clips

Output structure:
  <output_dir>/
    clip_infos/                    # One txt per video
    train_metadata.csv             # Frame-level metadata
    val_metadata.csv
    test_metadata.csv
    clips_64f/                     # Dense clips
      train_dense_64f_detailed.csv
      ...

Usage:
    python pitvis_prepare.py --step all
    python pitvis_prepare.py --step metadata
    python pitvis_prepare.py --step frames --videos_dir /path/to/videos
    python pitvis_prepare.py --step clips --window_size 64
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
from tqdm import tqdm

from gen_clips import generate_dense_clips

# Original int_step -> human-readable phase name
PHASE_MAPPING: Dict[int, str] = {
    -1: "operation_ended",
    1: "nasal corridor creation",
    2: "anterior sphenoidotomy",
    3: "septum displacement",
    4: "sphenoid sinus clearance",
    5: "sellotomy",
    6: "durotomy",
    7: "tumour excision",
    8: "haemostasis",
    9: "synthetic_graft_placement",
    10: "fat graft placement",
    11: "gasket seal construct",
    12: "dural sealant",
    13: "nasal packing",
    14: "debris clearance",
}

# Phases excluded from training metadata
FILTERED_PHASES: Set[int] = {-1, 11, 13}

# Sorted valid originals [1,2,...,10,12,14] -> 0..11
VALID_PHASES: List[int] = sorted(p for p in PHASE_MAPPING if p not in FILTERED_PHASES)
PHASE_REMAP: Dict[int, int] = {p: i for i, p in enumerate(VALID_PHASES)}

TRAIN_VIDEOS: List[str] = [
    "01", "03", "04", "05", "07", "08", "09", "10", "11", "14",
    "15", "16", "17", "18", "19", "20", "21", "22", "23", "25",
]
VAL_VIDEOS: List[str] = ["02", "06", "12", "13", "24"]
TEST_VIDEOS: List[str] = ["02", "06", "12", "13", "24"]

EXPECTED_ANNOT_COLS = ["int_video", "int_time", "int_step", "int_instrument1", "int_instrument2"]


def splits_for_video(video_id: str) -> List[str]:
    """Return which splits contain this video id."""
    out: List[str] = []
    if video_id in TRAIN_VIDEOS:
        out.append("train")
    if video_id in VAL_VIDEOS:
        out.append("val")
    if video_id in TEST_VIDEOS:
        out.append("test")
    return out


def videos_to_frames(
    input_path: Path,
    output_path: Path,
    fps: int = 1,
    debug: bool = False,
) -> None:
    """
    Extract frames from all *.mp4 under input_path into output_path.
    Output: output_path/video_<stem>/video_<stem>_%08d.jpg
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
        out_folder = output_path / f"video_{vid_id}"
        out_folder.mkdir(parents=True, exist_ok=True)
        pattern = out_folder / f"video_{vid_id}_%08d.jpg"

        cmd = [
            "ffmpeg", "-y", "-i", str(vid_path.resolve()),
            "-vf", f"fps={fps},scale=512:-1:flags=bicubic",
            "-vsync", "2", "-qscale:v", "2", "-start_number", "1",
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
    Write one line per frame path for a single video folder.
    Returns list of frame paths.
    """
    frame_files = sorted(
        (p for p in video_frames_dir.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")),
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

    all_videos = sorted(set(TRAIN_VIDEOS) | set(VAL_VIDEOS) | set(TEST_VIDEOS))

    for video_id in tqdm(all_videos, desc="Building frame-level metadata"):
        splits = splits_for_video(video_id)
        if not splits:
            continue

        frames_dir = frames_root / f"video_{video_id}"
        if not frames_dir.is_dir():
            print(f"[WARN] Missing frames directory: {frames_dir}, skip video {video_id}.")
            continue

        txt_path = clip_infos_dir / f"video_{video_id}.txt"
        frame_paths = generate_clip_txt(frames_dir, txt_path)
        if not frame_paths:
            print(f"[WARN] No frames under {frames_dir}, skip video {video_id}.")
            continue

        frame_path_by_idx = {}
        for fp in frame_paths:
            fname = Path(fp).stem
            parts = fname.split("_")
            if len(parts) >= 3:
                try:
                    idx = int(parts[-1])
                    frame_path_by_idx[idx] = fp
                except ValueError:
                    pass

        annot_path = annot_dir / f"annotations_{video_id}.csv"
        if not annot_path.is_file():
            print(f"[WARN] Missing annotation file: {annot_path}, skip video {video_id}.")
            continue

        try:
            df = pd.read_csv(annot_path)
        except Exception as e:
            print(f"[WARN] Failed to read {annot_path}: {e}")
            continue

        missing = [c for c in EXPECTED_ANNOT_COLS if c not in df.columns]
        if missing:
            print(f"[WARN] {annot_path.name} missing columns {missing}, skip.")
            continue

        for _, row in df.iterrows():
            int_step = int(row["int_step"])
            if int_step in FILTERED_PHASES:
                continue
            if int_step not in PHASE_REMAP:
                if debug:
                    print(f"[DEBUG] Video {video_id}: unknown int_step={int_step}, skip row.")
                continue

            int_time = int(row["int_time"])
            frame_index = int_time + 1
            frame_path = frame_path_by_idx.get(frame_index)
            if frame_path is None:
                if debug:
                    print(f"[DEBUG] Video {video_id}: frame {frame_index} missing")
                continue

            label = PHASE_REMAP[int_step]
            label_name = PHASE_MAPPING.get(int_step, f"unknown_{int_step}")

            for s in splits:
                by_split[s].append({
                    "Case_ID": int(video_id),
                    "Frame_Path": frame_path,
                    "Phase_GT": label,
                    "Phase_Name": label_name,
                })

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
        description="PitVis: End-to-end preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python pitvis_prepare.py --step all
    python pitvis_prepare.py --step metadata
    python pitvis_prepare.py --step clips --window_size 64
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
        default="data/Surge_Frames/PitVis/frames",
        help="Root containing video_XX/ folders with jpg frames",
    )
    parser.add_argument(
        "--annot_dir",
        type=str,
        default="data/NeuroSurgery/pitvits/26531686",
        help="Directory with annotations_XX.csv per video",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/Surge_Frames/PitVis",
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

    frames_root = Path(args.frames_root)
    annot_dir = Path(args.annot_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PitVis Preprocessing Pipeline")
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
    print("PitVis preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
