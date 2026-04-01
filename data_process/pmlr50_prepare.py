#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PmLR50 Preprocessing Pipeline
------------------------------
Aligned with NSJepa_jinlin/data_process/pmlr50_csv.py.

Expected layout:
  - Labels (pickle):
      data/Landscopy/PmLR50/PmLR50/labels/train/1fpstrain.pickle
      data/Landscopy/PmLR50/PmLR50/labels/infer/1fpsinfer.pickle (test)
      data/Landscopy/PmLR50/PmLR50/labels/test/1fpstest.pickle   (val)
  - Frames (pre-extracted, 1 fps):
      data/Surge_Frames/PmLR50/frames/XX/XXXXXXXX.jpg (XX = case id)

NOTE: This pipeline assumes frames already exist. Use --step frames only if
you have the raw mp4 files.

Pipeline Steps:
  --step all (default): metadata + clips (does NOT extract frames)
  --step frames:        Extract frames from videos (requires --videos_dir with mp4s)
  --step metadata:      Build frame-level metadata CSV
  --step clips:         Generate dense sliding-window clips

Output structure:
  <output_dir>/
    clip_infos/                    # One txt per case
    train_metadata.csv             # Frame-level metadata
    val_metadata.csv
    test_metadata.csv
    missing_frames_report.csv
    clips_64f/                     # Dense clips
      train_dense_64f_detailed.csv
      ...

Usage:
    python pmlr50_prepare.py --step all
    python pmlr50_prepare.py --step metadata
    python pmlr50_prepare.py --step frames --videos_dir /path/to/videos
    python pmlr50_prepare.py --step clips --window_size 64
"""

from __future__ import annotations

import argparse
import pickle
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from gen_clips import generate_dense_clips

# Phase id -> human-readable name (official PmLR50 mapping)
PHASE_ID_TO_NAME: Dict[int, str] = {
    0: "Preparation stage",
    1: "Knotting of the Foley catheter",
    2: "Procedure of the liver resection",
    3: "Release of the Foley catheter",
    4: "Postprocessing stage",
}


def videos_to_frames(
    input_path: Path,
    output_path: Path,
    fps: int = 1,
    debug: bool = False,
) -> None:
    """
    Extract frames from all *.mp4 under input_path into output_path.
    Output: output_path/<video_id>/<video_id>_%08d.jpg
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
        pattern = out_folder / f"%08d.jpg"

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
    Write one line per frame path for a single case directory.
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


def _load_pickle_labels(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def build_frame_level_metadata(
    frames_root: Path,
    train_label: Path,
    val_label: Path,
    test_label: Path,
    output_dir: Path,
    debug: bool = False,
) -> Tuple[Dict[str, pd.DataFrame], List[dict]]:
    """
    Build frame-level metadata with columns: Case_ID, Frame_Path, Phase_GT, Phase_Name.
    Returns (dict of DataFrames keyed by split, missing_frames list).
    """
    clip_infos_dir = output_dir / "clip_infos"
    clip_infos_dir.mkdir(parents=True, exist_ok=True)

    by_split: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    missing_frames: List[dict] = []

    split_specs: List[Tuple[str, Path]] = [
        ("train", train_label),
        ("val", val_label),
        ("test", test_label),
    ]

    for split_name, label_path in split_specs:
        if not label_path.is_file():
            print(f"[WARN] Label file not found, skipping split '{split_name}': {label_path}")
            continue

        try:
            data = _load_pickle_labels(label_path)
        except Exception as exc:
            print(f"[WARN] Failed to load {label_path}: {exc}")
            continue

        if not isinstance(data, dict):
            print(f"[WARN] Expected dict in {label_path}, got {type(data)}")
            continue

        for vid, entries in tqdm(data.items(), desc=f"PmLR50 {split_name}"):
            try:
                case_id = int(vid)
            except (TypeError, ValueError):
                print(f"[WARN] Invalid video id {vid!r} in {label_path.name}, skip.")
                continue

            video_dir = frames_root / f"{case_id:02d}"
            if not video_dir.is_dir():
                print(f"[WARN] Missing frame directory: {video_dir}, skip video {vid!r}.")
                continue

            txt_path = clip_infos_dir / f"{case_id:02d}.txt"
            frame_paths = generate_clip_txt(video_dir, txt_path)
            if not frame_paths:
                print(f"[WARN] No frames listed under {video_dir}, skip case {case_id}.")
                continue

            frame_path_by_id = {}
            for fp in frame_paths:
                fname = Path(fp).stem
                try:
                    fid = int(fname)
                    frame_path_by_id[fid] = fp
                except ValueError:
                    pass

            if not isinstance(entries, list):
                if debug:
                    print(f"[DEBUG] Case {case_id}: entries not a list, skip.")
                continue

            for entry in entries:
                if not isinstance(entry, dict):
                    continue

                frame_id = int(entry["frame_id"])
                phase_gt = int(entry.get("phase_gt", -1))
                frame_path = frame_path_by_id.get(frame_id)

                if frame_path is None:
                    missing_frames.append({
                        "split": split_name,
                        "case_id": case_id,
                        "missing_frame": f"{frame_id:08d}.jpg",
                        "label_file": label_path.name,
                    })
                    continue

                label_name = PHASE_ID_TO_NAME.get(phase_gt, "Unknown")

                by_split[split_name].append({
                    "Case_ID": case_id,
                    "Frame_Path": frame_path,
                    "Phase_GT": phase_gt,
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

    return result, missing_frames


def save_metadata_csvs(output_dir: Path, metadata_by_split: Dict[str, pd.DataFrame], missing_frames: List[dict]) -> None:
    """Save frame-level metadata CSVs and missing frames report."""
    for split_name, df in metadata_by_split.items():
        path = output_dir / f"{split_name}_metadata.csv"
        if len(df) == 0:
            print(f"[WARN] No rows for split '{split_name}', skip writing {path.name}.")
            continue
        df.to_csv(path, index=False)
        print(f"[INFO] Saved {len(df)} frame rows to {path}")

    miss_path = output_dir / "missing_frames_report.csv"
    if missing_frames:
        pd.DataFrame(missing_frames).to_csv(miss_path, index=False)
        print(f"[WARN] Missing frames: {len(missing_frames)} entries -> {miss_path}")
    else:
        pd.DataFrame(columns=["split", "case_id", "missing_frame", "label_file"]).to_csv(miss_path, index=False)
        print(f"[INFO] No missing frames; wrote empty report: {miss_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PmLR50: End-to-end preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python pmlr50_prepare.py --step all
    python pmlr50_prepare.py --step metadata
    python pmlr50_prepare.py --step clips --window_size 64
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
        default="data/Surge_Frames/PmLR50/frames",
        help="Root with per-case folders XX/XXXXXXXX.jpg",
    )
    parser.add_argument(
        "--train_label",
        type=str,
        default="data/Landscopy/PmLR50/PmLR50/labels/train/1fpstrain.pickle",
        help="Train pickle",
    )
    parser.add_argument(
        "--test_label",
        type=str,
        default="data/Landscopy/PmLR50/PmLR50/labels/infer/1fpsinfer.pickle",
        help="Test pickle",
    )
    parser.add_argument(
        "--val_label",
        type=str,
        default="data/Landscopy/PmLR50/PmLR50/labels/test/1fpstest.pickle",
        help="Validation pickle",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/Surge_Frames/PmLR50",
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PmLR50 Preprocessing Pipeline")
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
        metadata_by_split, missing_frames = build_frame_level_metadata(
            frames_root=frames_root,
            train_label=Path(args.train_label),
            val_label=Path(args.val_label),
            test_label=Path(args.test_label),
            output_dir=output_dir,
            debug=args.debug,
        )
        save_metadata_csvs(output_dir, metadata_by_split, missing_frames)

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
    print("PmLR50 preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
