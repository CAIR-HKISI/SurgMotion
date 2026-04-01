#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M2CAI2016 Preprocessing Pipeline
---------------------------------
Aligned with NSJepa_jinlin/data_process/m2cai2016_csv.py.

Annotation layout:
  - Train: data/Landscopy/m2cai16/train_dataset/*.txt
  - Test:  data/Landscopy/m2cai16/test_dataset/*.txt
  - Annotations at 25 fps

Frames (1 fps, pre-extracted): data/Surge_Frames/M2CAI16/frames/{video_name}/{video_name}_XXXXXXXX.jpg

NOTE: This pipeline assumes frames already exist. The original M2CAI16 does not
distribute raw videos with the public dataset. Use --step frames only if you have
the raw mp4 files.

Pipeline Steps:
  --step all (default): metadata + clips (does NOT extract frames)
  --step frames:        Extract frames from videos (requires --videos_dir with mp4s)
  --step metadata:      Build frame-level metadata CSV
  --step clips:         Generate dense sliding-window clips

Output structure:
  <output_dir>/
    clip_infos/                    # One txt per video
    train_metadata.csv             # Frame-level metadata
    val_metadata.csv               # Copy of test
    test_metadata.csv
    clips_64f/                     # Dense clips
      train_dense_64f_detailed.csv
      ...

Usage:
    python m2cai2016_prepare.py --step all
    python m2cai2016_prepare.py --step metadata
    python m2cai2016_prepare.py --step frames --videos_dir /path/to/videos
    python m2cai2016_prepare.py --step clips --window_size 64
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

from gen_clips import generate_dense_clips

# Phase name -> integer label
PHASE_MAPPING: Dict[str, int] = {
    "TrocarPlacement": 0,
    "Preparation": 1,
    "CalotTriangleDissection": 2,
    "ClippingCutting": 3,
    "GallbladderDissection": 4,
    "GallbladderPackaging": 5,
    "CleaningCoagulation": 6,
    "GallbladderRetraction": 7,
}

ID_TO_PHASE: Dict[int, str] = {v: k for k, v in PHASE_MAPPING.items()}


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


def _list_annotation_files(annot_dir: Path) -> List[Path]:
    """Return sorted .txt annotation paths, skipping timestamp / pred sidecars."""
    if not annot_dir.is_dir():
        return []
    out: List[Path] = []
    for p in sorted(annot_dir.glob("*.txt")):
        name = p.name
        if name.endswith("_timestamp.txt") or name.endswith("_pred.txt"):
            continue
        out.append(p)
    return out


def collect_sorted_video_names(train_annot_dir: Path, test_annot_dir: Path) -> List[str]:
    """All unique annotation stems (video names), sorted."""
    names = set()
    for d in (train_annot_dir, test_annot_dir):
        for p in _list_annotation_files(d):
            names.add(p.stem)
    return sorted(names)


def build_case_id_map(sorted_video_names: Sequence[str]) -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(sorted_video_names)}


def read_annotation_txt(path: Path) -> List[Tuple[int, str]]:
    """Parse annotation: skip header, each line 'frame_index phase_name' at 25 fps."""
    rows: List[Tuple[int, str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        frame_idx = int(parts[0])
        phase = parts[1]
        rows.append((frame_idx, phase))
    return rows


def convert_25fps_to_1fps(annotations: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    """
    Collapse 25 fps labels to 1 fps: one label per second (last wins within each second).
    Returns sorted (frame_1fps_index, phase) with frame_1fps_index in {0, 1, ...}.
    """
    bucket: Dict[int, str] = {}
    for frame_25, phase in annotations:
        frame_1 = frame_25 // 25
        bucket[frame_1] = phase
    return sorted(bucket.items(), key=lambda x: x[0])


def generate_clip_txt(video_frames_dir: Path, txt_path: Path) -> List[str]:
    """
    Write one line per frame file under video_frames_dir (sorted by filename).
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


def build_frame_level_metadata(
    frames_root: Path,
    train_annot_dir: Path,
    test_annot_dir: Path,
    output_dir: Path,
    debug: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Build frame-level metadata with columns: Case_ID, Frame_Path, Phase_GT, Phase_Name.
    Returns dict of DataFrames keyed by split name.
    """
    sorted_names = collect_sorted_video_names(train_annot_dir, test_annot_dir)
    case_id_map = build_case_id_map(sorted_names)
    clip_infos_dir = output_dir / "clip_infos"
    output_dir.mkdir(parents=True, exist_ok=True)

    by_split: Dict[str, List[dict]] = {"train": [], "test": []}

    for split_name, annot_dir in [("train", train_annot_dir), ("test", test_annot_dir)]:
        annot_files = _list_annotation_files(annot_dir)
        for annot_path in tqdm(annot_files, desc=f"M2CAI16 {split_name}"):
            video_name = annot_path.stem
            case_id = case_id_map[video_name]
            video_frames_dir = frames_root / video_name
            clip_txt_path = clip_infos_dir / f"{video_name}.txt"

            if not video_frames_dir.is_dir():
                if debug:
                    print(f"[DEBUG] Missing frames dir: {video_frames_dir}")
                continue

            frame_paths = generate_clip_txt(video_frames_dir, clip_txt_path)
            if not frame_paths:
                if debug:
                    print(f"[DEBUG] No frames under {video_frames_dir}")
                continue

            frame_path_by_idx = {}
            for fp in frame_paths:
                fname = Path(fp).stem
                parts = fname.split("_")
                if len(parts) >= 2:
                    try:
                        idx = int(parts[-1])
                        frame_path_by_idx[idx] = fp
                    except ValueError:
                        pass

            annotations = read_annotation_txt(annot_path)
            converted = convert_25fps_to_1fps(annotations)

            for frame_1fps, phase_name in converted:
                disk_idx = frame_1fps + 1
                frame_path = frame_path_by_idx.get(disk_idx)
                if frame_path is None:
                    if debug:
                        print(f"[DEBUG] Missing frame {disk_idx} for {video_name}")
                    continue

                label = PHASE_MAPPING.get(phase_name, -1)
                by_split[split_name].append({
                    "Case_ID": case_id,
                    "Frame_Path": frame_path,
                    "Phase_GT": label,
                    "Phase_Name": phase_name,
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

    test_path = output_dir / "test_metadata.csv"
    val_path = output_dir / "val_metadata.csv"
    if test_path.exists() and not val_path.exists():
        shutil.copy(test_path, val_path)
        print(f"[INFO] Copied test_metadata.csv to val_metadata.csv for compatibility")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M2CAI2016: End-to-end preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python m2cai2016_prepare.py --step all
    python m2cai2016_prepare.py --step metadata
    python m2cai2016_prepare.py --step clips --window_size 64
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
        default="data/Surge_Frames/M2CAI16/frames",
        help="Root folder containing per-video frame directories",
    )
    parser.add_argument(
        "--train_annot_dir",
        type=str,
        default="data/Landscopy/m2cai16/train_dataset",
        help="Directory with training phase annotation .txt files",
    )
    parser.add_argument(
        "--test_annot_dir",
        type=str,
        default="data/Landscopy/m2cai16/test_dataset",
        help="Directory with test phase annotation .txt files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/Surge_Frames/M2CAI16",
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
    train_annot_dir = Path(args.train_annot_dir)
    test_annot_dir = Path(args.test_annot_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("M2CAI2016 Preprocessing Pipeline")
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
            train_annot_dir=train_annot_dir,
            test_annot_dir=test_annot_dir,
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
    print("M2CAI2016 preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
