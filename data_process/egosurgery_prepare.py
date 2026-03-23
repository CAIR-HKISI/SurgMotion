#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EgoSurgery preprocessing for phase recognition (end-to-end pipeline style).

Data organization
-----------------
- Phase annotations (CSV per video and camera view):
    data/Open_surgery/EgoSurgery/annotations/phase/XX_Y.csv
  XX = video id (zero-padded in filenames, e.g. 01), Y = view id.
  Columns: Frame, Phase (phase name string).

- Extracted frames (shared per video id, not per view):
    data/Surge_Frames/EgoSurgery/frames/XX/<frame>.jpg

Official-style splits (by integer video id parsed from XX):
- train: [1, 2, 3, 4, 8, 9, 11, 13, 14, 15, 17, 20, 21]
- val:   [5, 19]
- test:  [6, 7, 10, 12, 18]

Pipeline steps
--------------
1. (Optional stub) videos_to_frames: extract frames from raw videos into --frames_root.
2. generate_clip_txt: for each annotation case (XX_Y), write a text file listing all frame
   paths under frames/XX/ (sorted), for downstream clip-style loaders.
3. build_metadata: read each annotation CSV, align rows with frame files, emit one metadata
   row per annotated frame (per-frame phase labels).

Outputs (under --output_dir)
----------------------------
- clip_infos/XX_Y.txt — all frame paths for video XX (same file duplicated per view Y if
  multiple views exist; paths are the full under frames_root).
- train_metadata.csv, val_metadata.csv, test_metadata.csv with columns:
    Index, clip_path, label, label_name, case_id, clip_idx
  - label: integer phase id (phase_gt); unknown names map to -1.
  - label_name: lowercase phase name from CSV (or raw string if unmapped).
  - case_id: integer video id (from XX).
  - clip_idx: integer view id (Y).
- missing_frames_report.csv: rows for frames referenced in annotations but missing on disk.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

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


def split_for_video_id(video_id: int) -> Optional[str]:
    """Return 'train', 'val', 'test', or None if the video is not in any split."""
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
    Stub: extract frames from raw EgoSurgery videos into output_path.

    Expected layout and ffmpeg settings depend on the raw release; implement this when
    video roots and naming are fixed. Frames should end up under output_path/XX/...jpg
    to match annotation paths.
    """
    print(
        "videos_to_frames: stub — no extraction run. "
        "Place frames under --frames_root (…/frames/XX/*.jpg) or implement extraction here."
    )
    if debug:
        print(f"  input_path={input_path}, output_path={output_path}, fps={fps}")


def generate_clip_txt(video_frames_dir: Path, txt_path: Path) -> int:
    """
    Write one line per frame file under video_frames_dir (sorted by filename).
    Paths use forward slashes, relative to the current working directory (typically project root).
    Returns the number of lines written.
    """
    if not video_frames_dir.is_dir():
        return 0

    frame_files = sorted(
        [p for p in video_frames_dir.iterdir() if p.is_file()],
        key=lambda p: p.name,
    )
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w", encoding="utf-8") as f:
        for frame_path in frame_files:
            f.write(str(frame_path).replace("\\", "/") + "\n")
    return len(frame_files)


def save_metadata_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Saved metadata ({len(rows)} rows) to: {path}")


def build_metadata(
    frames_root: Path,
    annot_dir: Path,
    output_dir: Path,
    debug: bool = False,
) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    """
    Read phase CSVs, generate clip txts per case (XX_Y), and build per-split metadata lists.

    Each annotated frame becomes one row: label is phase id, case_id is video id, clip_idx is view id.
    clip_path points to the txt for that (video, view) case (listing all frames in that video folder).
    """
    clip_infos_dir = output_dir / "clip_infos"
    clip_infos_dir.mkdir(parents=True, exist_ok=True)

    train_rows: List[dict] = []
    val_rows: List[dict] = []
    test_rows: List[dict] = []
    missing_report: List[dict] = []

    global_index = 0
    annot_files = sorted(annot_dir.glob("*.csv"))

    if not annot_files:
        print(f"No CSV annotations found under {annot_dir}")

    for annot_path in tqdm(annot_files, desc="EgoSurgery annotations"):
        stem = annot_path.stem
        parts = stem.split("_")
        if len(parts) != 2:
            if debug:
                print(f"Skipping unexpected annotation name: {annot_path.name}")
            continue
        video_id_str, view_id_str = parts[0], parts[1]
        try:
            case_id = int(video_id_str)
            view_id = int(view_id_str)
        except ValueError:
            if debug:
                print(f"Skipping non-integer id in name: {annot_path.name}")
            continue

        split = split_for_video_id(case_id)
        if split is None:
            if debug:
                print(f"Skipping {annot_path.name}: video id {case_id} not in train/val/test lists")
            continue

        try:
            df_phase = pd.read_csv(annot_path)
        except Exception as exc:
            print(f"Failed to read {annot_path}: {exc}")
            continue

        if "Frame" not in df_phase.columns or "Phase" not in df_phase.columns:
            print(f"Missing Frame/Phase columns in {annot_path.name}")
            continue

        video_frames_dir = frames_root / video_id_str
        clip_txt_path = clip_infos_dir / f"{stem}.txt"
        n_clip_lines = generate_clip_txt(video_frames_dir, clip_txt_path)
        if n_clip_lines == 0 and debug:
            print(f"No frames listed for {video_frames_dir} (case {stem})")

        clip_path_str = str(clip_txt_path).replace("\\", "/")

        for _, row in df_phase.iterrows():
            frame_name = f"{row['Frame']}.jpg"
            frame_path = video_frames_dir / frame_name

            if not frame_path.is_file():
                missing_report.append(
                    {
                        "video_id": video_id_str,
                        "view_id": view_id_str,
                        "missing_frame": frame_name,
                        "annotation_file": annot_path.name,
                    }
                )
                continue

            phase_raw = str(row["Phase"]).strip()
            phase_name = phase_raw.lower()
            label = PHASE_TO_ID.get(phase_name, -1)
            label_name = phase_name if phase_name else phase_raw

            item = {
                "Index": global_index,
                "clip_path": clip_path_str,
                "label": int(label),
                "label_name": label_name,
                "case_id": case_id,
                "clip_idx": view_id,
            }
            global_index += 1

            if split == "train":
                train_rows.append(item)
            elif split == "val":
                val_rows.append(item)
            else:
                test_rows.append(item)

    return train_rows, val_rows, test_rows, missing_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EgoSurgery: clip txts + per-frame phase metadata (train/val/test)."
    )
    parser.add_argument(
        "--frames_root",
        type=str,
        default="data/Surge_Frames/EgoSurgery/frames",
        help="Root directory containing per-video frame folders (…/XX/*.jpg).",
    )
    parser.add_argument(
        "--annot_dir",
        type=str,
        default="data/Open_surgery/EgoSurgery/annotations/phase",
        help="Directory with XX_Y.csv files (Frame, Phase).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/Surge_Frames/EgoSurgery",
        help="Output directory for clip_infos/, *_metadata.csv, missing_frames_report.csv.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extra diagnostics for skipped or empty cases.",
    )
    args = parser.parse_args()

    frames_root = Path(args.frames_root)
    annot_dir = Path(args.annot_dir)
    output_dir = Path(args.output_dir)

    train_rows, val_rows, test_rows, missing_report = build_metadata(
        frames_root=frames_root,
        annot_dir=annot_dir,
        output_dir=output_dir,
        debug=args.debug,
    )

    save_metadata_csv(output_dir / "train_metadata.csv", train_rows)
    save_metadata_csv(output_dir / "val_metadata.csv", val_rows)
    save_metadata_csv(output_dir / "test_metadata.csv", test_rows)

    miss_path = output_dir / "missing_frames_report.csv"
    miss_path.parent.mkdir(parents=True, exist_ok=True)
    if missing_report:
        save_metadata_csv(miss_path, missing_report)
        print(f"Missing frames: {len(missing_report)} entries (see {miss_path})")
    else:
        pd.DataFrame(
            columns=["video_id", "view_id", "missing_frame", "annotation_file"],
        ).to_csv(miss_path, index=False)
        print(f"No missing frames; wrote empty report: {miss_path}")


if __name__ == "__main__":
    main()
