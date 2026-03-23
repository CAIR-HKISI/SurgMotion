#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M2CAI2016 preprocessing: annotation (25 fps) -> 1 fps alignment with extracted frames,
per-video clip list txts, and train / test / val metadata CSVs.

Annotation layout:
  - Train: train_dataset/*.txt (exclude *_timestamp.txt, *_pred.txt)
  - Test: test_dataset/*.txt (e.g. test_workflow_videoXX.txt)

Frames (1 fps): frames_root/{video_name}/{video_name}_XXXXXXXX.jpg (1-based 8-digit index).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

# Phase name -> integer label (dataset definition)
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


def videos_to_frames(
    input_videos_root: Path,
    output_frames_root: Path,
    fps: int = 1,
    debug: bool = False,
) -> None:
    """
    Stub: extract frames from videos under input_videos_root into output_frames_root.

    Expected layout after extraction: output_frames_root/{video_name}/{video_name}_%08d.jpg
    starting at index 1. Wire this to ffmpeg or your extractor when raw videos are available.
    """
    # Placeholder: hook up ffmpeg (or another extractor) when raw videos are available.
    _ = (input_videos_root, output_frames_root, fps, debug)


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
    """All unique annotation stems (video names), sorted; used for stable numeric case_id."""
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


def convert_25fps_to_1fps(
    annotations: List[Tuple[int, str]],
) -> List[Tuple[int, str]]:
    """
    Collapse 25 fps labels to 1 fps: one label per second (last wins within each second).
    Returns sorted (frame_1fps_index, phase) with frame_1fps_index in {0, 1, ...}.
    """
    bucket: Dict[int, str] = {}
    for frame_25, phase in annotations:
        frame_1 = frame_25 // 25
        bucket[frame_1] = phase
    return sorted(bucket.items(), key=lambda x: x[0])


def frame_jpg_path(frames_root: Path, video_name: str, one_based_index: int) -> Path:
    """Disk path for one extracted frame (1-based 8-digit suffix)."""
    suffix = f"{one_based_index:08d}"
    return frames_root / video_name / f"{video_name}_{suffix}.jpg"


def generate_clip_txt(video_frames_dir: Path, txt_path: Path) -> int:
    """
    Write one line per frame file under video_frames_dir (sorted by filename).
    Paths are written as POSIX strings relative to how Path stringifies (typically project-relative).
    """
    frame_files = sorted(
        [p for p in video_frames_dir.iterdir() if p.is_file()],
        key=lambda p: p.name,
    )
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w", encoding="utf-8") as f:
        for frame_path in frame_files:
            f.write(str(frame_path).replace("\\", "/") + "\n")
    return len(frame_files)


def build_metadata_for_split(
    annot_dir: Path,
    frames_root: Path,
    clip_infos_dir: Path,
    case_id_map: Dict[str, int],
    split_label: str,
    global_index_start: int,
    debug: bool = False,
) -> Tuple[List[dict], int]:
    """
    Read annotations for one split, align to 1 fps frames, emit clip txt per video,
    and return metadata rows (per labeled frame that exists on disk).
    """
    metadata: List[dict] = []
    index = global_index_start
    annot_files = _list_annotation_files(annot_dir)

    for annot_path in tqdm(annot_files, desc=f"M2CAI16 {split_label}"):
        video_name = annot_path.stem
        case_id = case_id_map[video_name]
        video_frames_dir = frames_root / video_name
        clip_txt_path = clip_infos_dir / f"{video_name}.txt"

        if not video_frames_dir.is_dir():
            if debug:
                print(f"[debug] Missing frames dir: {video_frames_dir}")
            continue

        n_frames_written = generate_clip_txt(video_frames_dir, clip_txt_path)
        if n_frames_written == 0:
            if debug:
                print(f"[debug] No frames under {video_frames_dir}, skip {video_name}")
            continue

        annotations = read_annotation_txt(annot_path)
        converted = convert_25fps_to_1fps(annotations)

        for frame_1fps, phase_name in converted:
            # Extracted naming uses 1-based index matching 1 fps timeline
            disk_idx = frame_1fps + 1
            fp = frame_jpg_path(frames_root, video_name, disk_idx)
            if not fp.is_file():
                if debug:
                    print(f"[debug] Missing frame for label: {fp}")
                continue

            label = PHASE_MAPPING.get(phase_name, -1)
            metadata.append(
                {
                    "Index": index,
                    "clip_path": str(clip_txt_path).replace("\\", "/"),
                    "label": label,
                    "label_name": phase_name,
                    "case_id": case_id,
                    "clip_idx": 0,
                }
            )
            index += 1

    return metadata, index


def build_metadata(
    frames_root: Path,
    train_annot_dir: Path,
    test_annot_dir: Path,
    output_dir: Path,
    debug: bool = False,
) -> Tuple[List[dict], List[dict]]:
    """
    End-to-end metadata build: case_id map from all videos, clip txts, train and test rows.
    Validation split reuses test annotations (caller duplicates CSV if needed).
    """
    sorted_names = collect_sorted_video_names(train_annot_dir, test_annot_dir)
    case_id_map = build_case_id_map(sorted_names)
    clip_infos_dir = output_dir / "clip_infos"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_meta, next_idx = build_metadata_for_split(
        train_annot_dir,
        frames_root,
        clip_infos_dir,
        case_id_map,
        "train",
        global_index_start=0,
        debug=debug,
    )
    test_meta, _ = build_metadata_for_split(
        test_annot_dir,
        frames_root,
        clip_infos_dir,
        case_id_map,
        "test",
        global_index_start=0,
        debug=debug,
    )
    _ = next_idx  # train indices are contiguous; test restarts at 0 per user spec
    return train_meta, test_meta


def save_metadata_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Saved {len(rows)} rows -> {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="M2CAI2016: build clip txts and train/test/val metadata from 25 fps labels and 1 fps frames."
    )
    p.add_argument(
        "--frames_root",
        type=Path,
        default=Path("data/Surge_Frames/M2CAI16/frames"),
        help="Root folder containing per-video frame directories",
    )
    p.add_argument(
        "--train_annot_dir",
        type=Path,
        default=Path("data/Landscopy/m2cai16/train_dataset"),
        help="Directory with training phase annotation .txt files",
    )
    p.add_argument(
        "--test_annot_dir",
        type=Path,
        default=Path("data/Landscopy/m2cai16/test_dataset"),
        help="Directory with test phase annotation .txt files",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/Surge_Frames/M2CAI16"),
        help="Output directory for clip_infos/ and *_metadata.csv",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print extra diagnostics (missing dirs/frames)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    frames_root: Path = args.frames_root
    train_annot_dir: Path = args.train_annot_dir
    test_annot_dir: Path = args.test_annot_dir
    output_dir: Path = args.output_dir

    train_meta, test_meta = build_metadata(
        frames_root=frames_root,
        train_annot_dir=train_annot_dir,
        test_annot_dir=test_annot_dir,
        output_dir=output_dir,
        debug=args.debug,
    )

    save_metadata_csv(output_dir / "train_metadata.csv", train_meta)
    save_metadata_csv(output_dir / "test_metadata.csv", test_meta)
    # Val split mirrors test (same annotations and frames policy)
    save_metadata_csv(output_dir / "val_metadata.csv", list(test_meta))


if __name__ == "__main__":
    main()
