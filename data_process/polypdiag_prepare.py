#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PolypDiag preprocessing script
-------------------------------------
Data organization (already exists):
  - data/GI_Videos/PolypDiag/videos/*.mp4
  - data/GI_Videos/PolypDiag/splits/train.txt
  - data/GI_Videos/PolypDiag/splits/val.txt

Where train.txt / val.txt each line format:
    <video_filename>.mp4,<label>
label is 0/1 (e.g. normal=0, abnormal=1).

This script functions:
1. Copy and rename all mp4 in videos to consecutive numbers:
     data/GI_Videos/PolypDiag/videos_renumbered/{00001.mp4, 00002.mp4, ...}
   Record the mapping of old file names -> new file names (csv).

2. Extract frames from renamed videos:
     data/Surge_Frames/PolypDiag/frames/{00001, 00002, ...}/*.jpg

3. Generate a txt for each video frame folder, containing all frame paths:
     data/Surge_Frames/PolypDiag/clip_infos/{00001.txt, ...}
   Write paths starting with data/ (relative to the project root), for downstream use.

4. Based on the splits/train.txt and splits/val.txt in the original dataset,
   use the "renamed video numbers" to generate:
     - data/Surge_Frames/PolypDiag/train_metadata.csv
     - data/Surge_Frames/PolypDiag/val_metadata.csv

Fields are:
     Index, clip_path, label, label_name, orig_name, new_name
Where:
     - label_name for binary classification is default: {0: "normal", 1: "abnormal"}, can be modified as needed.
"""

import argparse
import csv
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm


# -----------------------
# Helper function: rename videos
# -----------------------
def renumber_videos(
    src_videos_dir: Path,
    dst_videos_dir: Path,
) -> Dict[str, str]:
    """
    Copy and rename all mp4 in src_videos_dir to dst_videos_dir,
    return a dictionary: {original file name: new file name}.
    """
    dst_videos_dir.mkdir(parents=True, exist_ok=True)
    video_files = sorted(src_videos_dir.glob("*.mp4"))

    mapping: Dict[str, str] = {}

    print(f"🎥 Detected {len(video_files)} videos, will be renamed and copied...")

    for idx, vid in enumerate(tqdm(video_files, desc="Copy & rename"), start=1):
        new_name = f"{idx:05d}.mp4"
        new_path = dst_videos_dir / new_name
        shutil.copy2(vid, new_path)
        mapping[vid.name] = new_name

    print(f"✅ Renaming completed, output directory: {dst_videos_dir}")
    return mapping


def save_mapping_csv(mapping: Dict[str, str], out_path: Path):
    """
    Save the mapping of "original file name -> new file name" to CSV, for reference and debugging.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["orig_name", "new_name"])
        for orig, new in sorted(mapping.items()):
            writer.writerow([orig, new])
    print(f"💾 Saved renaming mapping to: {out_path}")


def load_or_build_name_mapping(src_videos_dir: Path, dst_videos_dir: Path) -> Dict[str, str]:
    """
    Load or build the mapping of orig_name -> new_name:
      - If polypdiag_name_mapping.csv exists in the same directory as dst_videos_dir, load the mapping from it;
      - Otherwise, fall back to "identity mapping": orig_name maps to itself (applicable for unrenamed cases).
    """
    mapping_csv_path = dst_videos_dir.parent / "polypdiag_name_mapping.csv"
    mapping: Dict[str, str] = {}

    if mapping_csv_path.exists():
        print(f"📄 Loaded mapping from existing file: {mapping_csv_path}")
        with mapping_csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                orig = row.get("orig_name")
                new = row.get("new_name")
                if orig and new:
                    mapping[orig] = new
        print(f"✅ Loaded {len(mapping)} orig_name -> new_name mappings")
    else:
        print("⚠️ polypdiag_name_mapping.csv not found, using original file names as mapping (orig_name -> orig_name)")
        for vid in src_videos_dir.glob("*.mp4"):
            mapping[vid.name] = vid.name
        print(f"✅ Built {len(mapping)} identity mappings")

    return mapping


# -----------------------
# Extract frames
# -----------------------
def videos_to_frames(
    input_path: Path,
    output_path: Path,
    fps: int = 30,
    debug: bool = False,
):
    """
    Extract frames from all mp4 in input_path to output_path:
      input_path/00001.mp4 -> output_path/00001/00001_%05d.jpg
    """
    output_path.mkdir(parents=True, exist_ok=True)
    video_files = sorted(input_path.glob("*.mp4"))

    if not video_files:
        print(f"⚠️ No mp4 videos found in {input_path}.")
        return

    print(f"\n🎞️ Detected {len(video_files)} videos, starting to extract frames...\n")

    failed_videos: List[str] = []

    for vid_path in tqdm(video_files, desc="Extracting frames"):
        vid_id = vid_path.stem  # e.g. "00001"
        out_folder = output_path / vid_id
        out_folder.mkdir(parents=True, exist_ok=True)
        output_pattern = out_folder / f"{vid_id}_%05d.jpg"

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(vid_path.resolve()),
            "-safe",
            "0",
            "-vf",
            f"fps={fps},scale=512:-1:flags=bicubic",
            "-vsync",
            "2",
            "-qscale:v",
            "2",
            str(output_pattern),
        ]

        if debug:
            print("🔍 FFmpeg command:", " ".join(ffmpeg_cmd))

        try:
            result = subprocess.run(
                ffmpeg_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if debug:
                print(result.stderr.decode("utf-8", errors="ignore")[:200])
        except subprocess.CalledProcessError as e:
            log = e.stderr.decode("utf-8", errors="ignore")
            print(f"\n❌ Frame extraction failed: {vid_path}")
            if debug:
                print("Detailed error:\n", log[:400])
            failed_videos.append(str(vid_path))
            continue

    print("\n🎉 Frame extraction task completed.")

    if failed_videos:
        fail_log = output_path / "failed_videos.txt"
        with fail_log.open("w", encoding="utf-8") as f:
            f.write("\n".join(failed_videos))
        print(f"⚠️ {len(failed_videos)} videos failed to extract frames, details: {fail_log}")


# -----------------------
# Generate txt + metadata for each video
# -----------------------
def generate_clip_txt(video_frames_dir: Path, txt_path: Path) -> int:
    """
    Generate a txt for a single video frame directory video_frames_dir, containing all frame paths in data/... format.
    Return the number of frames.
    """
    frame_files = sorted(
        [p for p in video_frames_dir.iterdir() if p.is_file()],
        key=lambda p: p.name,
    )

    with txt_path.open("w", encoding="utf-8") as f:
        for frame_path in frame_files:
            # Write paths starting with data/... (relative to the project root)
            rel_path = frame_path
            f.write(str(rel_path).replace("\\", "/") + "\n")

    return len(frame_files)


def load_split_file(split_path: Path) -> Dict[str, int]:
    """
    Load train.txt / val.txt, return {orig_name: label_int}.
    The file format is: <video_filename>.mp4,<label>
    """
    mapping: Dict[str, int] = {}
    with split_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, label_str = line.split(",")
            mapping[name] = int(label_str)
    return mapping


def build_metadata_from_splits(
    frames_root: Path,
    clip_infos_dir: Path,
    train_split: Dict[str, int],
    val_split: Dict[str, int],
    name_mapping: Dict[str, str],
    label_name_map: Dict[int, str] = None,
):
    """
    Based on the orig_name -> label in splits, then through name_mapping to new_name,
    generate a txt for the frame directory corresponding to new_name, and build train / val metadata respectively.
    """
    if label_name_map is None:
        # Default binary classification: 0=normal, 1=abnormal
        label_name_map = {0: "normal", 1: "abnormal"}

    clip_infos_dir.mkdir(parents=True, exist_ok=True)

    train_metadata = []
    val_metadata = []
    index = 0
    case_id_counter = 0  # Global incrementing case_id, guaranteed to be pure digits and unique

    def process_split(split_dict: Dict[str, int], target_list: List[dict], split_name: str):
        nonlocal index, case_id_counter
        missing = []
        for orig_name, label in split_dict.items():
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
            num_frames = generate_clip_txt(video_frames_dir, txt_path)
            if num_frames == 0:
                continue

            # To be compatible with SurgicalVideoDataset / evaluation, add pure digit case_id and clip_idx fields
            # - case_id: Global incrementing integer ID (0,1,2,...), one unique case_id per video
            # - clip_idx: The clip number within the current video; currently each video has only one txt, corresponding to one clip, set to 0
            case_id = case_id_counter
            clip_idx = 0

            item = {
                "Index": index,
                "clip_path": str(txt_path).replace("\\", "/"),
                "label": int(label),
                "label_name": label_name_map.get(int(label), str(label)),
                "orig_name": orig_name,
                "new_name": new_name,
                "case_id": case_id,
                "clip_idx": clip_idx,
            }
            target_list.append(item)
            index += 1
            case_id_counter += 1

        if missing:
            print(f"⚠️ In {split_name} split, {len(missing)} files not found in renaming mapping or frame directory. Example: {missing[:5]}")

    process_split(train_split, train_metadata, "train")
    process_split(val_split, val_metadata, "val")

    return train_metadata, val_metadata


def save_metadata_csv(path: Path, metadata: List[dict]):
    df = pd.DataFrame(metadata)
    df.to_csv(path, index=False)
    print(f"💾 Saved metadata to: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="PolypDiag preprocessing: video renaming + frame extraction + txt + train/val metadata"
    )
    parser.add_argument(
        "--src_videos",
        type=str,
        default="data/GI_Videos/PolypDiag/videos",
        help="Original PolypDiag video directory",
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
        "--fps",
        type=int,
        default=1,
        help="Frame extraction FPS",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to print ffmpeg debug information",
    )

    args = parser.parse_args()

    src_videos_dir = Path(args.src_videos)
    dst_videos_dir = Path(args.dst_videos)
    frames_root = Path(args.frames_root)
    splits_dir = Path(args.splits_dir)

    # 1. Get orig_name -> new_name mapping
    #    - If renaming has already been performed and polypdiag_name_mapping.csv exists, load it directly;
    #    - If not renamed, fall back to orig_name -> orig_name identity mapping.
    # name_mapping = load_or_build_name_mapping(src_videos_dir, dst_videos_dir)

    # 2. Extract frames (optional)
    #    If you have already extracted frames beforehand, comment out the following line;
    #    If you want this script to handle frame extraction, keep this line.
    # videos_to_frames(dst_videos_dir, frames_root, fps=args.fps, debug=args.debug)

    # 3. Build train/val metadata based on splits
    train_split = load_split_file(splits_dir / "train.txt")
    val_split = load_split_file(splits_dir / "val.txt")

    base_dir = Path("data/Surge_Frames/PolypDiag_v1")
    clip_infos_dir = base_dir / "clip_infos"

    train_meta, val_meta = build_metadata_from_splits(
        frames_root=frames_root,
        clip_infos_dir=clip_infos_dir,
        train_split=train_split,
        val_split=val_split,
        name_mapping=name_mapping,
    )

    train_csv = base_dir / "train_metadata.csv"
    val_csv = base_dir / "val_metadata.csv"
    save_metadata_csv(train_csv, train_meta)
    save_metadata_csv(val_csv, val_meta)


if __name__ == "__main__":
    main()


