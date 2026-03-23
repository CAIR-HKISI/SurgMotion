#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SurgicalActions160 preprocessing script
-------------------------------------
Feature 1: Rename and copy all videos in data/Landscopy/SurgicalActions160 to
           data/Landscopy/SurgicalActions160_renumbered
Feature 2: Extract frames from renamed videos and save them to
           data/Surge_Frames/SurgicalActions160_v1/frames/fps{fps}
Feature 3: Generate a txt for each video frame directory (write all frame paths),
           txt is stored in
           data/Surge_Frames/SurgicalActions160_v1/clip_infos_fps{fps}
           At the same time, generate:
           - data/Surge_Frames/SurgicalActions160_v1/metadata_fps{fps}.csv
           - 4-fold split: train_metadata_fold{i}_fps{fps}.csv / test_metadata_fold{i}_fps{fps}.csv
"""

import argparse
import random
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

BASE_DIR = Path("data/Surge_Frames/SurgicalActions160_v1")


# 🧩 Step 1: Copy and rename videos (consistent with the logic of extract_gynsurg.py)
def clean_videos(
    src_root: str = "data/Landscopy/SurgicalActions160",
    dst_root: str = "data/Landscopy/SurgicalActions160_renumbered",
):
    src_root_path = Path(src_root)
    dst_root_path = Path(dst_root)
    dst_root_path.mkdir(parents=True, exist_ok=True)

    video_files = list(src_root_path.rglob("*.mp4"))
    video_files.sort()

    print(f"🎥 Detected {len(video_files)} video files, starting to copy and rename by number...")

    # Process by subdirectories, keeping the relative structure unchanged
    for folder in sorted({v.parent for v in video_files}):
        rel = folder.relative_to(src_root_path)
        out_subdir = dst_root_path / rel
        out_subdir.mkdir(parents=True, exist_ok=True)

        vids = sorted(folder.glob("*.mp4"))
        for idx, vid in enumerate(vids, start=1):
            new_name = f"{idx:05d}.mp4"
            new_path = out_subdir / new_name
            shutil.copy2(vid, new_path)
        print(f"✅ {rel}: {len(vids)} videos copied and renamed.")

    print("🎉 All video files copied and named by number.")
    return dst_root_path


# 🧩 Step 2: Extract frames (based on the implementation of extract_gynsurg.py)
def videos_to_frames(
    input_path: str,
    output_path: str,
    fps: int = 30,
    pattern: str = "*.mp4",
    debug: bool = False,
    save_failed: bool = True,
):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    video_files = list(input_path.rglob(pattern))
    video_files.sort()

    if not video_files:
        print(f"⚠️ No video files matching {pattern} found in {input_path}.")
        return

    print(f"\n🎞️ Detected {len(video_files)} videos, starting to extract frames...\n")

    failed_videos = []

    for vid_path in tqdm(video_files, desc="Extracting frames"):
        rel_path = vid_path.relative_to(input_path).parent
        out_folder = output_path / rel_path / vid_path.stem
        out_folder.mkdir(parents=True, exist_ok=True)
        output_pattern = out_folder / f"{vid_path.stem}_%05d.jpg"

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
            if "Invalid data found" in log:
                print("⚠️ Video is damaged or cannot be parsed")
            elif "moov atom not found" in log:
                print("⚠️ Video file is incomplete (missing index)")
            elif "Error while opening filter" in log:
                print("⚠️ Filter error, please check the video width and height")
            else:
                if debug:
                    print("Detailed error:\n", log[:400])
            failed_videos.append(str(vid_path))
            continue

    print("\n🎉 Frame extraction task completed.")

    if save_failed and failed_videos:
        fail_log = output_path / "failed_videos.txt"
        with fail_log.open("w", encoding="utf-8", errors="ignore") as f:
            f.write("\n".join(failed_videos))
        print(f"⚠️ {len(failed_videos)} videos failed to extract frames, details: {fail_log}")


# 🧩 Step 3: Generate txt + metadata + 4-fold for each video
def generate_txt_file(video_dir: Path, txt_path: Path) -> int:
    """
    Generate a txt for a single video directory, write the paths of all frames in the video (starting from the project root, usually starting with data/...).
    Return the number of frames in the video.
    """
    frame_files = sorted(
        [p for p in video_dir.iterdir() if p.is_file()],
        key=lambda p: p.name,
    )

    with txt_path.open("w", encoding="utf-8") as f:
        for frame_path in frame_files:
            # Write relative paths starting with data/... (no further trimming), for use in the project root directory
            rel_path = frame_path
            f.write(str(rel_path).replace("\\", "/") + "\n")

    return len(frame_files)


def build_metadata(frames_root: Path, clip_infos_dir: Path):
    """
    Traverse the action subdirectories and video directories in the frames_root directory:
    - Generate a txt for each video
    - Summarize the metadata list
    Label mapping rule:
      - Sort by action subdirectory name, assign values 0,1,2,...
    """
    metadata = []
    index = 0
    case_id_counter = 0  # Global increasing case_id, guaranteed to be pure digits and unique

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

        for video_dir in video_dirs:
            # clip_infos directory keeps the hierarchy of "action/video", to avoid overlapping of videos with the same name under different actions
            rel_video = video_dir.relative_to(frames_root)
            txt_parent = clip_infos_dir / rel_video.parent
            txt_parent.mkdir(parents=True, exist_ok=True)
            txt_path = txt_parent / f"{video_dir.name}.txt"
            num_frames = generate_txt_file(video_dir, txt_path)
            if num_frames == 0:
                continue

            clip_rel_path = txt_path

            # To be compatible with SurgicalVideoDataset / eval logic, add pure numeric case_id and clip_idx fields
            # - case_id: Global increasing integer ID (0,1,2,...), one unique case_id per video
            # - clip_idx: The clip number within the current video; currently each video has only one txt, corresponding to one clip, set to 0
            case_id = case_id_counter
            clip_idx = 0

            metadata.append(
                {
                    "Index": index,
                    "clip_path": str(clip_rel_path).replace("\\", "/"),
                    "label": label_id,
                    "label_name": label_name,
                    "case_id": case_id,
                    "clip_idx": clip_idx,
                }
            )
            index += 1
            case_id_counter += 1

    return metadata


def save_csv(path: Path, metadata_subset):
    df = pd.DataFrame(metadata_subset)
    df.to_csv(path, index=False)
    return path


def make_4_folds(metadata, seed: int = 42):
    """
    Based on "video level" (metadata rows), perform 4-fold splitting, roughly evenly distribute by label.
    Return folds: List[List[int]], each internal list is the index of the samples in the metadata for that fold.
    """
    rng = random.Random(seed)
    label_to_indices = defaultdict(list)

    for idx, item in enumerate(metadata):
        label_to_indices[item["label"]].append(idx)

    folds = [[] for _ in range(4)]

    for _, indices in label_to_indices.items():
        rng.shuffle(indices)
        for i, m_idx in enumerate(indices):
            folds[i % 4].append(m_idx)

    return folds


def main():
    parser = argparse.ArgumentParser(
        description="SurgicalActions160 preprocessing: renaming + frame extraction + txt + metadata + 4-fold"
    )
    parser.add_argument(
        "--src_root",
        type=str,
        default="data/Landscopy/SurgicalActions160",
        help="Original SurgicalActions160 video root directory",
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
        help="Frame extraction save directory (default will automatically append fps subdirectory)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frame extraction FPS",
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
        help="Whether to print ffmpeg debug information",
    )

    args = parser.parse_args()
    fps_tag = f"fps{args.fps}"

    # 1. Rename and copy videos
    dst_clean = clean_videos(args.src_root, args.dst_root)

    # 2. Extract frames
    default_frames_root = Path(parser.get_default("frames_root"))
    frames_dir = Path(args.frames_root)
    if frames_dir == default_frames_root:
        frames_dir = frames_dir / fps_tag

    videos_to_frames(
        input_path=str(dst_clean),
        output_path=str(frames_dir),
        fps=args.fps,
        debug=args.debug,
    )

    # 3. Generate txt & metadata & 4-fold
    base_dir = BASE_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    clip_infos_dir = base_dir / f"clip_infos_{fps_tag}"
    clip_infos_dir.mkdir(parents=True, exist_ok=True)

    metadata = build_metadata(frames_dir, clip_infos_dir)
    print(
        f"Found {len(metadata)} video clips, will write to metadata_{fps_tag}.csv and 4-fold split."
    )

    metadata_csv_path = base_dir / f"metadata_{fps_tag}.csv"
    save_csv(metadata_csv_path, metadata)
    print(f"Saved total metadata to: {metadata_csv_path}")

    folds = make_4_folds(metadata, seed=args.seed)

    for i in range(4):
        test_indices = set(folds[i])
        train_indices = [j for k, fold in enumerate(folds) if k != i for j in fold]

        train_meta = [metadata[j] for j in train_indices]
        test_meta = [metadata[j] for j in test_indices]

        train_csv = base_dir / f"train_metadata_fold{i}_{fps_tag}.csv"
        test_csv = base_dir / f"test_metadata_fold{i}_{fps_tag}.csv"

        save_csv(train_csv, train_meta)
        save_csv(test_csv, test_meta)

        print(
            f"Fold {i}: train={len(train_meta)} clips, test={len(test_meta)} clips -> "
            f"{train_csv.name}, {test_csv.name}"
        )


if __name__ == "__main__":
    main()


