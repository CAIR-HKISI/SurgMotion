#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GynSurg dataset 全流程脚本
-------------------------------------
1. 拷贝并重命名 data/Landscopy/[name]_dataset 下的原始 mp4
2. 抽帧到 data/Surge_Frames/[name]/frames_fps{fps}
3. 生成 clips_info_fps{fps} 及 metadata_fps{fps}.csv + 4-fold 划分
"""

import argparse
import random
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def _to_posix(path: Path) -> str:
    return str(path).replace("\\", "/")


def _rel_from_surges_frames(path: Path) -> Path:
    """
    返回以 data/Surge_Frames 开头的相对路径；若无法匹配则原样返回。
    """
    parts = path.parts
    for idx in range(len(parts) - 1):
        if parts[idx] == "data" and parts[idx + 1] == "Surge_Frames":
            return Path(*parts[idx:])
    return path


# 🧩 Step 1: 拷贝并重命名视频
def clean_videos(
    src_root="data/Landscopy/GynSurg_Action_Segments",
    dst_root="data/Landscopy/GynSurg_Action_Segments_Clean",
):
    """
    将原始 mp4 拷贝到一个新的目录并按 00001.mp4 重新编号。
    """
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    if not src_root.exists():
        raise FileNotFoundError(f"未找到源目录: {src_root}")

    dst_root.mkdir(parents=True, exist_ok=True)

    video_files = sorted(src_root.rglob("*.mp4"))
    print(f"🎥 检测到 {len(video_files)} 个视频文件，开始拷贝并重命名...")

    for folder in sorted({v.parent for v in video_files}):
        rel = folder.relative_to(src_root)
        out_subdir = dst_root / rel
        out_subdir.mkdir(parents=True, exist_ok=True)

        vids = sorted(folder.glob("*.mp4"))
        for idx, vid in enumerate(vids, start=1):
            new_name = f"{idx:05d}.mp4"
            new_path = out_subdir / new_name
            shutil.copy2(vid, new_path)
        print(f"✅ {rel}: {len(vids)} 个视频已复制重命名。")

    print("🎉 所有视频文件已复制并按数字命名。")
    return dst_root


# 🧩 Step 2: 抽帧
def videos_to_frames(
    input_path,
    output_root,
    fps=30,
    pattern="*.mp4",
    debug=False,
    save_failed=True,
):
    """
    将 mp4 抽帧至 frames_fps{fps} 目录，保持原有子目录结构。
    """
    input_path = Path(input_path)
    output_root = Path(output_root)
    frames_root = output_root / f"frames_fps{fps}"
    frames_root.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"抽帧输入目录不存在: {input_path}")

    video_files = sorted(input_path.rglob(pattern))
    if not video_files:
        print(f"⚠️ 未在 {input_path} 下找到匹配 {pattern} 的视频文件。")
        return frames_root

    print(f"\n🎞️ 共检测 {len(video_files)} 个视频，开始抽帧 -> {frames_root}\n")

    failed_videos = []

    for vid_path in tqdm(video_files, desc="Extracting frames"):
        rel_path = vid_path.relative_to(input_path).parent
        out_folder = frames_root / rel_path / vid_path.stem
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
            print("🔍 FFmpeg 命令:", " ".join(ffmpeg_cmd))

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
            print(f"\n❌ 抽帧失败: {vid_path}")
            if "Invalid data found" in log:
                print("⚠️ 视频损坏或无法解析")
            elif "moov atom not found" in log:
                print("⚠️ 视频文件不完整（缺少索引）")
            elif "Error while opening filter" in log:
                print("⚠️ 滤镜错误，请检查视频宽高")
            else:
                if debug:
                    print("详细错误:\n", log[:400])
            failed_videos.append(str(vid_path))

    print("\n🎉 抽帧任务完成。")

    if save_failed and failed_videos:
        fail_log = frames_root / "failed_videos.txt"
        with fail_log.open("w", encoding="utf-8", errors="ignore") as f:
            f.write("\n".join(failed_videos))
        print(f"⚠️ 共 {len(failed_videos)} 个视频抽帧失败，详情见: {fail_log}")

    return frames_root


# 🧩 Step 3: 生成 clip txt 与 metadata
def generate_txt_file(video_dir: Path, txt_path: Path, base_dir: Path) -> int:
    """
    为单个视频帧目录写入帧相对路径（相对 base_dir）。
    """
    frame_files = sorted(
        [p for p in video_dir.iterdir() if p.is_file()],
        key=lambda p: p.name,
    )

    with txt_path.open("w") as f:
        for frame_path in frame_files:
            rel_path = _rel_from_surges_frames(frame_path)
            f.write(_to_posix(rel_path) + "\n")

    return len(frame_files)


def build_metadata(frames_root: Path, clips_info_dir: Path, base_dir: Path):
    metadata = []
    index = 0

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
            video_name = video_dir.name
            txt_path = clips_info_dir / f"{video_name}.txt"
            num_frames = generate_txt_file(video_dir, txt_path, base_dir=base_dir)
            if num_frames == 0:
                continue

            clip_rel_path = _rel_from_surges_frames(txt_path)

            metadata.append(
                {
                    "Index": index,
                    "case_id": index,
                    "clip_path": _to_posix(clip_rel_path),
                    "label": label_id,
                    "label_name": label_name,
                }
            )
            index += 1

    return metadata


def save_csv(path: Path, metadata_subset):
    df = pd.DataFrame(metadata_subset)
    df.to_csv(path, index=False)
    return path


def make_4_folds(metadata, seed: int = 42):
    rng = random.Random(seed)
    label_to_indices = defaultdict(list)

    for idx, item in enumerate(metadata):
        label_to_indices[item["label"]].append(idx)

    folds = [[] for _ in range(4)]

    for indices in label_to_indices.values():
        rng.shuffle(indices)
        for i, m_idx in enumerate(indices):
            folds[i % 4].append(m_idx)

    return folds


def build_and_save_metadata(base_dir: Path, fps: int, seed: int = 42):
    frames_dir = base_dir / f"frames_fps{fps}"
    if not frames_dir.exists():
        raise FileNotFoundError(
            f"frames 目录不存在: {frames_dir}，请先执行抽帧或取消 --skip-frames"
        )

    clips_info_dir = base_dir / f"clips_info_fps{fps}"
    clips_info_dir.mkdir(parents=True, exist_ok=True)

    metadata = build_metadata(frames_dir, clips_info_dir, base_dir=base_dir)
    if not metadata:
        raise RuntimeError(f"{frames_dir} 下没有有效的 clip。")

    metadata_csv_path = base_dir / f"metadata_fps{fps}.csv"
    save_csv(metadata_csv_path, metadata)
    print(f"✅ 写入 metadata: {metadata_csv_path} ({len(metadata)} clips)")

    folds = make_4_folds(metadata, seed=seed)
    for i in range(4):
        test_indices = set(folds[i])
        train_indices = [j for k, fold in enumerate(folds) if k != i for j in fold]

        train_meta = [metadata[j] for j in train_indices]
        test_meta = [metadata[j] for j in test_indices]

        train_csv = base_dir / f"train_metadata_fold{i}_fps{fps}.csv"
        test_csv = base_dir / f"test_metadata_fold{i}_fps{fps}.csv"

        save_csv(train_csv, train_meta)
        save_csv(test_csv, test_meta)

        print(
            f"Fold {i}: train={len(train_meta)}, test={len(test_meta)} -> "
            f"{train_csv.name}, {test_csv.name}"
        )

    return metadata_csv_path


def run_pipeline(
    name: str,
    fps: int,
    seed: int,
    debug: bool,
    skip_clean: bool,
    skip_frames: bool,
    skip_metadata: bool,
):
    src_root = Path(f"data/Landscopy/{name}_dataset")
    clean_root = Path(f"data/Landscopy/{name}_dataset_renumbered")
    base_dir = Path("data/Surge_Frames") / name

    if skip_clean:
        if not clean_root.exists():
            raise FileNotFoundError(f"--skip-clean 但未找到目录: {clean_root}")
        print(f"⚠️ 跳过拷贝，直接使用 {clean_root}")
    else:
        clean_videos(src_root, clean_root)

    if skip_frames:
        frames_dir = base_dir / f"frames_fps{fps}"
        if not frames_dir.exists():
            raise FileNotFoundError(f"--skip-frames 但未找到目录: {frames_dir}")
        print(f"⚠️ 跳过抽帧，直接使用 {frames_dir}")
    else:
        videos_to_frames(
            input_path=clean_root,
            output_root=base_dir,
            fps=fps,
            debug=debug,
        )

    if skip_metadata:
        print("⚠️ 跳过 clip txt 和 metadata 生成。")
    else:
        build_and_save_metadata(base_dir, fps=fps, seed=seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GynSurg dataset: clean + frames + clip metadata (带 FPS 标识)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="GynSurg_action",
        help="数据集名 (例如 GynSurg_action)",
    )
    parser.add_argument("--fps", type=int, default=30, help="抽帧帧率")
    parser.add_argument("--seed", type=int, default=42, help="4-fold 随机种子")
    parser.add_argument("--debug", action="store_true", help="打印 FFmpeg 调试信息")
    parser.add_argument("--skip-clean", action="store_true", help="跳过拷贝&重命名")
    parser.add_argument("--skip-frames", action="store_true", help="跳过抽帧")
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="跳过 clip txt 与 metadata 生成",
    )
    args = parser.parse_args()

    run_pipeline(
        name=args.name,
        fps=args.fps,
        seed=args.seed,
        debug=args.debug,
        skip_clean=args.skip_clean,
        skip_frames=args.skip_frames,
        skip_metadata=args.skip_metadata,
    )

