#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SurgicalActions160 预处理脚本
-------------------------------------
功能一：对 data/Landscopy/SurgicalActions160 下所有视频进行数字编号并拷贝到
       data/Landscopy/SurgicalActions160_renumbered
功能二：对重命名后的视频进行抽帧，帧保存到
       data/Surge_Frames/SurgicalActions160/frames/fps{fps}
功能三：基于帧目录结构，为每个“视频帧文件夹”生成一个 txt（写入所有帧的路径），
       txt 统一放在
       data/Surge_Frames/SurgicalActions160/clip_infos_fps{fps}
       同时生成：
         - metadata_fps{fps}.csv（Index, clip_path, label, label_name, case_id, clip_idx）
         - 4-fold 划分：train_metadata_fold{i}_fps{fps}.csv / test_metadata_fold{i}_fps{fps}.csv
"""

import argparse
import random
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# 🧩 Step 1: 拷贝并重命名视频（与 extract_gynsurg.py 的逻辑保持一致风格）
def clean_videos(
    src_root: str = "data/Landscopy/SurgicalActions160",
    dst_root: str = "data/Landscopy/SurgicalActions160_renumbered",
):
    src_root_path = Path(src_root)
    dst_root_path = Path(dst_root)
    dst_root_path.mkdir(parents=True, exist_ok=True)

    video_files = list(src_root_path.rglob("*.mp4"))
    video_files.sort()

    print(f"🎥 检测到 {len(video_files)} 个视频文件，开始拷贝并按数字重命名...")

    # 按子目录分别处理，保持相对结构不变
    for folder in sorted({v.parent for v in video_files}):
        rel = folder.relative_to(src_root_path)
        out_subdir = dst_root_path / rel
        out_subdir.mkdir(parents=True, exist_ok=True)

        vids = sorted(folder.glob("*.mp4"))
        for idx, vid in enumerate(vids, start=1):
            new_name = f"{idx:05d}.mp4"
            new_path = out_subdir / new_name
            shutil.copy2(vid, new_path)
        print(f"✅ {rel}: {len(vids)} 个视频已复制并重命名。")

    print("🎉 所有视频文件已复制并按数字命名。")
    return dst_root_path


# 🧩 Step 2: 抽帧（基本沿用 extract_gynsurg.py 中的实现）
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
        print(f"⚠️ 未在 {input_path} 下找到匹配 {pattern} 的视频文件。")
        return

    print(f"\n🎞️ 共检测 {len(video_files)} 个视频，开始抽帧...\n")

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
            continue

    print("\n🎉 抽帧任务完成。")

    if save_failed and failed_videos:
        fail_log = output_path / "failed_videos.txt"
        with fail_log.open("w", encoding="utf-8", errors="ignore") as f:
            f.write("\n".join(failed_videos))
        print(f"⚠️ 共 {len(failed_videos)} 个视频抽帧失败，详情见: {fail_log}")


# 🧩 Step 3: 为每个视频生成 txt + metadata + 4-fold
def generate_txt_file(video_dir: Path, txt_path: Path) -> int:
    """
    为单个视频目录生成 txt，写入该视频所有帧的路径（从项目根目录起，通常以 data/... 开头）。
    返回该视频包含的帧数。
    """
    frame_files = sorted(
        [p for p in video_dir.iterdir() if p.is_file()],
        key=lambda p: p.name,
    )

    with txt_path.open("w", encoding="utf-8") as f:
        for frame_path in frame_files:
            # 直接写 data/... 开头的相对路径（不做再裁剪），便于在项目根目录下使用
            rel_path = frame_path
            f.write(str(rel_path).replace("\\", "/") + "\n")

    return len(frame_files)


def build_metadata(frames_root: Path, clip_infos_dir: Path):
    """
    遍历 frames_root 目录下的动作子文件夹和视频文件夹：
    - 为每个视频生成一个 txt
    - 汇总 metadata 列表
    label 映射规则：
      - 按动作子文件夹名称排序，依次赋值 0,1,2,...
    """
    metadata = []
    index = 0
    case_id_counter = 0  # 全局递增的 case_id，保证是纯数字且唯一

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

            txt_path = clip_infos_dir / f"{video_name}.txt"
            num_frames = generate_txt_file(video_dir, txt_path)
            if num_frames == 0:
                continue

            clip_rel_path = txt_path

            # 为兼容 SurgicalVideoDataset / eval 逻辑，增加纯数字的 case_id 和 clip_idx 字段
            # - case_id: 全局递增的整数 ID（0,1,2,...），每个视频一个唯一 case_id
            # - clip_idx: 当前视频内部的 clip 序号；目前每个视频只有一个 txt，对应一个 clip，统一设为 0
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
    基于“视频级别（metadata 行）”做 4-fold 划分，按 label 做大致均匀分配。
    返回 folds: List[List[int]]，每个内部 list 为该 fold 中样本在 metadata 中的索引。
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
        description="SurgicalActions160 预处理：重命名 + 抽帧 + txt + metadata + 4-fold"
    )
    parser.add_argument(
        "--src_root",
        type=str,
        default="data/Landscopy/SurgicalActions160",
        help="原始 SurgicalActions160 视频根目录",
    )
    parser.add_argument(
        "--dst_root",
        type=str,
        default="data/Landscopy/SurgicalActions160_renumbered",
        help="重命名后视频保存目录",
    )
    parser.add_argument(
        "--frames_root",
        type=str,
        default="data/Surge_Frames/SurgicalActions160/frames",
        help="抽帧保存目录（默认会自动追加 fps 子目录）",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="抽帧帧率",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="4-fold 划分随机种子",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="是否打印 ffmpeg 调试信息",
    )

    args = parser.parse_args()
    fps_tag = f"fps{args.fps}"

    # 1. 重命名并拷贝视频
    dst_clean = clean_videos(args.src_root, args.dst_root)

    # 2. 抽帧
    default_frames_root = parser.get_default("frames_root")
    frames_dir = Path(args.frames_root)
    if args.frames_root == default_frames_root:
        frames_dir = frames_dir / fps_tag

    videos_to_frames(
        input_path=str(dst_clean),
        output_path=str(frames_dir),
        fps=args.fps,
        debug=args.debug,
    )

    # 3. 生成 txt & metadata & 4-fold
    base_dir = Path("data/Surge_Frames/SurgicalActions160")
    clip_infos_dir = base_dir / f"clip_infos_{fps_tag}"
    clip_infos_dir.mkdir(parents=True, exist_ok=True)

    metadata = build_metadata(frames_dir, clip_infos_dir)
    print(
        f"发现 {len(metadata)} 个视频 clip，将写入 metadata_{fps_tag}.csv 和 4-fold 划分。"
    )

    metadata_csv_path = base_dir / f"metadata_{fps_tag}.csv"
    save_csv(metadata_csv_path, metadata)
    print(f"保存总 metadata 到: {metadata_csv_path}")

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


