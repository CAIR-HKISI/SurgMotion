#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PolypDiag 预处理脚本
-------------------------------------
数据组织（已存在）:
  - data/GI_Videos/PolypDiag/videos/*.mp4
  - data/GI_Videos/PolypDiag/splits/train.txt
  - data/GI_Videos/PolypDiag/splits/val.txt

其中 train.txt / val.txt 每行格式:
    <video_filename>.mp4,<label>
label 为 0/1（如: normal=0, abnormal=1）。

本脚本功能:
1. 将 videos 下所有 mp4 复制并重命名为连续数字:
     data/GI_Videos/PolypDiag/videos_renumbered/{00001.mp4, 00002.mp4, ...}
   同时记录旧文件名 -> 新文件名的映射 (csv)。

2. 对重命名后的视频统一抽帧:
     data/Surge_Frames/PolypDiag/frames/{00001, 00002, ...}/*.jpg

3. 为每个视频帧文件夹生成一个 txt，内容为所有帧的路径:
     data/Surge_Frames/PolypDiag/clip_infos/{00001.txt, ...}
   路径写为以 data/ 开头的项目内相对路径，方便下游使用。

4. 根据原始 splits/train.txt 和 splits/val.txt 中的划分与标签，
   使用 “重命名之后的视频编号” 生成:
     - data/Surge_Frames/PolypDiag/train_metadata.csv
     - data/Surge_Frames/PolypDiag/val_metadata.csv

   字段为:
     Index, clip_path, label, label_name, orig_name, new_name
   其中:
     - label_name 对二分类默认为: {0: "normal", 1: "abnormal"}，可根据需要修改。
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
# 辅助函数: 重命名视频
# -----------------------
def renumber_videos(
    src_videos_dir: Path,
    dst_videos_dir: Path,
) -> Dict[str, str]:
    """
    将 src_videos_dir 下所有 mp4 复制并重命名到 dst_videos_dir，
    返回一个字典: {原始文件名: 新文件名}。
    """
    dst_videos_dir.mkdir(parents=True, exist_ok=True)
    video_files = sorted(src_videos_dir.glob("*.mp4"))

    mapping: Dict[str, str] = {}

    print(f"🎥 检测到 {len(video_files)} 个视频，将进行重命名拷贝...")

    for idx, vid in enumerate(tqdm(video_files, desc="Copy & rename"), start=1):
        new_name = f"{idx:05d}.mp4"
        new_path = dst_videos_dir / new_name
        shutil.copy2(vid, new_path)
        mapping[vid.name] = new_name

    print(f"✅ 重命名完成，输出目录: {dst_videos_dir}")
    return mapping


def save_mapping_csv(mapping: Dict[str, str], out_path: Path):
    """
    保存 “原始文件名 -> 新文件名” 的映射到 CSV，便于对照和调试。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["orig_name", "new_name"])
        for orig, new in sorted(mapping.items()):
            writer.writerow([orig, new])
    print(f"💾 保存重命名映射到: {out_path}")


def load_or_build_name_mapping(src_videos_dir: Path, dst_videos_dir: Path) -> Dict[str, str]:
    """
    加载或构建 orig_name -> new_name 的映射:
      - 若 dst_videos_dir 同级目录下已存在 polypdiag_name_mapping.csv，则从中读取映射；
      - 否则退化为“恒等映射”：orig_name 映射到自身（适用于未重命名的情况）。
    """
    mapping_csv_path = dst_videos_dir.parent / "polypdiag_name_mapping.csv"
    mapping: Dict[str, str] = {}

    if mapping_csv_path.exists():
        print(f"📄 从已存在的映射文件加载: {mapping_csv_path}")
        with mapping_csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                orig = row.get("orig_name")
                new = row.get("new_name")
                if orig and new:
                    mapping[orig] = new
        print(f"✅ 加载到 {len(mapping)} 条 orig_name -> new_name 映射")
    else:
        print("⚠️ 未找到 polypdiag_name_mapping.csv，使用原始文件名作为映射（orig_name -> orig_name）")
        for vid in src_videos_dir.glob("*.mp4"):
            mapping[vid.name] = vid.name
        print(f"✅ 构建 {len(mapping)} 条恒等映射")

    return mapping


# -----------------------
# 抽帧
# -----------------------
def videos_to_frames(
    input_path: Path,
    output_path: Path,
    fps: int = 30,
    debug: bool = False,
):
    """
    将 input_path 下的所有 mp4 抽帧到 output_path 下:
      input_path/00001.mp4 -> output_path/00001/00001_%05d.jpg
    """
    output_path.mkdir(parents=True, exist_ok=True)
    video_files = sorted(input_path.glob("*.mp4"))

    if not video_files:
        print(f"⚠️ 未在 {input_path} 下找到 mp4 视频。")
        return

    print(f"\n🎞️ 共检测 {len(video_files)} 个视频，开始抽帧...\n")

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
            if debug:
                print("详细错误:\n", log[:400])
            failed_videos.append(str(vid_path))
            continue

    print("\n🎉 抽帧任务完成。")

    if failed_videos:
        fail_log = output_path / "failed_videos.txt"
        with fail_log.open("w", encoding="utf-8") as f:
            f.write("\n".join(failed_videos))
        print(f"⚠️ 共 {len(failed_videos)} 个视频抽帧失败，详情见: {fail_log}")


# -----------------------
# 生成每视频 txt + metadata
# -----------------------
def generate_clip_txt(video_frames_dir: Path, txt_path: Path) -> int:
    """
    为单个视频帧目录 video_frames_dir 生成 txt，写入所有帧的 data/... 路径。
    返回帧数。
    """
    frame_files = sorted(
        [p for p in video_frames_dir.iterdir() if p.is_file()],
        key=lambda p: p.name,
    )

    with txt_path.open("w", encoding="utf-8") as f:
        for frame_path in frame_files:
            # 直接写 data/... 开头的路径（相对于项目根目录）
            rel_path = frame_path
            f.write(str(rel_path).replace("\\", "/") + "\n")

    return len(frame_files)


def load_split_file(split_path: Path) -> Dict[str, int]:
    """
    读取 train.txt / val.txt，返回 {orig_name: label_int}。
    文件格式: <video_filename>.mp4,<label>
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
    根据 splits 中的 orig_name -> label, 再通过 name_mapping 转为 new_name，
    对 new_name 对应的帧目录生成 txt，并分别构建 train / val metadata。
    """
    if label_name_map is None:
        # 默认二分类: 0=normal, 1=abnormal
        label_name_map = {0: "normal", 1: "abnormal"}

    clip_infos_dir.mkdir(parents=True, exist_ok=True)

    train_metadata = []
    val_metadata = []
    index = 0
    case_id_counter = 0  # 全局递增的 case_id，保证是纯数字且唯一

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

            # 为兼容 SurgicalVideoDataset / evaluation，增加纯数字的 case_id 和 clip_idx 字段
            # - case_id: 全局递增的整数 ID（0,1,2,...），每个视频一个唯一 case_id
            # - clip_idx: 当前视频内部的 clip 序号；目前每个视频只有一个 txt，对应一个 clip，统一设为 0
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
            print(f"⚠️ 在 {split_name} 划分中，有 {len(missing)} 个文件未在重命名映射或帧目录中找到。示例: {missing[:5]}")

    process_split(train_split, train_metadata, "train")
    process_split(val_split, val_metadata, "val")

    return train_metadata, val_metadata


def save_metadata_csv(path: Path, metadata: List[dict]):
    df = pd.DataFrame(metadata)
    df.to_csv(path, index=False)
    print(f"💾 保存 metadata 到: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="PolypDiag 预处理：视频重命名 + 抽帧 + txt + train/val metadata"
    )
    parser.add_argument(
        "--src_videos",
        type=str,
        default="data/GI_Videos/PolypDiag/videos",
        help="原始 PolypDiag 视频目录",
    )
    parser.add_argument(
        "--dst_videos",
        type=str,
        default="data/GI_Videos/PolypDiag/videos_renumbered",
        help="重命名后视频目录",
    )
    parser.add_argument(
        "--frames_root",
        type=str,
        default="data/Surge_Frames/PolypDiag/frames",
        help="抽帧输出目录",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="data/GI_Videos/PolypDiag/splits",
        help="train/val 划分所在目录",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="抽帧帧率",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="是否打印 ffmpeg 调试信息",
    )

    args = parser.parse_args()

    src_videos_dir = Path(args.src_videos)
    dst_videos_dir = Path(args.dst_videos)
    frames_root = Path(args.frames_root)
    splits_dir = Path(args.splits_dir)

    # 1. 获取 orig_name -> new_name 映射
    #    - 若之前已执行过重命名并生成 polypdiag_name_mapping.csv，则此处直接加载；
    #    - 若未重命名，则退化为 orig_name -> orig_name 的恒等映射。
    # name_mapping = load_or_build_name_mapping(src_videos_dir, dst_videos_dir)

    # 2. 抽帧（可选）
    #    如果你已经提前抽好帧，可以注释掉下面这一行；
    #    若想由本脚本负责抽帧，则保留这一行。
    # videos_to_frames(dst_videos_dir, frames_root, fps=args.fps, debug=args.debug)

    # 3. 根据 splits 构建 train/val metadata
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


