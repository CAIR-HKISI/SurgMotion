
"""
data/Surge_Frames/GynSurg_action/frames/ 下面有四个子文件夹:
Coagulation  NeedlePassing  Rest  SuctionIrrigation  Transection

每个子文件夹是一种手术动作，包含若干个“视频的抽帧”文件夹，每个视频是一个手术动作。

任务：
1. 为每个“视频帧文件夹”生成一个 txt，包含该视频所有帧的相对路径（相对 data/Surge_Frames/GynSurg_action）。
   txt 文件名为 <视频文件夹名>.txt，统一放在 clips_info 目录下。
2. 生成 metadata.csv，字段：【Index, clip_path, label, label_name】
   - Index      : 从 0 开始的 clip 索引
   - clip_path  : 对应 txt 文件相对 data/Surge_Frames/GynSurg_action 的路径，如 "clips_info/XXXX.txt"
   - label      : 整数类别 id（按动作子文件夹名称排序 0,1,2,...）
   - label_name : 动作子文件夹名称，例如 "Coagulation"
3. 按照 4-fold 划分 train/test，将“视频级别”的 clip 做划分；
   每个 fold 输出 train_metadata_fold{i}.csv 与 test_metadata_fold{i}.csv（i=0,1,2,3），放在 clips_info 下。
"""

import os
import argparse
import random
from pathlib import Path

import pandas as pd


def generate_txt_file(video_dir: Path, txt_path: Path, base_dir: Path) -> int:
    """
    为单个视频目录生成 txt，写入该视频所有帧的“相对 base_dir 的路径”。
    返回该视频包含的帧数。
    """
    # 只保留文件（一般是 .jpg/.png），按文件名排序
    frame_files = sorted(
        [p for p in video_dir.iterdir() if p.is_file()],
        key=lambda p: p.name,
    )

    with txt_path.open("w") as f:
        for frame_path in frame_files:
            rel_path = frame_path.relative_to(base_dir)
            f.write(str(rel_path).replace("\\", "/") + "\n")

    return len(frame_files)


def build_metadata(frames_root: Path, clips_info_dir: Path, base_dir: Path):
    """
    遍历 frames_root 目录下的动作子文件夹和视频文件夹：
    - 为每个视频生成一个 txt
    - 汇总 metadata 列表
    """
    metadata = []
    index = 0

    # 动作子文件夹作为类别
    action_dirs = sorted([d for d in frames_root.iterdir() if d.is_dir()], key=lambda p: p.name)

    for label_id, action_dir in enumerate(action_dirs):
        label_name = action_dir.name

        # 每个动作子目录下的“视频帧文件夹”
        video_dirs = sorted([d for d in action_dir.iterdir() if d.is_dir()], key=lambda p: p.name)

        for video_dir in video_dirs:
            video_name = video_dir.name

            # 为该视频生成 txt
            txt_path = clips_info_dir / f"{video_name}.txt"
            num_frames = generate_txt_file(video_dir, txt_path, base_dir=base_dir)
            if num_frames == 0:
                # 没有帧的 clip 直接跳过
                continue

            # clip_path 以 BASE_DIR 为根的相对路径
            clip_rel_path = txt_path.relative_to(base_dir)

            metadata.append(
                {
                    "Index": index,
                    "clip_path": str(clip_rel_path).replace("\\", "/"),
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
    """
    基于“视频级别（metadata 行）”做 4-fold 划分，按 label 做大致均匀分配。
    返回 folds: List[List[int]]，每个内部 list 存储该 fold 中样本在 metadata 中的索引。
    """
    from collections import defaultdict

    rng = random.Random(seed)
    label_to_indices = defaultdict(list)

    for idx, item in enumerate(metadata):
        label_to_indices[item["label"]].append(idx)

    folds = [[] for _ in range(4)]

    for label, indices in label_to_indices.items():
        rng.shuffle(indices)
        for i, m_idx in enumerate(indices):
            folds[i % 4].append(m_idx)

    return folds


def main():
    parser = argparse.ArgumentParser(description="GynSurg action dataset clip & metadata generation")
    parser.add_argument(
        "--name",
        type=str,
        default="GynSurg_action",
        help="数据集根目录名（位于 data/Surge_Frames 下，例如 GynSurg_action）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="4-fold 划分的随机种子",
    )
    args = parser.parse_args()

    # 目录配置
    base_dir = Path("data/Surge_Frames") / args.name  # data/Surge_Frames/GynSurg_action
    frames_dir = base_dir / "frames"
    clips_info_dir = base_dir / "clips_info"
    clips_info_dir.mkdir(parents=True, exist_ok=True)

    # 1. 遍历 frames，生成 txt & metadata 列表
    metadata = build_metadata(frames_dir, clips_info_dir, base_dir=base_dir)
    print(f"发现 {len(metadata)} 个视频 clip，将写入 metadata.csv 和 4-fold 划分。")

    # 2. 保存总的 metadata.csv
    metadata_csv_path = clips_info_dir / "metadata.csv"
    save_csv(metadata_csv_path, metadata)
    print(f"保存总 metadata 到: {metadata_csv_path}")

    # 3. 4-fold 划分（按视频级别）
    folds = make_4_folds(metadata, seed=args.seed)

    for i in range(4):
        test_indices = set(folds[i])
        train_indices = [j for k, fold in enumerate(folds) if k != i for j in fold]

        train_meta = [metadata[j] for j in train_indices]
        test_meta = [metadata[j] for j in test_indices]

        train_csv = clips_info_dir / f"train_metadata_fold{i}.csv"
        test_csv = clips_info_dir / f"test_metadata_fold{i}.csv"

        save_csv(train_csv, train_meta)
        save_csv(test_csv, test_meta)

        print(
            f"Fold {i}: train={len(train_meta)} clips, test={len(test_meta)} clips -> "
            f"{train_csv.name}, {test_csv.name}"
        )


if __name__ == "__main__":
    main()
