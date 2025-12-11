"""
基于 JIGSAWS 视频抽帧、生成帧列表与按 4-fold 划分的 CSV。

配置来源：
- 视频：data/Landscopy/JIGSAWS_CVPR2021/data/JIGSAWS/video_encoded
- 标签：data/Landscopy/JIGSAWS_CVPR2021/data/JIGSAWS/label (FT-score-*.npy)
- 划分：data/Landscopy/JIGSAWS_CVPR2021/Towards-Unified-Surgical-Skill-Assessment/metas/JIGSAWS_split_4_fold.npy

输出：
- 帧：data/Surge_Frames/JIGSAWS_v1/frames_{fps}/{video_id}/%06d.jpg
- 帧列表：data/Surge_Frames/JIGSAWS_v1/clip_{fps}f.txt  每行：<video_id> <abs_frames_dir>
- CSV（按折拆分 train/test）：data/Surge_Frames/JIGSAWS_v1/{split}_fold{fold}_{fps}f.csv
  列：Index, case_id, clip_path, label, label_name
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


# ===== 基本路径与配置 =====
VIDEO_DIR = Path("data/Landscopy/JIGSAWS_CVPR2021/data/JIGSAWS/video_encoded")
LABEL_DIR = Path("data/Landscopy/JIGSAWS_CVPR2021/data/JIGSAWS/label")
SPLIT_PATH = Path(
    "data/Landscopy/JIGSAWS_CVPR2021/Towards-Unified-Surgical-Skill-Assessment/metas/JIGSAWS_split_4_fold.npy"
)

FRAME_ROOT = Path("data/Surge_Frames/JIGSAWS_v2")
FPS_LIST = [1,5,15,20]

LABEL_KEY = "GRS"
LABEL_RANGE = (6, 30)  # 归一化区间


# ===== 工具函数 =====
def load_splits() -> Dict[str, List[List[str]]]:
    return np.load(SPLIT_PATH, allow_pickle=True).item()


def build_video_map() -> Dict[str, List[str]]:
    """
    将不带 capture 后缀的基础名映射到实际存在的 <video_id>（含 captureX）。
    例：Knot_Tying_B001 -> [Knot_Tying_B001_capture1, Knot_Tying_B001_capture2]
    """
    mapping: Dict[str, List[str]] = {}
    for mp4 in VIDEO_DIR.glob("*.mp4"):
        base = mp4.stem  # 带 capture 后缀
        if "_capture" in base:
            root = base.split("_capture")[0]
        else:
            root = base
        mapping.setdefault(root, []).append(base)
    return mapping


def get_all_videos(splits: Dict[str, List[List[str]]]) -> List[str]:
    vids: List[str] = []
    for fold_lists in splits.values():
        for fold in fold_lists:
            vids.extend(fold)
    return sorted(set(vids))


def expand_split_ids(split_ids: Iterable[str], video_map: Dict[str, List[str]]) -> List[str]:
    """将无 capture 的名字展开为实际存在的 video_id 列表。"""
    expanded: List[str] = []
    for name in split_ids:
        matches = video_map.get(name, [])
        if not matches:
            print(f"[WARN] no matched video for '{name}' (expecting capture*)")
        expanded.extend(matches)
    return expanded


def expand_splits(
    splits: Dict[str, List[List[str]]], video_map: Dict[str, List[str]]
) -> Dict[str, List[List[str]]]:
    """对每个折的列表做 capture 展开。"""
    new_splits: Dict[str, List[List[str]]] = {}
    for task, fold_lists in splits.items():
        new_fold_lists: List[List[str]] = []
        for fold in fold_lists:
            new_fold_lists.append(expand_split_ids(fold, video_map))
        new_splits[task] = new_fold_lists
    return new_splits


def parse_case_id(video_id: str) -> int:
    """从 video_id 中提取主体编号，如 Knot_Tying_B003_capture2 -> 3"""
    parts = video_id.split("_")
    if len(parts) < 2:
        return -1
    subject = parts[-2]
    m = re.search(r"(\\d+)", subject)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return -1
    return -1


def load_label(video_id: str) -> float | None:
    # 修改为模糊匹配，与参考脚本逻辑一致
    # 参考：score_file = [i for i in os.listdir(label_dir) if 'FT-score' in i and 'npy' in i and video in i]
    # 确保只匹配到一个文件
    candidates = [
        f for f in LABEL_DIR.iterdir()
        if f.suffix == ".npy" and "FT-score" in f.name and video_id in f.name
    ]

    if len(candidates) == 0:
        print(f"[WARN] label missing for {video_id} (pattern: FT-score...{video_id}...)")
        return None
    
    if len(candidates) > 1:
        # 如果匹配到多个，尝试找精确匹配
        exact = LABEL_DIR / f"FT-score-{video_id}.npy"
        if exact in candidates:
            npy = exact
        else:
            print(f"[WARN] multiple label files found for {video_id}: {[f.name for f in candidates]}, skipping")
            return None
    else:
        npy = candidates[0]

    try:
        obj = np.load(npy, allow_pickle=True).item()
    except Exception as e:
        print(f"[WARN] failed to load {npy}: {e}")
        return None

    if LABEL_KEY not in obj:
        print(f"[WARN] label key '{LABEL_KEY}' missing in {npy.name}")
        return None
    
    raw = float(obj[LABEL_KEY])
    lo, hi = LABEL_RANGE
    norm = (raw - lo) / (hi - lo)
    return norm


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def extract_frames(video_id: str, fps: int) -> None:
    src = VIDEO_DIR / f"{video_id}.mp4"
    if not src.exists():
        print(f"[WARN] video missing: {src}")
        return
    dst_dir = FRAME_ROOT / f"frames_{fps}" / video_id
    # 若已存在帧则跳过
    if dst_dir.exists() and any(dst_dir.iterdir()):
        return
    ensure_dir(dst_dir)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        str(dst_dir / "%06d.jpg"),
    ]
    print(f"[INFO] extracting {video_id} @ {fps} fps")
    subprocess.run(cmd, check=True)


def write_clip_txt(video_ids: Iterable[str], fps: int) -> None:
    # 修改：clip_info_{fps}fps 是文件夹，下面每个视频一个txt
    out_dir = FRAME_ROOT / f"clip_info_{fps}fps"
    ensure_dir(out_dir)

    for vid in video_ids:
        frames_dir = (FRAME_ROOT / f"frames_{fps}" / vid) # 移除 .resolve() 保持相对路径
        if frames_dir.exists() and any(frames_dir.iterdir()):
            # 存相对路径，相对于项目根目录
            frames = sorted([str(f) for f in frames_dir.glob("*.jpg")])
            if frames:
                txt_path = out_dir / f"{vid}.txt"
                txt_path.write_text("\n".join(frames), encoding="utf-8")
    
    print(f"[INFO] clip info txts saved in: {out_dir}")


def build_split_csv(
    fps: int,
    fold_idx: int,
    split_name: str,
    task_name: str,
    video_ids: Iterable[str],
    start_index: int = 0,
) -> Tuple[pd.DataFrame, int]:
    rows = []
    idx = start_index
    # 保持相对路径
    clip_info_dir = FRAME_ROOT / f"clip_info_{fps}fps"

    for vid in video_ids:
        # clip_path 指向该视频的帧路径列表文件，使用相对路径
        txt_path = clip_info_dir / f"{vid}.txt"
        
        if not txt_path.exists():
            print(f"[WARN] clip info txt not found for {vid} @ {fps}fps, skip")
            continue
            
        label = load_label(vid)
        if label is None:
            continue
        case_id = parse_case_id(vid)
        rows.append(
            {
                "Index": idx,
                "case_id": case_id,
                "clip_path": str(txt_path), # 存入 CSV 的也是相对路径
                "label": label,
                "label_name": LABEL_KEY,
            }
        )
        idx += 1

    if not rows:
        return pd.DataFrame(), idx

    df = pd.DataFrame(rows)
    # 文件名增加 task_name
    out_csv = FRAME_ROOT / f"{task_name}_{split_name}_fold{fold_idx}_{fps}f.csv"
    df.to_csv(out_csv, index=False)
    print(f"[INFO] saved {len(df)} rows -> {out_csv}")
    return df, idx


def main() -> None:
    raw_splits = load_splits()
    video_map = build_video_map()
    splits = expand_splits(raw_splits, video_map)
    all_videos = get_all_videos(splits)

    # 1) 抽帧
    for fps in FPS_LIST:
        for vid in all_videos:
            extract_frames(vid, fps)

    # 2) clip_{fps}f.txt
    for fps in FPS_LIST:
        write_clip_txt(all_videos, fps)

    # 3) 按折生成 CSV（train/test），每个 fps 一套
    for fps in FPS_LIST:
        for fold_idx in range(4):
            # 修改：按 task 分别生成 CSV
            for task_name, task_folds in splits.items():
                if fold_idx >= len(task_folds):
                    continue
                
                test_videos = sorted(set(task_folds[fold_idx]))
                train_videos = []
                for j, fold in enumerate(task_folds):
                    if j != fold_idx:
                        train_videos.extend(fold)
                train_videos = sorted(set(train_videos))

                build_split_csv(fps, fold_idx, "train", task_name, train_videos, start_index=0)
                build_split_csv(fps, fold_idx, "test", task_name, test_videos, start_index=0)


if __name__ == "__main__":
    main()

