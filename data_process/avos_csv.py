import os
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ===============================
# 基础路径
# ===============================
FRAME_DIR = Path("data/Surge_Frames/AVOS/frames")
ANNO_FILE = Path("data/Open_surgery/AVOS/anno_meta.txt")
OUT_DIR = Path("data/Surge_Frames/AVOS")
CLIPS_INFO_DIR = OUT_DIR / "clips_info"  # 用来存放每个视频片段的帧列表 txt

# ===============================
# 参数（按“片段级别”划分 train/val/test）
# ===============================
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
MIN_SAMPLES_PER_SPLIT = 10  # 若某类在 train 或 test 中少于该阈值，则整体丢弃
EXCLUDE_LABEL_NAMES = {"Others"}  # 这些文本标签无论样本多少都直接丢弃


def parse_annotations(anno_file: Path):
    """读取标注文件 anno_meta.txt，返回 {clip_name(去掉.mp4): label_name}"""
    anno_dict = {}
    with anno_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            fn, label = line.split("\t")
            clip_name = fn.replace(".mp4", "")
            anno_dict[clip_name] = label
    print(f"✅ Parsed {len(anno_dict)} annotation items.")
    return anno_dict


def extract_case_id(clip_name: str) -> str:
    """
    从类似 '_3B8K5blJes_0_16' 的目录名提取 Case_ID。
    去掉前缀 '_'，然后取第一段字符串。
    """
    name = clip_name.lstrip("_")
    # AVOS 默认格式：<video_id>_<start>_<end>
    return name.split("_")[0]


def generate_txt_for_clip(clip_dir: Path, txt_path: Path) -> int:
    """
    为单个 clip 生成一个 txt，里面写入所有帧的路径（相对于项目根目录，如 data/...）。
    返回该 clip 的帧数。
    """
    frame_files = sorted(
        [p for p in clip_dir.iterdir() if p.is_file() and p.suffix.lower() in [".jpg", ".png"]],
        key=lambda p: p.name,
    )

    if not frame_files:
        return 0

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w") as f:
        for fp in frame_files:
            # 保留以 data/... 开头的路径，便于在项目根目录下直接读取
            f.write(str(fp).replace("\\", "/") + "\n")

    return len(frame_files)


def collect_clips(frame_dir: Path, anno_dict):
    """
    遍历 AVOS 帧文件夹，按【视频片段】构建一条样本：
    - 为每个 clip 生成一个 txt（帧列表）
    - metadata 中一行对应一个 clip，而不是一帧

    输出字段（与 SurgicalVideoDataset 兼容）：
    - Index      : clip 索引（从 0 开始）
    - clip_path  : txt 文件路径（例如 data/Surge_Frames/AVOS/clips_info/XXX.txt）
    - label_name : 文本标签（原始动作名）
    - case_id    : 病例/视频 id（从 clip 目录名中提取）
    """
    items = []
    index = 0

    for clip_name in tqdm(sorted(os.listdir(frame_dir)), desc="Collecting clip-level samples"):
        clip_path = frame_dir / clip_name
        if not clip_path.is_dir():
            continue

        normalized_name = clip_name.lstrip("_")
        label_name = anno_dict.get(normalized_name)
        if label_name is None:
            print(f"⚠️ {clip_name} not in annotation list, skip")
            continue

        # 1) 为该 clip 生成一个 txt，里面列出所有帧
        txt_path = CLIPS_INFO_DIR / f"{clip_name}.txt"
        num_frames = generate_txt_for_clip(clip_path, txt_path)
        if num_frames == 0:
            print(f"⚠️ {clip_name} has no frames, skip")
            continue

        # 2) 提取 case_id（可选，用于后续按病例统计/划分）
        case_id = extract_case_id(clip_name)

        items.append(
            {
                "Index": index,
                "clip_path": str(txt_path).replace("\\", "/"),
                "label_name": label_name,
                "case_id": case_id,
                "num_frames": num_frames,
            }
        )
        index += 1

    print(f"✅ Total {len(items)} clip-level samples collected.")
    return pd.DataFrame(items)


def label_to_id_map(df: pd.DataFrame):
    """生成 动作名称 → 整数ID 的映射，并打印出来。"""
    unique_labels = sorted(df["label_name"].unique())
    label2id = {name: idx for idx, name in enumerate(unique_labels)}
    print("🧩 Action label → ID mapping:")
    for name, idx in label2id.items():
        print(f"   {idx}: {name}")
    return label2id


def split_dataset(df: pd.DataFrame):
    """
    按 clip（而不是帧）做 train/val/test 划分；
    为了类别均衡，仍然是“每个类别内按比例切分”。
    """
    df["Split"] = ""
    for label_name in df["label_name"].unique():
        sub_df = df[df["label_name"] == label_name]
        idxs = list(sub_df.index)
        random.shuffle(idxs)

        n = len(idxs)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])

        df.loc[idxs[:n_train], "Split"] = "train"
        df.loc[idxs[n_train : n_train + n_val], "Split"] = "val"
        df.loc[idxs[n_train + n_val :], "Split"] = "test"

        print(
            f"label_name {label_name} → train:{n_train}, "
            f"val:{n_val}, test:{n - n_train - n_val}"
        )

    return df


def filter_rare_classes(df: pd.DataFrame, min_samples: int = MIN_SAMPLES_PER_SPLIT):
    """
    删除在 train 或 test 中样本过少（小于 min_samples）的类别：
    - 先基于 (label_name, Split) 统计每类在各 split 中的数量
    - 若某个 label 在 train 或 test 中任意一个 split 的数量 < min_samples，则从所有 split 中移除该 label
    """
    # 先直接丢弃显式指定要排除的标签（如 "Others"）
    if EXCLUDE_LABEL_NAMES:
        print(f"\n🚫 Explicitly excluding labels (by name): {sorted(EXCLUDE_LABEL_NAMES)}")
        df = df[~df["label_name"].isin(EXCLUDE_LABEL_NAMES)].copy()

    # 只统计当前已经分好 Split 的样本
    stats = df.groupby(["label_name", "Split"]).size().unstack(fill_value=0)

    labels_to_drop = []
    print("\n📊 Per-class sample stats (before filtering rare classes):")
    for label_name, row in stats.iterrows():
        n_train = int(row.get("train", 0))
        n_val = int(row.get("val", 0))
        n_test = int(row.get("test", 0))
        print(f"  label_name {label_name} -> train:{n_train}, val:{n_val}, test:{n_test}")

        # 根据需求：如果“训练或者测试样本只有个位数”，就删除该类
        if n_train < min_samples or n_test < min_samples:
            labels_to_drop.append(label_name)

    if labels_to_drop:
        print("\n⚠️ The following labels have too few samples (train or test < "
              f"{min_samples}) and will be removed from all splits:")
        for lname in labels_to_drop:
            row = stats.loc[lname]
            print(
                f"  label_name {lname} -> "
                f"train:{int(row.get('train', 0))}, "
                f"val:{int(row.get('val', 0))}, "
                f"test:{int(row.get('test', 0))}"
            )
        df = df[~df["label_name"].isin(labels_to_drop)].copy()
    else:
        print("\n✅ No labels need to be removed based on the current threshold.")

    # 重新编号 Index，保证 Index 从 0 连续增长，方便后续使用
    df = df.reset_index(drop=True)
    df["Index"] = df.index
    return df


def save_csv(df: pd.DataFrame):
    """
    保存为 3 个 CSV 文件（clip-level），文件名加上 _clip 后缀，
    避免和以前“逐帧标注”的 metadata 命名冲突：
    - data/Surge_Frames/AVOS/train_clip_metadata.csv
    - data/Surge_Frames/AVOS/val_clip_metadata.csv
    - data/Surge_Frames/AVOS/test_clip_metadata.csv
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        subset = df[df["Split"] == split]
        if subset.empty:
            continue
        out_path = OUT_DIR / f"{split}_clip_metadata.csv"
        subset.to_csv(out_path, index=False)
        print(f"💾 Saved {len(subset)} clip samples → {out_path}")


def main():
    if not FRAME_DIR.exists():
        print(f"❌ FRAME_DIR not found: {FRAME_DIR}")
        return
    if not ANNO_FILE.exists():
        print(f"❌ ANNO_FILE not found: {ANNO_FILE}")
        return

    anno_dict = parse_annotations(ANNO_FILE)
    df = collect_clips(FRAME_DIR, anno_dict)

    if df.empty:
        print("❌ Error: No clip-level samples collected. Check paths or folder names.")
        return

    # 先按照文本标签做 train/val/test 划分
    df = split_dataset(df)

    # 再根据各 split 中的样本数，删除在 train 或 test 中样本过少的类别
    df = filter_rare_classes(df)

    # 最后只对保留的类别重新编号，保证 label 连续从 0 开始
    label2id = label_to_id_map(df)
    df["label"] = df["label_name"].map(label2id)

    save_csv(df)


if __name__ == "__main__":
    main()