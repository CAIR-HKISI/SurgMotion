import os
import random
import pandas as pd
from tqdm import tqdm

# ===============================
# 基础路径
# ===============================
FRAME_DIR = "data/Surge_Frames/AVOS/frames"
ANNO_FILE = "data/Open_surgery/AVOS/anno_meta.txt"
OUT_DIR = "data/Surge_Frames/AVOS"

# ===============================
# 参数
# ===============================
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def parse_annotations(anno_file):
    """读取标注文件 anno_meta.txt"""
    anno_dict = {}
    with open(anno_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            fn, label = line.split('\t')
            clip_name = fn.replace(".mp4", "")
            anno_dict[clip_name] = label
    print(f"✅ Parsed {len(anno_dict)} annotation items.")
    return anno_dict


def extract_case_id(clip_name):
    """
    从类似 '_3B8K5blJes_0_16' 的目录名提取 Case_ID。
    去掉前缀 '_'，然后取第一段字符串。
    """
    name = clip_name.lstrip("_")
    # AVOS 默认格式：<video_id>_<start>_<end>
    return name.split("_")[0]


def collect_clips(frame_dir, anno_dict):
    """遍历帧文件夹并匹配标注"""
    all_items = []
    global_idx = 0

    for clip_name in tqdm(sorted(os.listdir(frame_dir)), desc="Collecting clips"):
        clip_path = os.path.join(frame_dir, clip_name)
        if not os.path.isdir(clip_path):
            continue

        normalized_name = clip_name.lstrip("_")
        label = anno_dict.get(normalized_name)
        if label is None:
            print(f"⚠️ {clip_name} not in annotation list")
            continue

        frame_files = sorted([
            os.path.join(clip_path, f)
            for f in os.listdir(clip_path)
            if f.endswith((".jpg", ".png"))
        ])
        if not frame_files:
            continue

        case_id = extract_case_id(clip_name)

        for frame_file in frame_files:
            all_items.append({
                "index": global_idx,
                "DataName": "AVOS",
                "Year": 2022,
                "Case_ID": case_id,
                "Case_Name": clip_name,
                "Frame_Path": frame_file,
                "Phase_Name": label,
            })
            global_idx += 1

    print(f"✅ Total {len(all_items)} frames collected.")
    return pd.DataFrame(all_items)


def label_to_id_map(df):
    """生成动作名称到ID的映射"""
    unique_labels = sorted(df["Phase_Name"].unique())
    phase2id = {name: idx for idx, name in enumerate(unique_labels)}
    print("🧩 Phase → ID mapping:")
    for k, v in phase2id.items():
        print(f"   {v}: {k}")
    return phase2id


def split_dataset(df):
    """每个动作类别按比例划分 train/val/test"""
    df["Split"] = ""
    for label in df["Phase_Name"].unique():
        sub_df = df[df["Phase_Name"] == label]
        idxs = list(sub_df.index)
        random.shuffle(idxs)

        n = len(idxs)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])

        df.loc[idxs[:n_train], "Split"] = "train"
        df.loc[idxs[n_train:n_train + n_val], "Split"] = "val"
        df.loc[idxs[n_train + n_val:], "Split"] = "test"

        print(f"{label:20s} → train:{n_train}, val:{n_val}, test:{n - n_train - n_val}")

    return df


def save_csv(df):
    """保存为3个CSV文件"""
    for split in ["train", "val", "test"]:
        subset = df[df["Split"] == split]
        if subset.empty:
            continue
        out_path = os.path.join(OUT_DIR, f"{split}_metadata.csv")
        subset.to_csv(out_path, index=False)
        print(f"💾 Saved {len(subset)} samples → {out_path}")


def main():
    anno_dict = parse_annotations(ANNO_FILE)
    df = collect_clips(FRAME_DIR, anno_dict)

    if df.empty:
        print("❌ Error: No frames collected. Check paths or folder names.")
        return

    phase2id = label_to_id_map(df)
    df["Phase_GT"] = df["Phase_Name"].map(phase2id)

    df = split_dataset(df)
    save_csv(df)


if __name__ == "__main__":
    main()