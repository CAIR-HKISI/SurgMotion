import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ==== 路径配置 ====
BASE_VIDEO_PATH = "data/Landscopy/JIGSAWS"
FRAME_ROOT = "data/Surge_Frames/JIGSAWS/frames"
OUT_DIR = "data/Surge_Frames/JIGSAWS"

TASKS = ["Knot_Tying", "Needle_Passing", "Suturing"]
YEAR = 2025

# ==== 函数 ====

def parse_meta_file(meta_path):
    """读取 meta 文件，返回 {Case_Name: Skill_Score}"""
    meta_dict = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                case_name = parts[0].strip()  # 例如 Knot_Tying_B001
                try:
                    score = int(parts[2])  # 第3列是得分
                except ValueError:
                    score = -1
                meta_dict[case_name] = score
    return meta_dict


def split_dataset(case_list, seed=42):
    """根据样本数量安全划分 70/10/20"""
    if len(case_list) == 0:
        return [], [], []
    elif len(case_list) < 3:
        return case_list, [], []
    else:
        train, test = train_test_split(case_list, test_size=0.3, random_state=seed, shuffle=True)
        val, test = train_test_split(test, test_size=2/3, random_state=seed)
        return train, val, test


def map_case_to_meta(frame_case, meta_info):
    """根据 capture1 命名找到对应 meta name"""
    for k in meta_info.keys():
        if frame_case.startswith(k):
            return k
    return None


def generate_csv():
    all_data = []
    global_idx = 0

    for task in TASKS:
        print(f"\n=== Processing {task} ===")

        meta_path = os.path.join(BASE_VIDEO_PATH, task, f"meta_file_{task}.txt")
        if not os.path.exists(meta_path):
            print(f"⚠️ Meta file not found: {meta_path}")
            continue

        meta_info = parse_meta_file(meta_path)
        if not meta_info:
            print(f"⚠️ Meta file empty for {task}. Skipping...")
            continue

        # 帧目录中列出文件夹
        all_frame_dirs = [d for d in os.listdir(FRAME_ROOT)
                          if d.startswith(task) and os.path.isdir(os.path.join(FRAME_ROOT, d))]
        if not all_frame_dirs:
            print(f"⚠️ No frame folders found for {task}")
            continue

        print(f"📂 Found {len(all_frame_dirs)} frame folders under {FRAME_ROOT}")

        # 数据划分
        train_cases, val_cases, test_cases = split_dataset(all_frame_dirs)
        print(f"→ Train: {len(train_cases)}, Val: {len(val_cases)}, Test: {len(test_cases)}")

        for split_name, cases in zip(["train", "val", "test"], [train_cases, val_cases, test_cases]):
            for case in tqdm(cases, desc=f"{task} {split_name}"):
                frame_dir = os.path.join(FRAME_ROOT, case)

                # 匹配 meta 名
                meta_case = map_case_to_meta(case, meta_info)
                if meta_case is None:
                    print(f"⚠️ No meta entry found for {case}")
                    continue

                skill_score = meta_info.get(meta_case, -1)

                # 帧文件
                frame_files = sorted([
                    f for f in os.listdir(frame_dir)
                    if f.lower().endswith(('.png', '.jpg'))
                ])
                if not frame_files:
                    print(f"⚠️ No frames found in {frame_dir}")
                    continue

                # 提取 case_id
                digits = ''.join(filter(str.isdigit, meta_case))
                try:
                    case_id = int(digits)
                except ValueError:
                    case_id = 0

                for i, frame_fn in enumerate(frame_files):
                    frame_path = os.path.join(frame_dir, frame_fn)
                    data_item = {
                        "index": global_idx,
                        "DataName": "JIGSAWS",
                        "Year": YEAR,
                        "Case_Name": meta_case,  # 保持和 meta 文件一致
                        "Case_ID": case_id,
                        "Frame_Path": frame_path,
                        "Phase_GT": skill_score,  # 手术技能得分
                        "Phase_Name": task,       # 任务类型
                        "Split": split_name
                    }
                    all_data.append(data_item)
                    global_idx += 1

    # === 保存 CSV ===
    df = pd.DataFrame(all_data)
    print(f"\n✅ Total frames processed: {len(df)}")
    os.makedirs(OUT_DIR, exist_ok=True)

    for split in ["train", "val", "test"]:
        df_split = df[df["Split"] == split]
        if not df_split.empty:
            out_csv = os.path.join(OUT_DIR, f"{split}_metadata.csv")
            df_split.to_csv(out_csv, index=False)
            print(f"💾 Saved {len(df_split)} frames → {out_csv}")
        else:
            print(f"⚠️ No frames for split: {split}")


if __name__ == "__main__":
    generate_csv()

