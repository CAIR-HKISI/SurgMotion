import os
import re
import pandas as pd
from tqdm import tqdm

# ========================================
# 基础路径配置
# ========================================
FRAME_PATH = "data/Surge_Frames/CATARACTS/frames"
GT_BASE_PATH = "data/Ophthalmology/CATARACTS/ground_truth/CATARACTS_2020"
OUT_DIR = "data/Surge_Frames/CATARACTS"

# 三个split的标注路径
split_dict = {
    "train": os.path.join(GT_BASE_PATH, "train_gt"),
    "val": os.path.join(GT_BASE_PATH, "dev_gt"),
    "test": os.path.join(GT_BASE_PATH, "test_gt"),
}

# 抽帧比例
FPS_ORIG = 30
FPS_TARGET = 1
STEP = FPS_ORIG // FPS_TARGET

# ================================
# Phase名称映射表
# ================================
step_names = [
    'Toric Marking', 'Implant Ejection', 'Incision', 'Viscodilatation', 'Capsulorhexis',
    'Hydrodissetion', 'Nucleus Breaking', 'Phacoemulsification', 'Vitrectomy',
    'Irrigation/Aspiration', 'Preparing Implant', 'Manual Aspiration',
    'Implantation', 'Positioning', 'OVD Aspiration', 'Suturing',
    'Sealing Control', 'Wound Hydratation'
]

def parse_case_name(filename: str):
    """
    从CSV标注文件名提取视频名(case_name)和编号(case_id)
    支持 train01.csv, train_01.csv, test06.csv 等。
    """
    base = os.path.splitext(filename)[0]
    m = re.match(r"([a-zA-Z]+)_?(\d+)$", base)
    if m:
        prefix, num = m.groups()
        return f"{prefix}{int(num):02d}", int(num)
    return None, None


def generate_csv():
    all_data = []
    global_idx = 0

    for split, gt_path in split_dict.items():
        print(f"\n=== Processing split: {split.upper()} ===")

        if not os.path.exists(gt_path):
            print(f"⚠️ Ground truth directory not found: {gt_path}")
            continue

        csv_files = sorted([f for f in os.listdir(gt_path) if f.endswith(".csv")])
        if not csv_files:
            print(f"⚠️ No annotation CSV files found in {gt_path}")
            continue

        for gt_file in tqdm(csv_files, desc=f"{split} videos"):
            case_name, case_id = parse_case_name(gt_file)
            if not case_name:
                print(f"⚠️ Unexpected filename format: {gt_file}")
                continue

            gt_csv_path = os.path.join(gt_path, gt_file)
            try:
                df_gt = pd.read_csv(gt_csv_path)
            except Exception as e:
                print(f"⚠️ Failed to read {gt_file}: {e}")
                continue

            if not {"Frame", "Steps"}.issubset(df_gt.columns):
                print(f"⚠️ Invalid structure in {gt_file} (missing Frame/Steps)")
                continue

            # 1️⃣ 检查帧文件目录是否存在
            frame_dir = os.path.join(FRAME_PATH, case_name)
            if not os.path.exists(frame_dir):
                print(f"⚠️ Missing frame directory: {frame_dir}")
                continue

            frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])
            if not frame_files:
                print(f"⚠️ No frame images found in {frame_dir}")
                continue

            # 2️⃣ 遍历所有帧
            for i, frame_file in enumerate(frame_files):
                frame_path = os.path.join(frame_dir, frame_file)

                # 检查物理文件存在
                if not os.path.exists(frame_path):
                    print(f"⚠️ Missing frame file: {frame_path}")
                    continue

                label_index = min(i * STEP, len(df_gt) - 1)
                try:
                    step_index = int(df_gt.iloc[label_index]["Steps"])
                except Exception:
                    print(f"⚠️ Invalid Steps index in {gt_file}, row {label_index}")
                    continue

                # 获取 phase 名称（若越界则标记 Unknown）
                if 0 <= step_index < len(step_names):
                    phase_name = step_names[step_index]
                else:
                    phase_name = f"Unknown_{step_index}"

                all_data.append({
                    "index": global_idx,
                    "DataName": "CATARACTS",
                    "Year": 2020,
                    "Case_Name": case_name,
                    "Case_ID": case_id,
                    "Frame_Path": frame_path,
                    "Phase_GT": step_index,
                    "Phase_Name": phase_name,
                    "Split": split,
                })
                global_idx += 1

    # ===============================
    # 保存结果
    # ===============================
    if not all_data:
        print("\n⚠️ No valid frames found, nothing to save.")
        return

    df = pd.DataFrame(all_data)
    print(f"\n✅ Total valid frames: {len(df)}")
    os.makedirs(OUT_DIR, exist_ok=True)

    for split in ["train", "val", "test"]:
        df_split = df[df["Split"] == split]
        if not df_split.empty:
            out_path = os.path.join(OUT_DIR, f"{split}_metadata.csv")
            df_split.to_csv(out_path, index=False)
            print(f"💾 Saved {len(df_split)} frames -> {out_path}")
        else:
            print(f"⚠️ No data in split {split}")

if __name__ == "__main__":
    generate_csv()

