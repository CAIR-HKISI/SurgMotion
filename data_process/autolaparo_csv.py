
# """
# '01': [{'unique_id': 0,
#    'frame_id': 0,
#    'original_frame_id': 0,
#    'video_id': '01',
#    'tool_gt': None,
#    'frames': 6388,
#    'phase_gt': 1,
#    'phase_name': 'Dividing Ligament and Peritoneum',
#    'fps': 1}
# """

# import pickle
# import pandas as pd
# import os

# test_label = "/scratch/esg8sdce/wjl/Surgformer/data/autolaparo/task1/labels_pkl/test/1fpstest.pickle"
# train_label = "/scratch/esg8sdce/wjl/Surgformer/data/autolaparo/task1/labels_pkl/train/1fpstrain.pickle"
# val_label = "/scratch/esg8sdce/wjl/Surgformer/data/autolaparo/task1/labels_pkl/val/1fpsval.pickle"

# path_dict = {
#     'test': test_label,
#     'train': train_label,
#     'val': val_label
# }

# for split, path in path_dict.items():
#     print(f"Processing {split}: {path}")
#     if not os.path.exists(path):
#         print(f"Error: Label file {path} does not exist.")
#         continue

#     data = pickle.load(open(path, "rb"))
#     all_data = []
#     global_idx = 0

#     for vid, values in data.items():
#         # Convert video id to int, and format as 2-digit string
#         try:
#             case_id = int(vid)
#             case_name = f"{case_id:02d}"
#         except Exception:
#             print(f"Warning: Video id {vid} is not an integer.")
#             continue

#         for value in values:
#             frame_id = value["frame_id"] + 1
#             frame_path = f'data/Surge_Frames/AutoLaparo_v2/frames/{case_name}/{case_name}_{frame_id:08d}.jpg'
#             if not os.path.exists(frame_path):
#                 print(f"Warning: Frame {frame_path} does not exist")
#                 continue

#             data_item = {
#                 'index': global_idx,
#                 'Hospital': 'AutoLaparo',
#                 'Year': 2021,
#                 'Case_Name': case_name,
#                 'Case_ID': case_id,
#                 'Frame_Path': frame_path,
#                 'Phase_GT': value.get("phase_gt"),
#                 'Phase_Name': value.get("phase_name"),
#                 'Split': split
#             }
#             all_data.append(data_item)
#             global_idx += 1

#     if all_data:
#         df = pd.DataFrame(all_data)
#         out_csv = f'data/Surge_Frames/AutoLaparo_v2/{split}_metadata.csv'
#         df.to_csv(out_csv, index=False)
#         print(f"Saved {len(df)} items to {out_csv}")
#     else:
#         print(f"No data for split {split}, csv not saved.")

import os
import numpy as np
import pandas as pd
from pathlib import Path

# -------------------- 路径配置 --------------------
frames_root = Path("data/Surge_Frames/AutoLaparo/frames")
labels_root = Path("data/Landscopy/autolaparo/task1/labels")
output_root = Path("data/Surge_Frames/AutoLaparo")
output_root.mkdir(exist_ok=True, parents=True)

# -------------------- 固定划分 --------------------
TRAIN_NUMBERS = np.arange(1, 11).tolist()   # 1–10
VAL_NUMBERS   = np.arange(11, 15).tolist()  # 11–14
TEST_NUMBERS  = np.arange(15, 22).tolist()  # 15–21

# -------------------- 参数 --------------------
fps = 1
data_name = "AutoLaparo"
year = 2021

# phase编号映射表（转换为0–6）
phase_map = {
    0: "Preparation",
    1: "Dividing Ligament and Peritoneum",
    2: "Dividing Uterine Vessels and Ligament",
    3: "Transecting the Vagina",
    4: "Specimen Removal",
    5: "Suturing",
    6: "Washing"
}

# -------------------- 准备数据 --------------------
all_data = []
global_idx = 0

for label_file in sorted(labels_root.glob("label_*.txt")):
    case_id = int(label_file.stem.split("_")[-1])
    case_name = f"{case_id:02d}"

    frames_folder = frames_root / case_name
    if not frames_folder.exists():
        print(f"⚠️ Frames folder not found for case {case_name}")
        continue

    # 读取标签文件
    df_label = pd.read_csv(label_file, sep=r'\s+', engine="python")

    for _, row in df_label.iterrows():
        frame_num = int(row["Frame"])
        # 把原始phase从1–7映射到0–6
        phase_gt = int(row["Phase"]) - 1
        if phase_gt not in phase_map:
            print(f"⚠️ Unknown phase {phase_gt} in {label_file}")
            continue

        frame_name = f"{case_name}_{frame_num:08d}.jpg"
        frame_path = frames_folder / frame_name
        if not frame_path.exists():
            print(f"⚠️ Frame not found: {frame_path}")
            continue

        data_item = {
            "index": global_idx,
            "DataName": data_name,
            "Year": year,
            "Case_Name": case_name,
            "Case_ID": case_id,
            "Frame_Path": str(frame_path),
            "Phase_GT": phase_gt,
            "Phase_Name": phase_map[phase_gt]
        }
        all_data.append(data_item)
        global_idx += 1

print(f"✅ Total labeled frames collected: {len(all_data)}")

# -------------------- 构建DataFrame --------------------
df_all = pd.DataFrame(all_data)

# -------------------- 按视频编号划分 --------------------
def assign_split(case_id: int):
    if case_id in TRAIN_NUMBERS:
        return "train"
    elif case_id in VAL_NUMBERS:
        return "val"
    elif case_id in TEST_NUMBERS:
        return "test"
    else:
        return "ignore"

df_all["Split"] = df_all["Case_ID"].apply(assign_split)
df_all = df_all[df_all["Split"] != "ignore"]

# -------------------- 保存CSV --------------------
for split_name in ["train", "val", "test"]:
    split_df = df_all[df_all["Split"] == split_name]
    if not split_df.empty:
        out_csv = output_root / f"{split_name}_metadata.csv"
        split_df.to_csv(out_csv, index=False)
        print(f"💾 Saved {len(split_df)} records to {out_csv}")
    else:
        print(f"⚠️ No data for {split_name}")

print("🎉 All done!")

