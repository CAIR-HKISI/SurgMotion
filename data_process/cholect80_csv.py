
# """
# {
# "video41": [{'unique_id': 0, 
#   'frame_id': 0.0, 
#   'video_id': 'video41', 
#   'tool_gt': [1, 0, 0, 0, 0, 0, 0], 
#   'phase_gt': 0, 
#   'phase_name': 'Preparation', 
#   'fps': 1, 
#   'original_frames': 77576, 
#   'frames': 3103.0}]
# }
# """

# import pickle
# import pandas as pd
# import os

# test_label = "data/Cholec80/labels/test/1fpsval_test.pickle"
# train_label = "data/Cholec80/labels/train/1fpstrain.pickle"

# path_dict = {
#     'test': test_label,
#     'train': train_label
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
#             case_id = int(vid[-2:])
#             case_name = vid
#         except Exception:
#             print(f"Warning: Video id {vid} is not an integer.")
#             continue

#         for value in values:
#             frame_id = int(value["frame_id"])
#             frame_path = f'data/Surge_Frames/Cholec80/frames_cutmargin/{case_name}/{frame_id:05d}.png'
#             if not os.path.exists(frame_path):
#                 print(f"Warning: Frame {frame_path} does not exist")
#                 continue

#             data_item = {
#                 'index': global_idx,
#                 'Hospital': 'Cholec80',
#                 'Year': 2016,
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
#         out_csv = f'data/Surge_Frames/Cholec80/{split}_metadata.csv'
#         df.to_csv(out_csv, index=False)
#         print(f"Saved {len(df)} items to {out_csv}")
#     else:
#         print(f"No data for split {split}, csv not saved.")
 

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# 配置参数
FRAME_PATH = "data/Surge_Frames/Cholec80/frames"
ANNOT_PATH = "data/Landscopy/cholec80/phase_annotations"
OUT_DIR = "data/Surge_Frames/Cholec80"

# 数据划分
TRAIN_NUMBERS = np.arange(1, 41).tolist()
TEST_NUMBERS = np.arange(41, 81).tolist()

# 阶段类别映射
phase2id = {
    'Preparation': 0,
    'CalotTriangleDissection': 1,
    'ClippingCutting': 2,
    'GallbladderDissection': 3,
    'GallbladderPackaging': 4,
    'CleaningCoagulation': 5,
    'GallbladderRetraction': 6
}

def get_split(case_id):
    """判断属于 train 还是 test"""
    if case_id in TRAIN_NUMBERS:
        return "train"
    elif case_id in TEST_NUMBERS:
        return "test"
    else:
        return None

def generate_csv():
    all_data = []
    global_idx = 0
    
    # 遍历每个视频的标注文件
    for annot_file in tqdm(sorted(os.listdir(ANNOT_PATH)), desc="Processing Cholec80 annotations"):
        if not annot_file.endswith(".txt"):
            continue

        case_name = annot_file.replace("-phase.txt", "")  # e.g., video01
        case_id = int(case_name.replace("video", ""))

        split = get_split(case_id)
        if split is None:
            print(f"⚠️ Skipped {case_name}, out of train/test range.")
            continue

        fps_orig = 25
        fps_target = 1
        step = fps_orig // fps_target  # 每隔25帧取一个

        phase_file = os.path.join(ANNOT_PATH, annot_file)
        df_phase = pd.read_csv(phase_file, sep='\t')

        frames_dir = os.path.join(FRAME_PATH, case_name)
        if not os.path.exists(frames_dir):
            print(f"⚠️ Warning: Frame directory not found for {case_name}")
            continue

        # 对应1fps抽帧图像（按1秒间隔）
        frame_files = sorted(
            [f for f in os.listdir(frames_dir) if f.endswith(".jpg")]
        )
        
        for i, frame_file in enumerate(frame_files):
            frame_id = i  # 对应1fps帧编号
            label_index = min(frame_id * step, len(df_phase) - 1)  # 映射回原标注帧
            phase_name = df_phase.iloc[label_index]["Phase"]
            phase_gt = phase2id.get(phase_name, -1)

            frame_path = os.path.join(frames_dir, frame_file)

            data_item = {
                'index': global_idx,
                'DataName': 'Cholec80',
                'Year': 2016,
                'Case_Name': case_name,
                'Case_ID': case_id,
                'Frame_Path': frame_path,
                'Phase_GT': phase_gt,
                'Phase_Name': phase_name,
                'Split': split
            }
            all_data.append(data_item)
            global_idx += 1

    # 转为DataFrame并分别保存
    df = pd.DataFrame(all_data)
    print(f"✅ Total frames processed: {len(df)}")

    for split in ["train", "test"]:
        df_split = df[df["Split"] == split]
        if not df_split.empty:
            out_csv = os.path.join(OUT_DIR, f"{split}_metadata.csv")
            df_split.to_csv(out_csv, index=False)
            print(f"💾 Saved {len(df_split)} frames to {out_csv}")
        else:
            print(f"⚠️ No data found for split: {split}")

if __name__ == "__main__":
    generate_csv()