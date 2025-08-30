
"""
'01': [{'unique_id': 0,
   'frame_id': 0,
   'original_frame_id': 0,
   'video_id': '01',
   'tool_gt': None,
   'frames': 6388,
   'phase_gt': 1,
   'phase_name': 'Dividing Ligament and Peritoneum',
   'fps': 1}
"""

import pickle
import pandas as pd
import os

test_label = "/scratch/esg8sdce/wjl/Surgformer/data/autolaparo/task1/labels_pkl/test/1fpstest.pickle"
train_label = "/scratch/esg8sdce/wjl/Surgformer/data/autolaparo/task1/labels_pkl/train/1fpstrain.pickle"
val_label = "/scratch/esg8sdce/wjl/Surgformer/data/autolaparo/task1/labels_pkl/val/1fpsval.pickle"

path_dict = {
    'test': test_label,
    'train': train_label,
    'val': val_label
}

for split, path in path_dict.items():
    print(f"Processing {split}: {path}")
    if not os.path.exists(path):
        print(f"Error: Label file {path} does not exist.")
        continue

    data = pickle.load(open(path, "rb"))
    all_data = []
    global_idx = 0

    for vid, values in data.items():
        # Convert video id to int, and format as 2-digit string
        try:
            case_id = int(vid)
            case_name = f"{case_id:02d}"
        except Exception:
            print(f"Warning: Video id {vid} is not an integer.")
            continue

        for value in values:
            frame_id = value["frame_id"]
            frame_path = f'data/Surge_Frames/AutoLaparo/frames/{case_name}/{frame_id:05d}.png'
            if not os.path.exists(frame_path):
                print(f"Warning: Frame {frame_path} does not exist")
                continue

            data_item = {
                'index': global_idx,
                'Hospital': 'AutoLaparo',
                'Year': 2021,
                'Case_Name': case_name,
                'Case_ID': case_id,
                'Frame_Path': frame_path,
                'Phase_GT': value.get("phase_gt"),
                'Phase_Name': value.get("phase_name"),
                'Split': split
            }
            all_data.append(data_item)
            global_idx += 1

    if all_data:
        df = pd.DataFrame(all_data)
        out_csv = f'data/Surge_Frames/AutoLaparo/{split}_metadata.csv'
        df.to_csv(out_csv, index=False)
        print(f"Saved {len(df)} items to {out_csv}")
    else:
        print(f"No data for split {split}, csv not saved.")