
"""
{
"video41": [{'unique_id': 0, 
  'frame_id': 0.0, 
  'video_id': 'video41', 
  'tool_gt': [1, 0, 0, 0, 0, 0, 0], 
  'phase_gt': 0, 
  'phase_name': 'Preparation', 
  'fps': 1, 
  'original_frames': 77576, 
  'frames': 3103.0}]
}
"""

import pickle
import pandas as pd
import os

test_label = "data/Cholec80/labels/test/1fpsval_test.pickle"
train_label = "data/Cholec80/labels/train/1fpstrain.pickle"

path_dict = {
    'test': test_label,
    'train': train_label
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
            case_id = int(vid[-2:])
            case_name = vid
        except Exception:
            print(f"Warning: Video id {vid} is not an integer.")
            continue

        for value in values:
            frame_id = int(value["frame_id"])
            frame_path = f'data/Surge_Frames/Cholec80/frames_cutmargin/{case_name}/{frame_id:05d}.png'
            if not os.path.exists(frame_path):
                print(f"Warning: Frame {frame_path} does not exist")
                continue

            data_item = {
                'index': global_idx,
                'Hospital': 'Cholec80',
                'Year': 2016,
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
        out_csv = f'data/Surge_Frames/Cholec80/{split}_metadata.csv'
        df.to_csv(out_csv, index=False)
        print(f"Saved {len(df)} items to {out_csv}")
    else:
        print(f"No data for split {split}, csv not saved.")
 
 