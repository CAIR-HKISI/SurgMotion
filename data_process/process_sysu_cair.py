import pandas as pd
import numpy as np
import os
import cv2
import argparse
from pathlib import Path
from datetime import timedelta
import glob
import re
from tqdm import tqdm

def parse_time(time_str):
    """Parses hh:mm:ss string to total seconds."""
    if pd.isna(time_str):
        return None
    try:
        # Handle cases where time might be datetime object already
        if hasattr(time_str, 'hour'):
            return time_str.hour * 3600 + time_str.minute * 60 + time_str.second
        
        parts = str(time_str).split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2: # mm:ss
            return int(parts[0]) * 60 + int(parts[1])
    except Exception as e:
        print(f"Error parsing time {time_str}: {e}")
    return None

# ================= ROI Cropping Config =================
BINARY_THRESHOLD = 15
EXPANSION_SCALE = 1.2
# =======================================================

def get_roi_bbox(image):
    """
    Calculates the bounding box of the surgical area.
    Returns: (x, y, w, h) or None
    """
    if image is None:
        return None

    # 1. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Binarize
    _, thresh = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)

    # 3. Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # 4. Find largest contour
    max_contour = max(contours, key=cv2.contourArea)

    # 5. Filter small areas
    if cv2.contourArea(max_contour) < 1000:
        return None

    # 6. Get bbox
    x, y, w, h = cv2.boundingRect(max_contour)
    return (x, y, w, h)

def get_video_path(excel_path, video_dir):
    """Finds corresponding video file for an excel file."""
    # Excel: 20230713_092647.xlsx
    # Video: TV_CAM_设备_20230713_092647.avi
    base_name = os.path.basename(excel_path)
    timestamp = base_name.replace('.xlsx', '')
    
    # Search for video with this timestamp
    pattern = os.path.join(video_dir, f"*{timestamp}.avi")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    return None

def task1_extract_frames(label_dir, video_dir, output_root):
    print("=== Starting Task 1: Frame Extraction ===")
    
    # Output directories
    frames_root = os.path.join(output_root, "frames")
    os.makedirs(frames_root, exist_ok=True)
    
    metadata_list = []
    
    excel_files = sorted(glob.glob(os.path.join(label_dir, "*.xlsx")))
    print(f"Found {len(excel_files)} Excel files to process.")
    
    global_idx = 0
    case_id_counter = 0
    total_skipped_frames = 0
    
    for excel_path in tqdm(excel_files, desc="Task 1: Processing Cases"):
        video_path = get_video_path(excel_path, video_dir)
        if not video_path:
            # tqdm.write allows printing without breaking the progress bar
            tqdm.write(f"Warning: No video found for {excel_path}, skipping.")
            continue
            
        case_name = os.path.splitext(os.path.basename(excel_path))[0]
        # print(f"Processing Case: {case_name}") # Removed to reduce noise with tqdm
        
        # Read Excel
        try:
            # Header is at row 8 (0-indexed), data starts row 9
            df = pd.read_excel(excel_path, header=8)
        except Exception as e:
            tqdm.write(f"Error reading {excel_path}: {e}")
            continue
            
        # Filter relevant columns and rows
        # Columns: 阶段起始, 阶段结束, 阶段名字, 操作打分, 打分依据, 持续时间
        # Indices: 0, 1, 2, 3, 4, 5
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            tqdm.write(f"Could not open video {video_path}")
            continue
            
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        row_idx = 0
        for _, row in df.iterrows():
            # Skip invalid rows
            if pd.isna(row.iloc[2]) or row.iloc[2] == '阶段名字':
                continue
                
            start_time_str = row.iloc[0]
            end_time_str = row.iloc[1]
            phase_name_raw = str(row.iloc[2]).strip()
            score = row.iloc[3]
            basis = row.iloc[4]
            
            start_sec = parse_time(start_time_str)
            end_sec = parse_time(end_time_str)
            
            if start_sec is None or end_sec is None:
                continue
                
            # Extract Phase GT (number)
            try:
                phase_gt = int(phase_name_raw.split('.')[0])
            except:
                phase_gt = -1 # Or handle appropriately
            
            # Create clip directory
            clip_idx = row_idx
            clip_dir = os.path.join(frames_root, case_name, f"clip_{clip_idx:03d}")
            os.makedirs(clip_dir, exist_ok=True)
            
            # Frame extraction at 1 FPS
            # We want frames at t = start_sec, start_sec+1, ..., <= end_sec
            current_sec = start_sec
            frame_count_in_clip = 0
            skipped_in_clip = 0
            
            while current_sec <= end_sec:
                # Calculate frame index in video
                frame_idx = int(current_sec * video_fps)
                
                if frame_idx >= total_frames_video:
                    break
                    
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Apply ROI cropping logic from cut_sysy.py
                    bbox = get_roi_bbox(frame)
                    if bbox:
                        x, y, w, h = bbox
                        img_h, img_w = frame.shape[:2]
                        
                        center_x = x + w / 2
                        center_y = y + h / 2
                        
                        new_w = w * EXPANSION_SCALE
                        new_h = h * EXPANSION_SCALE
                        
                        new_x = center_x - new_w / 2
                        new_y = center_y - new_h / 2
                        
                        # Boundary protection
                        x1 = int(max(0, new_x))
                        y1 = int(max(0, new_y))
                        x2 = int(min(img_w, new_x + new_w))
                        y2 = int(min(img_h, new_y + new_h))
                        
                        # Crop
                        frame = frame[y1:y2, x1:x2]
                        
                        # Save
                        save_name = f"{frame_idx:08d}.jpg"
                        save_path = os.path.join(clip_dir, save_name)
                        
                        if cv2.imwrite(save_path, frame) and os.path.exists(save_path):
                            # Metadata
                            # Columns: Index,DataName,Year,Case_Name,Case_ID,Frame_Path,Phase_GT,Phase_Name,skill score, score name,Split
                            metadata_list.append({
                                'Index': global_idx,
                                'DataName': 'Private_SYSU_Brochiscopy_labeled',
                                'Year': 2023,
                                'Case_Name': case_name,
                                'Case_ID': case_id_counter,
                                'Frame_Path': save_path,
                                'Phase_GT': phase_gt,
                                'Phase_Name': phase_name_raw,
                                'skill score': score,
                                'score name': basis,
                                'Split': 'train' # Default
                            })
                            
                            global_idx += 1
                            frame_count_in_clip += 1
                        else:
                            tqdm.write(f"Error: Failed to write frame to {save_path}")
                    else:
                        # No valid ROI found (e.g. black frame), skip
                        skipped_in_clip += 1
                        total_skipped_frames += 1
                        tqdm.write(f"[Skip] No valid ROI at {case_name} clip_{clip_idx:03d} frame_{frame_idx} ({current_sec}s)")
                
                current_sec += 1
            
            row_idx += 1
            
        cap.release()
        case_id_counter += 1
        tqdm.write(f"  -> Case {case_name}: extracted {frame_count_in_clip} frames, skipped {skipped_in_clip} (Total valid: {global_idx}, Total skipped: {total_skipped_frames})")
        
    # Save frames_metadata.csv
    if metadata_list:
        meta_df = pd.DataFrame(metadata_list)
        # Reorder columns as requested
        cols = ['Index','DataName','Year','Case_Name','Case_ID','Frame_Path','Phase_GT','Phase_Name','skill score','score name','Split']
        # Ensure all cols exist
        for c in cols:
            if c not in meta_df.columns:
                meta_df[c] = None
        meta_df = meta_df[cols]
        
        output_csv = os.path.join(output_root, "frames_metadata.csv")
        meta_df.to_csv(output_csv, index=False)
        print(f"Task 1 Complete. Metadata saved to {output_csv}")
        return output_csv
    else:
        print("No metadata generated!")
        return None

def task2_generate_clips(frames_metadata_path, output_root, window_size=32, stride=1):
    print(f"=== Starting Task 2: Clip Generation (Window={window_size}) ===")
    
    df = pd.read_csv(frames_metadata_path)
    
    # Create label mapping (Phase_Name -> 0..N)
    # Use static mapping for consistency
    print(f"Using Static Phase Label Mapping: {PHASE_NAME_MAPPING}")
    phase_to_label = PHASE_NAME_MAPPING
    
    # Output dirs
    step_clips_dir = os.path.join(output_root, f"step_clips_infos_{window_size}f")
    os.makedirs(step_clips_dir, exist_ok=True)
    
    df['clip_identifier'] = df['Frame_Path'].apply(lambda x: os.path.dirname(x))
    
    grouped = df.groupby('clip_identifier')
    print(f"Found {len(grouped)} unique clips (steps) to process.")
    
    all_step_metadata = []
    global_clip_index = 0
    
    for clip_dir, group in tqdm(grouped, desc="Task 2: Generating Step Clips"):
        # Sort by frame index (filename)
        group = group.sort_values('Frame_Path').reset_index(drop=True)
        
        # Get common info
        case_id = group['Case_ID'].iloc[0]
        phase_gt = group['Phase_GT'].iloc[0]
        phase_name = group['Phase_Name'].iloc[0]
        
        # Filter out invalid phases (e.g. "11.  空白") or unmapped phases
        if phase_name not in phase_to_label:
            # print(f"Skipping clip with unknown/ignored phase: {phase_name}")
            continue

        # Clip idx from path
        clip_idx_str = os.path.basename(clip_dir) # clip_000
        try:
            clip_idx_val = int(clip_idx_str.split('_')[1])
        except:
            clip_idx_val = 0
            
        total_frames = len(group)
        frames_per_window = window_size
        
        start_idx = 0
        
        while start_idx < total_frames:
            end_idx = start_idx + 1 
            # Logic: Window ends at end_idx. Window = [end_idx - window_size : end_idx]
            
            current_ptr = start_idx
            end_ptr = current_ptr + 1
            window_start_ptr = end_ptr - frames_per_window
            
            # Prepare frames
            is_padded = False
            padding_len = 0
            
            if window_start_ptr < 0:
                # Need padding
                valid_frames = group.iloc[0:end_ptr]
                actual_len = len(valid_frames)
                padding_len = frames_per_window - actual_len
                
                # Pad with last frame available in the valid window
                padding_path = valid_frames.iloc[-1]['Frame_Path'] 
                frames_list = valid_frames['Frame_Path'].tolist() + [padding_path] * padding_len
                is_padded = True
            else:
                window_frames = group.iloc[window_start_ptr:end_ptr]
                frames_list = window_frames['Frame_Path'].tolist()
                actual_len = len(window_frames)
            
            # Save txt
            clip_name = f"case{case_id}_c{clip_idx_val:03d}_w{global_clip_index:06d}"
            txt_path = os.path.join(step_clips_dir, f"{clip_name}.txt")
            
            with open(txt_path, 'w') as f:
                for p in frames_list:
                    f.write(f"{os.path.relpath(p)}\n")
            
            # Metadata entry
            all_step_metadata.append({
                'Index': global_clip_index,
                'clip_path': txt_path,
                'label': phase_to_label.get(phase_name, -1),
                'label_name': phase_name,
                'case_id': case_id,
                'clip_idx': clip_idx_val,
                'start_frame': max(window_start_ptr, 0),
                'end_frame': end_ptr, 
                'actual_frames': actual_len,
                'padded_frames': padding_len,
                's_padded': is_padded
            })
            
            global_clip_index += 1
            start_idx += stride
            
    # Save step_all_metadata.csv
    step_df = pd.DataFrame(all_step_metadata)
    out_step_csv = os.path.join(output_root, "step_all_metadata.csv")
    step_df.to_csv(out_step_csv, index=False)
    print(f"Task 2 Complete. Metadata saved to {out_step_csv}")

# ================= Score Mapping Config =================
SCORE_KEYWORD_MAPPING = {
    '初学者': 0,
    '合格': 1,
    '熟练': 2,
    '专家': 3
}

# Generated based on data/Surge_Frames/Private_SYSU_Brochiscopy_labeled/step_all_metadata.csv
PHASE_LIST = [
    '1. 入镜',
    '2. 过声门',
    '3. 麻醉',
    '4. 气道观察',
    '5. 肺泡灌洗',
    '6. 直视下活检',
    '7. 超声引导活检',
    '8. 超声引导穿刺',
    '9. 止血',
    '10. 退镜',
    '11. 退镜'
]
PHASE_NAME_MAPPING = {name: i for i, name in enumerate(PHASE_LIST)}
# ====================================================

def map_score_to_label(score_str):
    """
    Maps score string to integer label.
    0: Novice (<=2分 or '初学者')
    1: Intermediate (3分 or '合格')
    2: Expert (>=4分 or '熟练'/'专家')
    """
    if pd.isna(score_str):
        return None
    s = str(score_str).strip()
    
    # Extract number
    match = re.search(r'(\d+)', s)
    if match:
        score = int(match.group(1))
        if score <= 2:
            return 0
        elif score == 3:
            return 1
        elif score >= 4:
            return 2
            
    # Fallback to text matching using dictionary
    for keyword, label in SCORE_KEYWORD_MAPPING.items():
        if keyword in s:
            return label
    
    return None

def task3_generate_skill_clips(frames_metadata_path, output_root):
    print("=== Starting Task 3: Skill Clip Generation ===")
    
    df = pd.read_csv(frames_metadata_path)
    
    # Output dirs
    skill_clips_dir = os.path.join(output_root, "skill_clip_infos")
    os.makedirs(skill_clips_dir, exist_ok=True)
    
    # Prepare identifier to group by clip (step)
    df['clip_identifier'] = df['Frame_Path'].apply(lambda x: os.path.dirname(x))
    grouped = df.groupby('clip_identifier')
    print(f"Found {len(grouped)} unique clips to process for skills.")
    
    all_skill_metadata = []
    global_idx = 0
    
    for clip_dir, group in tqdm(grouped, desc="Task 3: Generating Skill Clips"):
        # Sort frames
        group = group.sort_values('Frame_Path').reset_index(drop=True)
        
        # Common info
        case_id = group['Case_ID'].iloc[0]
        score_str = group['skill score'].iloc[0]
        
        skill_label = map_score_to_label(score_str)
        if skill_label is None:
            # print(f"Warning: Could not map score '{score_str}' to label. Skipping clip {clip_dir}")
            continue
            
        # Clip idx from path
        clip_idx_str = os.path.basename(clip_dir) # clip_000
        try:
            clip_idx_val = int(clip_idx_str.split('_')[1])
        except:
            clip_idx_val = 0
            
        frames_list = group['Frame_Path'].tolist()
        
        # Save txt for the WHOLE clip (step)
        clip_name = f"case{case_id}_c{clip_idx_val:03d}_skill"
        txt_path = os.path.join(skill_clips_dir, f"{clip_name}.txt")
        
        with open(txt_path, 'w') as f:
            for p in frames_list:
                f.write(f"{os.path.relpath(p)}\n")
                
        # Metadata entry
        all_skill_metadata.append({
            'Index': global_idx,
            'clip_path': txt_path,
            'label': skill_label,
            'label_name': score_str, 
            'case_id': case_id,
            'clip_idx': clip_idx_val,
            'frame_count': len(frames_list),
            'Split': group['Split'].iloc[0]
        })
        global_idx += 1
        
    # Save skill_all_metadata.csv
    skill_df = pd.DataFrame(all_skill_metadata)
    out_skill_csv = os.path.join(output_root, "skill_all_metadata.csv")
    skill_df.to_csv(out_skill_csv, index=False)
    print(f"Task 3 Complete. Metadata saved to {out_skill_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_dir', default='data/GI_Videos/SYSU-CAIR/label_2023_Guo')
    parser.add_argument('--video_dir', default='data/GI_Videos/SYSU-CAIR/brochiscopy2023')
    parser.add_argument('--output_root', default='data/Surge_Frames/Private_SYSU_Brochiscopy_labeled')
    parser.add_argument('--window_size', type=int, default=32)
    args = parser.parse_args()
    
    # Task 1
    # Check if frames_metadata.csv exists to avoid re-running extraction if not needed (for dev speed)
    frames_meta = os.path.join(args.output_root, "frames_metadata.csv")
    if os.path.exists(frames_meta):
        print(f"Found existing frames metadata at {frames_meta}, skipping extraction.")
        meta_csv = frames_meta
    else:
        meta_csv = task1_extract_frames(args.label_dir, args.video_dir, args.output_root)
    
    if meta_csv:
        # Task 2
        task2_generate_clips(meta_csv, args.output_root, window_size=args.window_size)
        # Task 3
        task3_generate_skill_clips(meta_csv, args.output_root)

if __name__ == '__main__':
    main()
