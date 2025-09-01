#!/usr/bin/env python3
"""
光流分析演示版本 - 修复版本
"""

import pandas as pd
import numpy as np
import cv2
import os
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

def read_pitvis_csv(csv_path):
    """读取pitvis数据集的csv文件"""
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"CSV contains {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    return df

def group_frames_by_video(df):
    """按视频分组帧"""
    print("Grouping frames by video...")
    
    video_groups = defaultdict(list)
    
    for _, row in df.iterrows():
        video_name = row['Case_Name']
        frame_path = row['Frame_Path']
        phase_gt = row['Phase_GT']
        phase_name = row['Phase_Name']
        
        frame_filename = os.path.basename(frame_path)
        frame_number = int(frame_filename.split('_')[-1].split('.')[0])
        
        video_groups[video_name].append({
            'frame_path': frame_path,
            'frame_number': frame_number,
            'phase_gt': phase_gt,
            'phase_name': phase_name
        })
    
    for video_name in video_groups:
        video_groups[video_name].sort(key=lambda x: x['frame_number'])
    
    print(f"Found {len(video_groups)} videos")
    for video_name, frames in video_groups.items():
        print(f"  {video_name}: {len(frames)} frames")
    
    return video_groups

def calculate_optical_flow_magnitude(frame1, frame2):
    """计算两帧之间的光流幅值"""
    if len(frame1.shape) == 3:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = frame1
        
    if len(frame2.shape) == 3:
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = frame2
    
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    
    return magnitude, flow

def process_video_optical_flow(video_frames, frames_dir):
    """处理单个视频的光流计算"""
    print(f"Processing video with {len(video_frames)} frames")
    
    flow_magnitudes = []
    
    for i in tqdm(range(len(video_frames) - 1), desc="Calculating optical flow"):
        frame1_path = video_frames[i]['frame_path']
        frame2_path = video_frames[i + 1]['frame_path']
        
        frame1 = cv2.imread(frame1_path)
        frame2 = cv2.imread(frame2_path)
        
        if frame1 is None or frame2 is None:
            print(f"Warning: Cannot read frame {frame1_path} or {frame2_path}")
            continue
        
        magnitude, flow = calculate_optical_flow_magnitude(frame1, frame2)
        
        mean_magnitude = np.mean(magnitude)
        std_magnitude = np.std(magnitude)
        max_magnitude = np.max(magnitude)
        min_magnitude = np.min(magnitude)
        
        flow_magnitudes.append({
            'frame_pair': (i, i + 1),
            'frame_numbers': (video_frames[i]['frame_number'], video_frames[i + 1]['frame_number']),
            'mean_magnitude': mean_magnitude,
            'std_magnitude': std_magnitude,
            'max_magnitude': max_magnitude,
            'min_magnitude': min_magnitude,
            'phase_gt': video_frames[i]['phase_gt'],
            'phase_name': video_frames[i]['phase_name']
        })
    
    return flow_magnitudes

def visualize_flow_statistics(all_flow_stats, output_dir="flow_analysis_demo_fixed"):
    """可视化光流统计信息"""
    os.makedirs(output_dir, exist_ok=True)
    
    all_magnitudes = []
    all_phases = []
    
    for video_name, stats in all_flow_stats.items():
        for flow_stat in stats['flow_magnitudes']:
            all_magnitudes.append(flow_stat['mean_magnitude'])
            all_phases.append(flow_stat['phase_gt'])
    
    # Set English font
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(12, 8))
    
    # Overall distribution
    plt.subplot(2, 2, 1)
    plt.hist(all_magnitudes, bins=50, alpha=0.7, color='blue')
    plt.xlabel('Average Optical Flow Magnitude')
    plt.ylabel('Frequency')
    plt.title('Optical Flow Magnitude Distribution')
    plt.grid(True, alpha=0.3)
    
    # By phase
    plt.subplot(2, 2, 2)
    phase_magnitudes = defaultdict(list)
    for mag, phase in zip(all_magnitudes, all_phases):
        phase_magnitudes[phase].append(mag)
    
    for phase, magnitudes in phase_magnitudes.items():
        plt.hist(magnitudes, bins=30, alpha=0.6, label=f'Phase {phase}')
    
    plt.xlabel('Average Optical Flow Magnitude')
    plt.ylabel('Frequency')
    plt.title('Optical Flow Magnitude by Phase')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Trends by video
    plt.subplot(2, 2, 3)
    for video_name, stats in list(all_flow_stats.items())[:5]:
        magnitudes = [s['mean_magnitude'] for s in stats['flow_magnitudes']]
        plt.plot(magnitudes, label=video_name, alpha=0.7)
    
    plt.xlabel('Frame Pair Index')
    plt.ylabel('Average Optical Flow Magnitude')
    plt.title('Optical Flow Magnitude Trends by Video')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Statistics
    plt.subplot(2, 2, 4)
    stats_text = f"""
    Total Frame Pairs: {len(all_magnitudes)}
    Mean Magnitude: {np.mean(all_magnitudes):.3f}
    Std Deviation: {np.std(all_magnitudes):.3f}
    Max Magnitude: {np.max(all_magnitudes):.3f}
    Min Magnitude: {np.min(all_magnitudes):.3f}
    """
    plt.text(0.1, 0.5, stats_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    plt.axis('off')
    plt.title('Statistical Summary')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'flow_statistics.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Results saved to {output_dir} directory")

def demo_optical_flow_analysis(max_videos=2, max_frames_per_video=50):
    """演示光流分析功能，只处理少量数据"""
    print("=" * 60)
    print("Optical Flow Analysis Demo (Fixed Version)")
    print("=" * 60)
    
    csv_path = "data/Surge_Frames/Pitvis/train_metadata.csv"
    frames_dir = "data/Surge_Frames/Pitvis/frames"
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    if not os.path.exists(frames_dir):
        print(f"Error: Frames directory not found: {frames_dir}")
        return
    
    # 1. Read CSV data
    print("1. Reading CSV data...")
    df = read_pitvis_csv(csv_path)
    
    # 2. Group frames by video
    print("\n2. Grouping frames by video...")
    video_groups = group_frames_by_video(df)
    
    # 3. Limit processing
    print(f"\n3. Limiting processing: max {max_videos} videos, {max_frames_per_video} frames each")
    
    limited_video_groups = {}
    video_count = 0
    
    for video_name, frames in video_groups.items():
        if video_count >= max_videos:
            break
            
        limited_frames = frames[:max_frames_per_video]
        limited_video_groups[video_name] = limited_frames
        
        print(f"  Selected video {video_name}: {len(limited_frames)} frames")
        video_count += 1
    
    # 4. Calculate optical flow
    print(f"\n4. Calculating optical flow...")
    all_flow_stats = {}
    
    for video_name, video_frames in limited_video_groups.items():
        print(f"\nProcessing video: {video_name} ({len(video_frames)} frames)")
        
        if len(video_frames) < 2:
            print(f"Skipping video {video_name}: insufficient frames")
            continue
            
        flow_magnitudes = process_video_optical_flow(video_frames, frames_dir)
        
        all_flow_stats[video_name] = {
            'flow_magnitudes': flow_magnitudes
        }
        
        if flow_magnitudes:
            magnitudes = [f['mean_magnitude'] for f in flow_magnitudes]
            print(f"  Average optical flow magnitude: {np.mean(magnitudes):.3f}")
            print(f"  Max optical flow magnitude: {np.max(magnitudes):.3f}")
            print(f"  Min optical flow magnitude: {np.min(magnitudes):.3f}")
    
    # 5. Visualize results
    print(f"\n5. Generating visualization...")
    output_dir = "flow_analysis_demo_fixed"
    visualize_flow_statistics(all_flow_stats, output_dir)
    
    print(f"\nDemo completed! Results saved in {output_dir} directory")
    print("=" * 60)

def main():
    """主函数"""
    demo_optical_flow_analysis(max_videos=2, max_frames_per_video=50)

if __name__ == "__main__":
    main()

