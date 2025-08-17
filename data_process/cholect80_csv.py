#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CholecT80 数据集处理脚本（采样1fps，标注对齐采样帧）
"""

import os
import cv2
import pandas as pd
from tqdm import tqdm
import argparse

CLASS_LABELS = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]

def get_video_id_and_case_name(filename):
    base = os.path.splitext(filename)[0]
    if base.lower().startswith("video"):
        try:
            vid = int(base.replace("video", ""))
            return vid, f"video{vid:02d}"
        except:
            pass
    parts = base.split('-')
    try:
        vid = int(parts[0].replace("video", ""))
        return vid, f"video{vid:02d}"
    except:
        return None, None

def get_split(video_id):
    if 1 <= video_id <= 40:
        return 'train'
    elif 41 <= video_id <= 48:
        return 'val'
    elif 49 <= video_id <= 80:
        return 'test'
    return None

def extract_frames_1fps(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return 0, []
    os.makedirs(output_folder, exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS)
    stride = int(round(fps)) if fps > 0 else 25
    frame_idx, saved_idx = 0, 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total, desc=os.path.basename(video_path))
    sampled_indices = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            out_fn = os.path.join(output_folder, f'frame_{saved_idx:06d}.jpg')
            cv2.imwrite(out_fn, frame)
            sampled_indices.append(frame_idx)
            saved_idx += 1
        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    return saved_idx - 1, sampled_indices

def extract_all_videos_1fps(video_dir, output_dir):
    import json
    frames_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    if not video_files:
        print("未找到视频文件")
        return False
    for video_file in tqdm(video_files, desc="抽帧"):
        video_id, case_name = get_video_id_and_case_name(video_file)
        if not video_id:
            print(f"跳过无法识别ID的视频文件: {video_file}")
            continue
        video_path = os.path.join(video_dir, video_file)
        out_folder = os.path.join(frames_dir, case_name)
        count, sampled_indices = extract_frames_1fps(video_path, out_folder)
        # 保存采样帧原始编号
        sampled_indices_path = os.path.join(out_folder, 'sampled_indices.txt')
        with open(sampled_indices_path, 'w') as f:
            f.write('\n'.join(map(str, sampled_indices)))
        print(f"{video_file}: 共抽取 {count} 帧")
    return True

def process_annotations_to_csv(annot_dir, frames_dir, output_dir):
    all_data = []
    global_idx = 0
    annot_files = [f for f in os.listdir(annot_dir) if f.endswith('.txt')]
    if not annot_files:
        print("未找到标注文件")
        return False
    for annot_file in tqdm(annot_files, desc="处理标注"):
        video_id, case_name = get_video_id_and_case_name(annot_file)
        if not video_id:
            print(f"警告: 无法识别视频ID, 跳过 {annot_file}")
            continue
        split = get_split(video_id)
        if not split:
            print(f"警告: 视频ID {video_id} 不在split范围，跳过 {annot_file}")
            continue
        video_frames_path = os.path.join(frames_dir, case_name)
        if not os.path.isdir(video_frames_path):
            print(f"警告: 未找到帧目录 {video_frames_path}, 跳过")
            continue
        frame_files = sorted(
            [f for f in os.listdir(video_frames_path) if f.endswith('.jpg')],
            key=lambda x: int(x.split('_')[1].split('.')[0])
        )
        # 加载采样帧编号
        sampled_indices_path = os.path.join(video_frames_path, 'sampled_indices.txt')
        if not os.path.isfile(sampled_indices_path):
            print(f"警告: 未找到 {sampled_indices_path}, 跳过")
            continue
        with open(sampled_indices_path, 'r') as f:
            sampled_indices = [int(x.strip()) for x in f if x.strip()]
        # 加载标注
        annot_path = os.path.join(annot_dir, annot_file)
        with open(annot_path, 'r') as f:
            annot_lines = [line.strip() for line in f if line.strip()]
        # 检查数量一致
        if len(frame_files) != len(sampled_indices):
            print(f"警告: 采样帧数不等于编号数 ({case_name})，跳过")
            continue
        for i, (frame_file, frame_idx) in enumerate(zip(frame_files, sampled_indices)):
            if frame_idx >= len(annot_lines):
                print(f"警告: 索引越界 ({case_name})，跳过")
                continue
            parts = annot_lines[frame_idx].split('\t')
            if len(parts) < 2:
                continue
            phase_name = parts[1].strip()
            if phase_name not in CLASS_LABELS:
                print(f"警告: 未知标注 {phase_name} ({annot_file})")
                continue
            phase_gt = CLASS_LABELS.index(phase_name)
            frame_rel_path = f'frames/{case_name}/{frame_file}'
            all_data.append({
                'index': global_idx,
                'Hospital': 'CholecT80',
                'Year': 2016,
                'Case_Name': case_name,
                'Case_ID': video_id,
                'Frame_Path': frame_rel_path,
                'Frame': frame_file,
                'Phase_GT': phase_gt,
                'Phase_Name': phase_name,
                'Split': split
            })
            global_idx += 1
    if not all_data:
        print("没有有效数据生成CSV")
        return False
    columns = ['index', 'Hospital', 'Year', 'Case_Name', 'Case_ID',
               'Frame_Path', 'Frame', 'Phase_GT', 'Phase_Name', 'Split']
    df = pd.DataFrame(all_data, columns=columns)
    for split in ['train', 'val', 'test']:
        split_df = df[df['Split'] == split].reset_index(drop=True)
        out_path = os.path.join(output_dir, f'{split}_metadata.csv')
        split_df.to_csv(out_path, index=False)
        print(f"已保存: {out_path} ({len(split_df)}条记录)")
    return True

def main():
    parser = argparse.ArgumentParser(description="CholecT80数据处理（采样1fps，标注对齐采样帧）")
    subparsers = parser.add_subparsers(dest='command')
    # 抽帧
    p1 = subparsers.add_parser('extract', help='视频抽帧')
    p1.add_argument('--video_dir', default="/data2/wjl/CholecT80/videos", help='视频目录')
    p1.add_argument('--output_dir', default="data/Surge_Frames/CholecT80", help='输出根目录')
    # 生成CSV
    p2 = subparsers.add_parser('csv', help='生成train/val/test_metadata.csv')
    p2.add_argument('--annot_dir', default="/data2/wjl/CholecT80/phase_annotations", help='标注目录')
    p2.add_argument('--output_dir', default="data/Surge_Frames/CholecT80", help='输出根目录')
    # 一步到位
    p3 = subparsers.add_parser('all', help='完整流程')
    p3.add_argument('--video_dir', default="/data2/wjl/CholecT80/videos", help='视频目录')
    p3.add_argument('--annot_dir', default="/data2/wjl/CholecT80/phase_annotations", help='标注目录')
    p3.add_argument('--output_dir', default="data/Surge_Frames/CholecT80", help='输出根目录')
    args = parser.parse_args()
    if args.command == 'extract':
        return 0 if extract_all_videos_1fps(args.video_dir, args.output_dir) else 1
    elif args.command == 'csv':
        frames_dir = os.path.join(args.output_dir, 'frames')
        return 0 if process_annotations_to_csv(args.annot_dir, frames_dir, args.output_dir) else 1
    elif args.command == 'all':
        if not extract_all_videos_1fps(args.video_dir, args.output_dir):
            return 1
        frames_dir = os.path.join(args.output_dir, 'frames')
        if not process_annotations_to_csv(args.annot_dir, frames_dir, args.output_dir):
            return 1
        print("\n完成! 输出目录结构:")
        print(f"{args.output_dir}/")
        print("├── frames/")
        print("│   ├── video01/")
        print("│   └── ...")
        print("├── train_metadata.csv")
        print("├── val_metadata.csv")
        print("└── test_metadata.csv")
        return 0
    else:
        parser.print_help()
        return 1

if __name__ == '__main__':
    exit(main())

