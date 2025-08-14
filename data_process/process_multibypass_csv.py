import os
import json
import csv
import re
from pathlib import Path

def parse_case_info(video_name):
    """解析视频名称获取医院、年份和病例信息"""
    # 匹配BernBypass(BBP)和StrasBypass(SBP)的视频命名格式
    bern_pattern = r'BBP(\d+)\.mp4'
    stras_pattern = r'SBP(\d+)\.mp4'
    
    if re.match(bern_pattern, video_name):
        hospital = "Bern"
        case_id = re.match(bern_pattern, video_name).group(1)
    elif re.match(stras_pattern, video_name):
        hospital = "Strasbourg"
        case_id = re.match(stras_pattern, video_name).group(1)
    else:
        hospital = "Unknown"
        case_id = "Unknown"
    
    # 假设年份信息从病例ID推断或默认
    year = "2020"  # 实际应用中可能需要根据实际数据调整
    
    case_name = f"{hospital}Case{case_id}"
    return hospital, year, case_name, case_id

def get_gt_label(timestamp, label_entries):
    """根据时间戳获取对应的标签"""
    for entry in label_entries:
        if entry['start'] <= timestamp <= entry['end']:
            return entry['label_id'], entry['label_name']
    return -1, "Unknown"

def process_split(split_dir, labels_root, dataset_root, csv_writer, phase_type='phases'):
    """处理训练/验证分割并写入CSV"""
    # 获取当前分割的帧目录
    frames_dir = os.path.join(split_dir, 'frames')
    if not os.path.exists(frames_dir):
        print(f"警告: 帧目录不存在 - {frames_dir}")
        return
    
    # 确定是Bern还是Strasbourg的标签
    if 'bern' in labels_root.lower():
        label_subdir = 'bern/labels_by70'
    elif 'strasbourg' in labels_root.lower():
        label_subdir = 'strasbourg/labels_by70'
    else:
        print(f"警告: 无法识别的标签目录 - {labels_root}")
        return
    
    # 处理每个视频的帧
    for video_id in os.listdir(frames_dir):
        video_frames_dir = os.path.join(frames_dir, video_id)
        if not os.path.isdir(video_frames_dir):
            continue
            
        # 解析视频信息
        video_name = f"{video_id}.mp4"
        hospital, year, case_name, case_id = parse_case_info(video_name)
        
        # 加载对应的标签文件
        label_file = os.path.join(labels_root, label_subdir, f"{video_id}.mp4.json")
        if not os.path.exists(label_file):
            print(f"警告: 标签文件不存在 - {label_file}")
            continue
            
        with open(label_file, 'r') as f:
            label_data = json.load(f)
        
        # 处理每一帧
        frame_files = [f for f in os.listdir(video_frames_dir) if f.endswith(('.jpg', '.png'))]
        frame_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))  # 按帧号排序
        
        for frame_file in frame_files:
            # 提取帧号(假设帧率为1fps，帧号对应毫秒级时间戳)
            frame_num = int(re.findall(r'\d+', frame_file)[0])
            timestamp = frame_num * 1000  # 转换为毫秒
            
            # 获取对应的阶段标签
            phase_id, phase_name = get_gt_label(timestamp, label_data[phase_type])
            
            # 构建帧路径
            frame_path = os.path.join(video_frames_dir, frame_file)
            # 转换为相对于数据集根目录的路径
            relative_path = os.path.relpath(frame_path, dataset_root)
            
            # 写入CSV
            csv_writer.writerow([
                hospital,
                year,
                case_name,
                video_name,
                case_id,
                relative_path,
                phase_id,
                phase_name
            ])

def main():
    # 配置路径(根据实际情况修改)
    mby140_root = os.environ.get('MBy140', './MultiBypass140')  # 从环境变量获取或默认
    dataset_root = os.path.join(mby140_root, 'datasets', 'MultiBypass140')
    
    # 标签和分割路径
    splits = {
        'train': {
            'bern': os.path.join(mby140_root, 'labels', 'bern', 'labels_by70_splits', 'train'),
            'strasbourg': os.path.join(mby140_root, 'labels', 'strasbourg', 'labels_by70_splits', 'train')
        },
        'val': {
            'bern': os.path.join(mby140_root, 'labels', 'bern', 'labels_by70_splits', 'val'),
            'strasbourg': os.path.join(mby140_root, 'labels', 'strasbourg', 'labels_by70_splits', 'val')
        }
    }
    
    # CSV头部
    fieldnames = ['Hospital', 'Year', 'Case_Name', 'Video_Name', 'Case_ID', 'Frame_Path', 'Phase_GT', 'Phase_Name']
    
    # 处理所有数据并生成总metadata.csv
    with open('metadata.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        
        # 处理训练集和验证集
        for split_type in ['train', 'val']:
            for hospital in ['bern', 'strasbourg']:
                split_dir = splits[split_type][hospital]
                if os.path.exists(split_dir):
                    process_split(
                        split_dir=split_dir,
                        labels_root=os.path.join(mby140_root, 'labels'),
                        dataset_root=dataset_root,
                        csv_writer=writer
                    )
    
    # 分别生成训练集和验证集CSV
    for split_type in ['train', 'val']:
        with open(f'{split_type}_metadata.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)
            
            for hospital in ['bern', 'strasbourg']:
                split_dir = splits[split_type][hospital]
                if os.path.exists(split_dir):
                    process_split(
                        split_dir=split_dir,
                        labels_root=os.path.join(mby140_root, 'labels'),
                        dataset_root=dataset_root,
                        csv_writer=writer
                    )
    
    print("CSV文件生成完成!")

if __name__ == "__main__":
    main()