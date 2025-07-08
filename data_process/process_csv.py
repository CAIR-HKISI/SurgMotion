import pandas as pd
import os
from pathlib import Path
from datetime import timedelta

def process_video_csv_fixed_length(
    input_csv_path,
    output_csv_path,
    clip_info_dir,
    frames_per_clip=64,  # 每个片段的帧数
    fps=1,  # 帧率
    base_video_path="/path/to/your/video/frames"
):
    """
    按固定帧数分割视频片段
    
    Args:
        input_csv_path: 输入CSV文件路径
        output_csv_path: 输出CSV文件路径
        clip_info_dir: 输出clip信息的目录
        frames_per_clip: 每个片段的帧数
        fps: 帧率
        base_video_path: 视频帧文件的基础路径
    """
    
    # 读取CSV文件
    df = pd.read_csv(input_csv_path)
    
    # 按视频ID分组
    video_groups = df.groupby('Case_ID')
    
    # 存储所有片段的信息
    all_clips_data = []
    
    for case_id, video_df in video_groups:
        # 确保按帧路径排序（假设帧路径按顺序命名）
        video_df = video_df.sort_values('Frame_Path').reset_index(drop=True)
        total_frames = len(video_df)
        
        print(f"\n处理视频 {case_id}:")
        print(f"  - 总帧数: {total_frames}")
        print(f"  - 总时长: {total_frames/fps:.2f} 秒")
        
        # 计算片段数量（向上取整）
        num_clips = (total_frames + frames_per_clip - 1) // frames_per_clip
        
        # 按固定帧数分割片段
        for clip_idx in range(num_clips):
            # 计算片段的起始和结束帧索引
            start_idx = clip_idx * frames_per_clip
            end_idx = min(start_idx + frames_per_clip, total_frames)
            clip_frames = video_df.iloc[start_idx:end_idx]
            
            # 确保片段至少有一帧
            if len(clip_frames) == 0:
                continue
            
            # 获取最后一帧作为标签
            last_frame = clip_frames.iloc[-1]
            clip_label = last_frame['Phase_GT']
            clip_phase_name = last_frame['Phase_Name']
            
            # 计算时间信息
            clip_start_time = start_idx / fps  # 秒
            clip_end_time = end_idx / fps
            clip_start_time_str = str(timedelta(seconds=int(clip_start_time)))
            clip_end_time_str = str(timedelta(seconds=int(clip_end_time)))
            
            # 生成片段标识符
            clip_identifier = f"case{case_id}_c{clip_idx:03d}_f{start_idx:06d}-{end_idx:06d}"
            
            # 创建片段信息目录
            os.makedirs(clip_info_dir, exist_ok=True)
            clip_frames_file = os.path.join(clip_info_dir, f"{clip_identifier}.txt")
            
            # 将片段的所有帧路径写入文件
            with open(clip_frames_file, 'w') as f:
                for _, row in clip_frames.iterrows():
                    full_path = os.path.join(base_video_path, row['Frame_Path'])
                    f.write(f"{full_path}\n")
            
            # 保存片段信息
            clip_info = {
                'clip_path': str(clip_frames_file),
                'label': clip_label,
                'label_name': clip_phase_name,
                'case_id': case_id,
                'clip_idx': clip_idx,
                'start_frame': start_idx,
                'end_frame': end_idx,
                'start_time': clip_start_time_str,
                'end_time': clip_end_time_str,
                'duration_seconds': len(clip_frames) / fps
            }
            
            all_clips_data.append(clip_info)
        
        print(f"  - 生成 {num_clips} 个片段")
    
    # 创建输出DataFrame
    output_df = pd.DataFrame(all_clips_data)
    
    # 保存为指定格式的CSV（路径+标签）
    with open(output_csv_path, 'w') as f:
        for _, row in output_df.iterrows():
            f.write(f"{row['clip_path']} {row['label']}\n")
    
    # 保存详细信息
    detailed_path = output_csv_path.replace('.csv', '_detailed.csv')
    output_df.to_csv(detailed_path, index=False)
    
    print(f"\n=== 处理完成 ===")
    print(f"总共生成 {len(all_clips_data)} 个片段")
    print(f"输出文件:")
    print(f"  - 主文件: {output_csv_path}")
    print(f"  - 详细信息: {detailed_path}")
    
    return output_df


# 使用示例
if __name__ == "__main__":
    process_video_csv_fixed_length(
        input_csv_path="/data/wjl/vjepa2/data/pitvis/train_metadata.csv",
        output_csv_path="fixed_64frames_clips.csv",
        clip_info_dir="/data/wjl/vjepa2/data_process/clip_64frames_info",
        frames_per_clip=64,  # 64帧/片段（64秒）
        fps=1,
        base_video_path="/data/wjl/vjepa2/data/pitvis"
    )

