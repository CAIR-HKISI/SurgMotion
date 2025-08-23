import pandas as pd
import os
from pathlib import Path
from datetime import timedelta
import numpy as np

def process_video_csv_dense_sampling(
    input_csv_path,
    output_csv_path,
    clip_info_dir,
    window_size=16,  # 窗口大小（帧数，每帧=1秒）
    stride=1,        # 步长（帧数，每帧=1秒）
    base_video_path="/path/to/your/video/frames",
    keyword="Phase"
):
    """
    使用滑动窗口进行dense采样
    在bernbypass数据集中：1帧 = 1秒
    
    Args:
        input_csv_path: 输入CSV文件路径
        output_csv_path: 输出详细CSV文件路径
        clip_info_dir: 输出clip信息的目录
        window_size: 窗口大小（帧数，每帧=1秒）
        stride: 滑动步长（帧数，每帧=1秒）
        base_video_path: 视频帧文件的基础路径
    """
    
    # 读取CSV文件
    df = pd.read_csv(input_csv_path)
    
    # 按Case_Name分组（bernbypass数据集按Case_Name分组）
    video_groups = df.groupby('Case_Name')
    
    # 存储所有片段的信息
    all_clips_data = []
    
    for case_name, video_df in video_groups:
        # 提取Case_ID
        case_id = video_df['Case_ID'].iloc[0]
        
        # 修复：正确提取帧ID - 从文件名中提取数字部分
        def extract_frame_id(frame_path):
            """从帧路径中提取帧ID"""
            filename = os.path.basename(frame_path)
            # 去掉扩展名
            name_without_ext = filename.split('.')[0]
            # 提取最后的数字部分（例如从 BBP01_00000014 中提取 00000014）
            if '_' in name_without_ext:
                frame_id_str = name_without_ext.split('_')[-1]
            else:
                frame_id_str = name_without_ext
            
            # 转换为整数
            try:
                return int(frame_id_str)
            except ValueError:
                print(f"警告: 无法从 {frame_path} 中提取帧ID，使用默认值0")
                return 0
        
        video_df = video_df.copy()
        video_df['frame_id'] = video_df['Frame_Path'].apply(extract_frame_id)
        
        # 按frame_id排序
        video_df = video_df.sort_values('frame_id').reset_index(drop=True)
        total_frames = len(video_df)
        
        # 获取视频的起始帧ID，用于计算相对时间
        start_frame_id = int(video_df['frame_id'].min())  # 转换为int
        
        print(f"\n处理视频 {case_name} (Case_ID: {case_id}):")
        print(f"  - 总帧数: {total_frames} 帧")
        print(f"  - 帧ID范围: {video_df['frame_id'].min()} - {video_df['frame_id'].max()}")
        print(f"  - 总时长: {total_frames} 秒 (1帧=1秒)")
        print(f"  - 窗口大小: {window_size} 帧 ({window_size} 秒)")
        print(f"  - 滑动步长: {stride} 帧 ({stride} 秒)")
        
        # 如果视频长度小于窗口大小，至少生成一个片段
        if total_frames < window_size:
            print(f"  - 警告: 视频长度({total_frames}帧)小于窗口大小({window_size}帧)，使用整个视频作为一个片段")
            clip_frames = video_df
            
            # 获取最后一帧作为标签
            last_frame = clip_frames.iloc[-1]
            if keyword == "Phase":
                clip_label = last_frame['Phase_GT']
                clip_phase_name = last_frame['Phase_Name']
            elif keyword == "Step":
                clip_label = last_frame['Step_GT']
                clip_phase_name = last_frame['Step_Name']
            # clip_label = last_frame['Step_GT']
            # clip_phase_name = last_frame['Step_Name']
            
            
            # 获取实际的帧ID范围
            clip_start_frame_id = int(clip_frames.iloc[0]['frame_id'])  # 转换为int
            clip_end_frame_id = int(clip_frames.iloc[-1]['frame_id'])   # 转换为int
            
            # 计算相对于视频开始的时间（假设帧ID连续）
            relative_start_seconds = clip_start_frame_id - start_frame_id
            relative_end_seconds = clip_end_frame_id - start_frame_id + 1  # +1因为包含结束帧
            
            # 生成片段标识符
            clip_identifier = f"{case_name}_c000_f{clip_start_frame_id:08d}-{clip_end_frame_id:08d}"
            
            # 创建片段信息目录
            os.makedirs(clip_info_dir, exist_ok=True)
            clip_frames_file = os.path.join(clip_info_dir, f"{clip_identifier}.txt")
            
            # 将片段的所有帧路径写入文件
            with open(clip_frames_file, 'w') as f:
                for _, row in clip_frames.iterrows():
                    # 使用相对路径或绝对路径
                    frame_path = row['Frame_Path']
                    if not os.path.isabs(frame_path):
                        full_path = os.path.join(base_video_path, frame_path)
                    else:
                        full_path = frame_path
                    f.write(f"{full_path}\n")
            
            # 保存片段信息
            clip_info = {
                'clip_path': str(clip_frames_file),
                'label': clip_label,
                'label_name': clip_phase_name,
                'case_name': case_name,
                'case_id': case_id,
                'clip_idx': 0,
                'start_frame': clip_start_frame_id,
                'end_frame': clip_end_frame_id,
                'start_time': str(timedelta(seconds=int(relative_start_seconds))),
                'end_time': str(timedelta(seconds=int(relative_end_seconds))),
                'duration_seconds': total_frames,  # 帧数=秒数
                'duration_frames': total_frames
            }
            
            all_clips_data.append(clip_info)
            print(f"  - 生成 1 个片段 (时长: {total_frames}秒)")
            continue
        
        # 使用滑动窗口采样
        clip_count = 0
        start_idx = 0
        
        while start_idx + window_size <= total_frames:
            # 获取当前窗口的帧
            end_idx = start_idx + window_size
            clip_frames = video_df.iloc[start_idx:end_idx]
            
            # 获取最后一帧作为标签
            last_frame = clip_frames.iloc[-1]
            # clip_label = last_frame['Step_GT']
            # clip_phase_name = last_frame['Step_Name']
            if keyword == "Phase":
                clip_label = last_frame['Phase_GT']
                clip_phase_name = last_frame['Phase_Name']
            elif keyword == "Step":
                clip_label = last_frame['Step_GT']
                clip_phase_name = last_frame['Step_Name']
            
            # 获取实际的帧ID范围
            clip_start_frame_id = int(clip_frames.iloc[0]['frame_id'])  # 转换为int
            clip_end_frame_id = int(clip_frames.iloc[-1]['frame_id'])   # 转换为int
            
            # 计算相对于视频开始的时间
            relative_start_seconds = clip_start_frame_id - start_frame_id
            relative_end_seconds = clip_end_frame_id - start_frame_id + 1  # +1因为包含结束帧
            
            # 计算时间信息 (1帧=1秒) - 确保是int类型
            clip_start_time_str = str(timedelta(seconds=int(relative_start_seconds)))
            clip_end_time_str = str(timedelta(seconds=int(relative_end_seconds)))
            
            # 生成片段标识符 - 使用实际的帧ID
            clip_identifier = f"{case_name}_c{clip_count:03d}_f{clip_start_frame_id:08d}-{clip_end_frame_id:08d}"
            
            # 创建片段信息目录
            os.makedirs(clip_info_dir, exist_ok=True)
            clip_frames_file = os.path.join(clip_info_dir, f"{clip_identifier}.txt")
            
            # 将片段的所有帧路径写入文件
            with open(clip_frames_file, 'w') as f:
                for _, row in clip_frames.iterrows():
                    # 使用相对路径或绝对路径
                    frame_path = row['Frame_Path']
                    # if not os.path.isabs(frame_path):
                    #     full_path = os.path.join(base_video_path, frame_path)
                    # else:
                    full_path = frame_path
                    f.write(f"{full_path}\n")
            
            # 保存片段信息
            clip_info = {
                'clip_path': str(clip_frames_file),
                'label': clip_label,
                'label_name': clip_phase_name,
                'case_name': case_name,
                'case_id': case_id,
                'clip_idx': clip_count,
                'start_frame': clip_start_frame_id,
                'end_frame': clip_end_frame_id,
                'start_time': clip_start_time_str,
                'end_time': clip_end_time_str,
                'duration_seconds': window_size,  # 窗口大小=秒数
                'duration_frames': window_size    # 窗口大小=帧数
            }
            
            all_clips_data.append(clip_info)
            
            # 移动到下一个位置
            start_idx += stride
            clip_count += 1
        
        print(f"  - 生成 {clip_count} 个片段，每个片段 {window_size} 秒({window_size} 帧)")
    
    # 创建输出DataFrame
    output_df = pd.DataFrame(all_clips_data)
    
    # 只保存详细信息CSV
    output_df.to_csv(output_csv_path, index=False)
    
    print(f"\n=== 处理完成 ===")
    print(f"总共生成 {len(all_clips_data)} 个片段")
    print(f"每个片段包含 {window_size} 帧 ({window_size} 秒)")
    print(f"输出文件: {output_csv_path}")
    
    return output_df


def analyze_phase_durations(metadata_csv_path, output_stats_path=None):
    """
    统计每个手术阶段的持续时间统计信息
    
    Args:
        metadata_csv_path: 元数据CSV文件路径
        output_stats_path: 统计结果输出路径（可选）
    
    Returns:
        DataFrame: 包含各阶段统计信息的DataFrame
    """
    # 读取元数据
    df = pd.read_csv(metadata_csv_path)
    
    print(f"\n=== 分析阶段持续时间 ===")
    print(f"数据文件: {metadata_csv_path}")
    print(f"总帧数: {len(df)}")
    print(f"总视频数: {df['Case_Name'].nunique()}")
    
    # 存储每个阶段的持续时间
    phase_durations = {}
    
    # 按视频分组
    for case_name, video_df in df.groupby('Case_Name'):
        video_df = video_df.sort_values('Frame_Path').reset_index(drop=True)
        
        # 识别阶段变化点
        phase_changes = []
        current_phase = video_df.iloc[0]['Phase_GT']
        current_phase_name = video_df.iloc[0]['Phase_Name']
        start_idx = 0
        
        for i in range(1, len(video_df)):
            if video_df.iloc[i]['Phase_GT'] != current_phase:
                # 阶段结束
                duration = i - start_idx  # 秒数（1帧=1秒）
                phase_changes.append({
                    'case_name': case_name,
                    'phase_id': current_phase,
                    'phase_name': current_phase_name,
                    'duration_seconds': duration,
                    'start_frame': start_idx,
                    'end_frame': i-1
                })
                
                # 开始新阶段
                current_phase = video_df.iloc[i]['Phase_GT']
                current_phase_name = video_df.iloc[i]['Phase_Name']
                start_idx = i
        
        # 添加最后一个阶段
        duration = len(video_df) - start_idx
        phase_changes.append({
            'case_name': case_name,
            'phase_id': current_phase,
            'phase_name': current_phase_name,
            'duration_seconds': duration,
            'start_frame': start_idx,
            'end_frame': len(video_df)-1
        })
        
        # 按阶段分组存储
        for phase_info in phase_changes:
            phase_id = phase_info['phase_id']
            phase_name = phase_info['phase_name']
            duration = phase_info['duration_seconds']
            
            key = (phase_id, phase_name)
            if key not in phase_durations:
                phase_durations[key] = []
            phase_durations[key].append(duration)
    
    # 计算统计信息
    stats_data = []
    
    for (phase_id, phase_name), durations in phase_durations.items():
        durations = np.array(durations)
        
        stats = {
            'Phase_ID': phase_id,
            'Phase_Name': phase_name,
            'Count': len(durations),  # 出现次数
            'Mean_Duration_Seconds': float(np.mean(durations)),  # 转换为float
            'Min_Duration_Seconds': float(np.min(durations)),
            'Max_Duration_Seconds': float(np.max(durations)),
            'Median_Duration_Seconds': float(np.median(durations)),
            'Std_Duration_Seconds': float(np.std(durations)),
            'Q25_Duration_Seconds': float(np.percentile(durations, 25)),
            'Q75_Duration_Seconds': float(np.percentile(durations, 75)),
            'Total_Duration_Seconds': float(np.sum(durations))
        }
        
        # 转换为分钟
        for key in ['Mean_Duration_Seconds', 'Min_Duration_Seconds', 'Max_Duration_Seconds', 
                   'Median_Duration_Seconds', 'Std_Duration_Seconds', 'Q25_Duration_Seconds', 
                   'Q75_Duration_Seconds', 'Total_Duration_Seconds']:
            stats[key.replace('_Seconds', '_Minutes')] = stats[key] / 60
        
        stats_data.append(stats)
    
    # 创建DataFrame并排序
    stats_df = pd.DataFrame(stats_data)
    stats_df = stats_df.sort_values('Phase_ID').reset_index(drop=True)
    
    # 打印统计结果
    print(f"\n{'='*100}")
    print("手术阶段持续时间统计")
    print(f"{'='*100}")
    
    for _, row in stats_df.iterrows():
        print(f"\n阶段 {row['Phase_ID']}: {row['Phase_Name']}")
        print(f"  出现次数: {row['Count']}")
        print(f"  平均时长: {row['Mean_Duration_Minutes']:.2f} 分钟 ({row['Mean_Duration_Seconds']:.1f} 秒)")
        print(f"  最短时长: {row['Min_Duration_Minutes']:.2f} 分钟 ({row['Min_Duration_Seconds']:.1f} 秒)")
        print(f"  最长时长: {row['Max_Duration_Minutes']:.2f} 分钟 ({row['Max_Duration_Seconds']:.1f} 秒)")
        print(f"  中位数时长: {row['Median_Duration_Minutes']:.2f} 分钟 ({row['Median_Duration_Seconds']:.1f} 秒)")
        print(f"  标准差: {row['Std_Duration_Minutes']:.2f} 分钟 ({row['Std_Duration_Seconds']:.1f} 秒)")
        print(f"  25%分位数: {row['Q25_Duration_Minutes']:.2f} 分钟 ({row['Q25_Duration_Seconds']:.1f} 秒)")
        print(f"  75%分位数: {row['Q75_Duration_Minutes']:.2f} 分钟 ({row['Q75_Duration_Seconds']:.1f} 秒)")
        print(f"  总时长: {row['Total_Duration_Minutes']:.2f} 分钟 ({row['Total_Duration_Seconds']:.1f} 秒)")
    
    # 保存统计结果
    if output_stats_path:
        stats_df.to_csv(output_stats_path, index=False)
        print(f"\n统计结果已保存至: {output_stats_path}")
    
    return stats_df


def process_bernbypass_train_val_test(
    base_data_path="data/MultiBypass140/BernBypass70",
    metadata_dir="./",  # CSV元数据文件目录
    output_base_path="./bernbypass_clips",
    window_size=16,  # 窗口大小（帧数，每帧=1秒）
    keyword="Phase"
):
    """
    对bernbypass的train、val和test数据集都进行dense采样处理
    在bernbypass数据集中：1帧 = 1秒
    """
    
    datasets = ['train', 'val', 'test']
    results = {}
    
    print(f"{keyword} 配置信息:")
    print(f"  - 窗口大小: {window_size} 帧 ({window_size} 秒)")
    print(f"  - 滑动步长: 1 帧 (1 秒)")
    print(f"  - 帧率关系: 1帧 = 1秒")
    
    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"=== 处理{dataset.upper()}集 ===")
        print(f"{'='*50}")
        
        input_csv = os.path.join(metadata_dir, f"{dataset}_metadata.csv")
        output_csv = os.path.join(output_base_path, f"{dataset}_dense_{window_size}f_detailed.csv")
        clip_info_dir = os.path.join(output_base_path, f"clip_dense_{window_size}f_info/{dataset}")
        
        # 检查输入文件是否存在
        if not os.path.exists(input_csv):
            print(f"警告: 输入文件 {input_csv} 不存在，跳过{dataset}集")
            continue
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        
        result_df = process_video_csv_dense_sampling(
            input_csv_path=input_csv,
            output_csv_path=output_csv,
            clip_info_dir=clip_info_dir,
            window_size=window_size,  # 窗口大小（帧数）
            stride=1,        # 1帧步长
            base_video_path=base_data_path,
            keyword=keyword
        )
        
        results[dataset] = result_df
        
        # 分析阶段持续时间
        print(f"\n--- 分析{dataset.upper()}集阶段持续时间 ---")
        stats_output_path = os.path.join(output_base_path, f"{dataset}_phase_duration_stats.csv")
        analyze_phase_durations(input_csv, stats_output_path)
    
    # 打印总体统计信息
    print(f"\n\n{'='*50}")
    print("=== 总体统计 ===")
    print(f"{'='*50}")
    
    total_clips = 0
    total_duration = 0
    
    for dataset, df in results.items():
        if df is not None:
            dataset_clips = len(df)
            dataset_duration = df['duration_seconds'].sum()
            total_clips += dataset_clips
            total_duration += dataset_duration
            
            print(f"\n{dataset.upper()}集:")
            print(f"  - 片段数: {dataset_clips}")
            print(f"  - 视频数: {df['case_name'].nunique()}")
            print(f"  - 总时长: {dataset_duration} 秒 ({dataset_duration/60:.1f} 分钟)")
            
            # 统计每个阶段的片段数
            print(f"  - 各阶段分布:")
            label_counts = df['label'].value_counts().sort_index()
            for label, count in label_counts.items():
                if len(df[df['label'] == label]) > 0:
                    label_name = df[df['label'] == label]['label_name'].iloc[0]
                    label_duration = df[df['label'] == label]['duration_seconds'].sum()
                    print(f"    阶段 {label} ({label_name}): {count} 个片段, {label_duration} 秒")
    
    print(f"\n总计:")
    print(f"  - 总片段数: {total_clips}")
    print(f"  - 总时长: {total_duration} 秒 ({total_duration/60:.1f} 分钟, {total_duration/3600:.1f} 小时)")
    
    return results


# 使用示例
if __name__ == "__main__":
    # 配置参数
    window_size=64  # 窗口大小（帧数，每帧=1秒）
    
    # # 处理bernbypass数据集
    results = process_bernbypass_train_val_test(
        base_data_path="data/MultiBypass140/BernBypass70",  # 图片数据路径
        metadata_dir="data/MultiBypass140/BernBypass70",  # CSV元数据文件目录（train_data.csv, val_data.csv, test_data.csv）
        output_base_path=f"data/Surge_Frames/bernbypass_phase_clips_{window_size}f",  # 输出目录
        window_size=window_size,
        keyword="Phase"
    )
    
    # 也可以单独分析某个数据集的阶段持续时间
    # analyze_phase_durations("data/MultiBypass140/BernBypass70/train_metadata.csv", "data/MultiBypass140/BernBypass70/train_phase_stats.csv")
