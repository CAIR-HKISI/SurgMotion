import pandas as pd
import os
from pathlib import Path
from datetime import timedelta

# def process_video_csv_dense_sampling(
#     input_csv_path,
#     output_csv_path,
#     clip_info_dir,
#     window_size=16,  # 窗口大小（秒）
#     stride=1,        # 步长（秒）
#     fps=1,           # 帧率
#     base_video_path="/path/to/your/video/frames"
# ):
#     """
#     使用滑动窗口进行dense采样
    
#     Args:
#         input_csv_path: 输入CSV文件路径
#         output_csv_path: 输出CSV文件路径
#         clip_info_dir: 输出clip信息的目录
#         window_size: 窗口大小（秒）
#         stride: 滑动步长（秒）
#         fps: 帧率
#         base_video_path: 视频帧文件的基础路径
#     """
    
#     # 读取CSV文件
#     df = pd.read_csv(input_csv_path)
    
#     # 按视频ID分组
#     video_groups = df.groupby('Case_ID')
    
#     # 存储所有片段的信息
#     all_clips_data = []
    
#     # 计算窗口和步长对应的帧数
#     frames_per_window = window_size * fps
#     frames_per_stride = stride * fps
    
#     for case_id, video_df in video_groups:
#         # 确保按帧路径排序
#         video_df = video_df.sort_values('Frame_Path').reset_index(drop=True)
#         total_frames = len(video_df)
        
#         print(f"\n处理视频 {case_id}:")
#         print(f"  - 总帧数: {total_frames}")
#         print(f"  - 总时长: {total_frames/fps:.2f} 秒")
#         print(f"  - 窗口大小: {window_size} 秒 ({frames_per_window} 帧)")
#         print(f"  - 滑动步长: {stride} 秒 ({frames_per_stride} 帧)")
        
#         # 如果视频长度小于窗口大小，至少生成一个片段
#         if total_frames < frames_per_window:
#             print(f"  - 警告: 视频长度小于窗口大小，使用整个视频作为一个片段")
#             clip_frames = video_df
            
#             # 获取最后一帧作为标签
#             last_frame = clip_frames.iloc[-1]
#             clip_label = last_frame['Phase_GT']
#             clip_phase_name = last_frame['Phase_Name']
            
#             # 生成片段标识符
#             clip_identifier = f"case{case_id}_c000_f000000-{total_frames:06d}"
            
#             # 创建片段信息目录
#             os.makedirs(clip_info_dir, exist_ok=True)
#             clip_frames_file = os.path.join(clip_info_dir, f"{clip_identifier}.txt")
            
#             # 将片段的所有帧路径写入文件
#             with open(clip_frames_file, 'w') as f:
#                 for _, row in clip_frames.iterrows():
#                     full_path = os.path.join(base_video_path, row['Frame_Path'])
#                     f.write(f"{full_path}\n")
            
#             # 保存片段信息
#             clip_info = {
#                 'clip_path': str(clip_frames_file),
#                 'label': clip_label,
#                 'label_name': clip_phase_name,
#                 'case_id': case_id,
#                 'clip_idx': 0,
#                 'start_frame': 0,
#                 'end_frame': total_frames,
#                 'start_time': "0:00:00",
#                 'end_time': str(timedelta(seconds=int(total_frames/fps))),
#                 'duration_seconds': total_frames / fps
#             }
            
#             all_clips_data.append(clip_info)
#             print(f"  - 生成 1 个片段")
#             continue
        
#         # 使用滑动窗口采样
#         clip_count = 0
#         start_idx = 0
        
#         while start_idx < total_frames:  # 修改条件，允许处理边界情况
#             # 计算当前窗口的结束索引
#             end_idx = min(start_idx + frames_per_window, total_frames)
            
#             # 获取当前窗口的帧
#             clip_frames = video_df.iloc[start_idx:end_idx]
#             actual_frames = len(clip_frames)
            
#             # 如果帧数不足窗口大小，需要填充
#             if actual_frames < frames_per_window:
#                 print(f"  - 片段 {clip_count}: 实际帧数 {actual_frames} < 窗口大小 {frames_per_window}，进行填充")
                
#                 # 获取最后一帧用于填充
#                 last_frame_row = clip_frames.iloc[-1]
#                 last_frame_path = last_frame_row['Frame_Path']
                
#                 # 计算需要填充的帧数
#                 padding_frames = frames_per_window - actual_frames
                
#                 # 创建填充的DataFrame
#                 padding_data = []
#                 for i in range(padding_frames):
#                     padding_data.append({
#                         'Frame_Path': last_frame_path,
#                         'Phase_GT': last_frame_row['Phase_GT'],
#                         'Phase_Name': last_frame_row['Phase_Name'],
#                         'Case_ID': last_frame_row['Case_ID']
#                     })
                
#                 padding_df = pd.DataFrame(padding_data)
                
#                 # 将原始帧和填充帧合并
#                 clip_frames = pd.concat([clip_frames, padding_df], ignore_index=True)
                
#                 print(f"    - 填充了 {padding_frames} 帧，使用最后一帧路径: {last_frame_path}")
            
#             # 获取最后一帧作为标签
#             last_frame = clip_frames.iloc[-1]
#             clip_label = last_frame['Phase_GT']
#             clip_phase_name = last_frame['Phase_Name']
            
#             # 计算时间信息
#             clip_start_time = start_idx / fps  # 秒
#             clip_end_time = end_idx / fps
#             clip_start_time_str = str(timedelta(seconds=int(clip_start_time)))
#             clip_end_time_str = str(timedelta(seconds=int(clip_end_time)))
            
#             # 生成片段标识符
#             clip_identifier = f"case{case_id}_c{clip_count:03d}_f{start_idx:06d}-{end_idx:06d}"
#             if actual_frames < frames_per_window:
#                 clip_identifier += "_padded"  # 标记为填充片段
            
#             # 创建片段信息目录
#             os.makedirs(clip_info_dir, exist_ok=True)
#             clip_frames_file = os.path.join(clip_info_dir, f"{clip_identifier}.txt")
            
#             # 将片段的所有帧路径写入文件
#             with open(clip_frames_file, 'w') as f:
#                 for _, row in clip_frames.iterrows():
#                     full_path = row['Frame_Path']
#                     f.write(f"{full_path}\n")
            
#             # 保存片段信息
#             clip_info = {
#                 'clip_path': str(clip_frames_file),
#                 'label': clip_label,
#                 'label_name': clip_phase_name,
#                 'case_id': case_id,
#                 'clip_idx': clip_count,
#                 'start_frame': start_idx,
#                 'end_frame': end_idx,
#                 'actual_frames': actual_frames,  # 记录实际帧数
#                 'padded_frames': frames_per_window - actual_frames if actual_frames < frames_per_window else 0,  # 记录填充帧数
#                 'start_time': clip_start_time_str,
#                 'end_time': clip_end_time_str,
#                 'duration_seconds': frames_per_window / fps,
#                 'is_padded': actual_frames < frames_per_window  # 标记是否为填充片段
#             }
            
#             all_clips_data.append(clip_info)
            
#             # 移动到下一个位置
#             start_idx += frames_per_stride
#             clip_count += 1
            
#             # 如果下一个起始位置已经超出视频范围，退出循环
#             if start_idx >= total_frames:
#                 break
        
#         print(f"  - 生成 {clip_count} 个片段")
    
#     # 创建输出DataFrame
#     output_df = pd.DataFrame(all_clips_data)
    
#     # # 保存为指定格式的CSV（路径+标签）
#     # with open(output_csv_path, 'w') as f:
#     #     for _, row in output_df.iterrows():
#     #         f.write(f"{row['clip_path']} {row['label']}\n")
    
#     # 保存详细信息
#     detailed_path = output_csv_path.replace('.csv', '_detailed.csv')
#     output_df.to_csv(detailed_path, index=True, index_label="Index")
    
#     print(f"\n=== 处理完成 ===")
#     print(f"总共生成 {len(all_clips_data)} 个片段")
#     print(f"输出文件:")
#     print(f"  - 主文件: {output_csv_path}")
#     print(f"  - 详细信息: {detailed_path}")
    
#     return output_df


def process_video_csv_dense_sampling(
    input_csv_path,
    output_csv_path,
    clip_info_dir,
    window_size=16,
    stride=1,
    fps=1,
    base_video_path="/path/to/your/video/frames"
):
    """
    使用滑动窗口进行dense采样
    - 不足窗口大小时，用本窗口最后一帧补齐
    - 打印每个窗口对应的帧索引，方便调试
    """
    import pandas as pd
    import os
    from datetime import timedelta

    df = pd.read_csv(input_csv_path)
    video_groups = df.groupby('Case_ID')
    all_clips_data = []

    frames_per_window = window_size * fps
    frames_per_stride = stride * fps

    for case_id, video_df in video_groups:
        video_df = video_df.sort_values('Frame_Path').reset_index(drop=True)
        total_frames = len(video_df)

        print(f"\n处理视频 {case_id}:")
        print(f"  - 总帧数: {total_frames}")
        print(f"  - 窗口大小: {window_size} 秒 ({frames_per_window} 帧)")
        print(f"  - 步长: {stride} 秒 ({frames_per_stride} 帧)")

        clip_count = 0
        start_idx = 0

        while start_idx < total_frames:
            # 窗口结束位置（含）
            end_idx = start_idx + 1
            start_idx_for_window = end_idx - frames_per_window

            if start_idx_for_window < 0:
                # 开头不足 window_size
                valid_frames = video_df.iloc[0:end_idx].copy()
                actual_frames = len(valid_frames)

                last_row = valid_frames.iloc[-1]
                padding_frames = frames_per_window - actual_frames
                padding_data = [last_row.to_dict()] * padding_frames
                padding_df = pd.DataFrame(padding_data)

                clip_frames = pd.concat([valid_frames, padding_df], ignore_index=True)
                is_padded = True
            else:
                # 正常窗口
                clip_frames = video_df.iloc[start_idx_for_window:end_idx].copy()
                actual_frames = len(clip_frames)
                padding_frames = 0
                is_padded = False

            # ====== 打印窗口中的帧下标 ======
            frame_indices = list(range(max(start_idx_for_window, 0), end_idx))
            if is_padded:
                print(f"片段 {clip_count}: 帧索引 {frame_indices} + 补齐 {padding_frames} 个 (用 {end_idx-1}号帧)")
            else:
                print(f"片段 {clip_count}: 帧索引 {frame_indices}")
            # ==============================

            # 标签：窗口最后一帧
            last_frame = clip_frames.iloc[-1]
            clip_label = last_frame['Phase_GT']
            clip_phase_name = last_frame['Phase_Name']

            # 时间戳
            clip_start_time = max(start_idx_for_window, 0) / fps
            clip_end_time = end_idx / fps
            clip_start_time_str = str(timedelta(seconds=int(clip_start_time)))
            clip_end_time_str = str(timedelta(seconds=int(clip_end_time)))

            # 片段ID
            clip_identifier = (
                f"case{case_id}_c{clip_count:03d}_f{start_idx_for_window:06d}-{end_idx:06d}"
            )
            if is_padded:
                clip_identifier += "_padded"

            # 保存帧路径到 txt
            os.makedirs(clip_info_dir, exist_ok=True)
            clip_frames_file = os.path.join(clip_info_dir, f"{clip_identifier}.txt")
            with open(clip_frames_file, 'w') as f:
                for _, row in clip_frames.iterrows():
                    full_path = os.path.join(row['Frame_Path'])
                    f.write(f"{full_path}\n")

            # 保存 clip 信息
            clip_info = {
                'clip_path': str(clip_frames_file),
                'label': clip_label,
                'label_name': clip_phase_name,
                'case_id': case_id,
                'clip_idx': clip_count,
                'start_frame': max(start_idx_for_window, 0),
                'end_frame': end_idx,
                'actual_frames': actual_frames,
                'padded_frames': padding_frames,
                'start_time': clip_start_time_str,
                'end_time': clip_end_time_str,
                'duration_seconds': frames_per_window / fps,
                'is_padded': is_padded
            }
            all_clips_data.append(clip_info)

            start_idx += frames_per_stride
            clip_count += 1

        print(f"  - 生成 {clip_count} 个片段")

    # 输出
    output_df = pd.DataFrame(all_clips_data)
    detailed_path = output_csv_path.replace('.csv', '_detailed.csv')
    output_df.to_csv(detailed_path, index=True, index_label="Index")

    print(f"\n=== 处理完成 ===")
    print(f"总共生成 {len(all_clips_data)} 个片段")
    print(f"输出文件:")
    print(f"  - 主文件: {output_csv_path}")
    print(f"  - 详细信息: {detailed_path}")

    return output_df



def process_train_and_val(base_data_path="/data/wjl/vjepa2/data/pitvis",
                         output_base_path="/data/wjl/vjepa2/data_process",
                         window_size=16):
    """
    对train和val数据集都进行dense采样处理
    """
    
    # 处理训练集
    print("=== 处理训练集 ===")
    train_df = process_video_csv_dense_sampling(
        input_csv_path=os.path.join(base_data_path, "train_metadata.csv"),
        output_csv_path=os.path.join(output_base_path, f"train_dense_{window_size}f.csv"),
        clip_info_dir=os.path.join(output_base_path, f"clip_dense_{window_size}f_info/train"),
        window_size=window_size,  # 16秒窗口
        stride=1,        # 1秒步长
        fps=1,
        base_video_path=base_data_path
    )
    
    # 处理验证集
    print("\n\n=== 处理验证集 ===")
    val_df = process_video_csv_dense_sampling(
        input_csv_path=os.path.join(base_data_path, "val_metadata.csv"),
        output_csv_path=os.path.join(output_base_path, f"val_dense_{window_size}f.csv"),
        clip_info_dir=os.path.join(output_base_path, f"clip_dense_{window_size}f_info/val"),
        window_size=window_size,  # 16秒窗口
        stride=1,        # 1秒步长
        fps=1,
        base_video_path=base_data_path
    )

    # 处理验证集
    print("\n\n=== 处理测试集 ===")
    test_df = process_video_csv_dense_sampling(
        input_csv_path=os.path.join(base_data_path, "test_metadata.csv"),
        output_csv_path=os.path.join(output_base_path, f"test_dense_{window_size}f.csv"),
        clip_info_dir=os.path.join(output_base_path, f"clip_dense_{window_size}f_info/test"),
        window_size=window_size,  # 16秒窗口
        stride=1,        # 1秒步长
        fps=1,
        base_video_path=base_data_path
    )
    
    # 打印统计信息
    print("\n\n=== 总体统计 ===")
    print(f"训练集片段数: {len(train_df)}")
    print(f"验证集片段数: {len(val_df)}")
    print(f"测试集片段数: {len(test_df)}")
    
    # 统计填充片段
    if 'is_padded' in train_df.columns:
        train_padded = train_df['is_padded'].sum()
        val_padded = val_df['is_padded'].sum()
        test_padded = test_df['is_padded'].sum()
        print(f"\n填充片段统计:")
        print(f"  训练集填充片段: {train_padded}/{len(train_df)} ({train_padded/len(train_df)*100:.1f}%)")
        print(f"  验证集填充片段: {val_padded}/{len(val_df)} ({val_padded/len(val_df)*100:.1f}%)")
        print(f"  测试集填充片段: {test_padded}/{len(test_df)} ({test_padded/len(test_df)*100:.1f}%)")
    
    # 统计每个类别的片段数
    print("\n训练集各类别分布:")
    train_label_counts = train_df['label'].value_counts().sort_index()
    for label, count in train_label_counts.items():
        label_name = train_df[train_df['label'] == label]['label_name'].iloc[0]
        print(f"  类别 {label} ({label_name}): {count} 个片段")
    
    print("\n验证集各类别分布:")
    val_label_counts = val_df['label'].value_counts().sort_index()
    for label, count in val_label_counts.items():
        label_name = val_df[val_df['label'] == label]['label_name'].iloc[0]
        print(f"  类别 {label} ({label_name}): {count} 个片段")
    
    print("\n测试集各类别分布:")
    test_label_counts = test_df['label'].value_counts().sort_index()
    for label, count in test_label_counts.items():
        label_name = test_df[test_df['label'] == label]['label_name'].iloc[0]
        print(f"  类别 {label} ({label_name}): {count} 个片段")


# 使用示例
if __name__ == "__main__":
    
    # for window_size in [16, 32, 64, 128]:
    #     print(f"###### 处理StrasBypass70数据集，窗口大小: {window_size}f ######")
    #     process_train_and_val(
    #             base_data_path="data/Surge_Frames/MultiBypass140/StrasBypass70",
    #             output_base_path=f"data/Surge_Frames/MultiBypass140/StrasBypass70/clips_{window_size}f",
    #             window_size=window_size
    #         )
    
    
    # for window_size in [16, 32, 64, 128]:
    #     print(f"###### 处理BernBypass70数据集，窗口大小: {window_size}f ######")
    #     process_train_and_val(
    #         base_data_path="data/Surge_Frames/MultiBypass140/BernBypass70",
    #         output_base_path=f"data/Surge_Frames/MultiBypass140/BernBypass70/clips_{window_size}f",
    #         window_size=window_size
    #     )

    # for window_size in [16, 32, 64, 128]:
    #     print(f"###### 处理Cholec80数据集，窗口大小: {window_size}f ######")
    #     process_train_and_val(
    #         base_data_path="data/Surge_Frames/Cholec80",
    #         output_base_path=f"data/Surge_Frames/Cholec80/clips_{window_size}f",
    #         window_size=window_size
    #     )


    # for window_size in [16, 32, 64, 128]:
    #     print(f"###### 处理AutoLaparo数据集，窗口大小: {window_size}f ######")
    #     process_train_and_val(
    #         base_data_path="data/Surge_Frames/AutoLaparo",
    #         output_base_path=f"data/Surge_Frames/AutoLaparo/clips_{window_size}f",
    #         window_size=window_size
    #     )

    # for window_size in [16, 32, 64, 128]:
    #     print(f"###### 处理M2CAI2016数据集，窗口大小: {window_size}f ######")
    #     process_train_and_val(
    #         base_data_path="data/Surge_Frames/M2CAI16",
    #         output_base_path=f"data/Surge_Frames/M2CAI16/clips_{window_size}f",
    #         window_size=window_size
    #     )

    # for window_size in [16, 32, 64, 128]:
    #     print(f"###### 处理PitVis数据集，窗口大小: {window_size}f ######")
    #     process_train_and_val(
    #             base_data_path="data/Surge_Frames/PitVis",
    #             output_base_path=f"data/Surge_Frames/PitVis/clips_{window_size}f",
    #             window_size=window_size
    #         )

    # for window_size in [16, 32, 64, 128]:
    #     print(f"###### 处理EgoSurgery数据集，窗口大小: {window_size}f ######")
    #     process_train_and_val(
    #             base_data_path="data/Surge_Frames/EgoSurgery",
    #             output_base_path=f"data/Surge_Frames/EgoSurgery/clips_{window_size}f",
    #             window_size=window_size
    #         )

    # for window_size in [16, 32, 64, 128]:
    #     print(f"###### 处理EgoSurgery数据集，窗口大小: {window_size}f ######")
    #     process_train_and_val(
    #             base_data_path="data/Surge_Frames/BernBypass70",
    #             output_base_path=f"data/Surge_Frames/BernBypass70/clips_{window_size}f",
    #             window_size=window_size
    #         )

    # for window_size in [16, 32, 64, 128]:
    #     print(f"###### 处理EgoSurgery数据集，窗口大小: {window_size}f ######")
    #     process_train_and_val(
    #             base_data_path="data/Surge_Frames/StrasBypass70",
    #             output_base_path=f"data/Surge_Frames/StrasBypass70/clips_{window_size}f",
    #             window_size=window_size
    #         )

    # for window_size in [16, 32, 64, 128]:
    #     print(f"###### 处理EgoSurgery数据集，窗口大小: {window_size}f ######")
    #     process_train_and_val(
    #             base_data_path="data/Surge_Frames/PmLR50",
    #             output_base_path=f"data/Surge_Frames/PmLR50/clips_{window_size}f",
    #             window_size=window_size
    #         )

    # for window_size in [16, 32, 64, 128]:
    #         print(f"###### 处理EgoSurgery数据集，窗口大小: {window_size}f ######")
    #         process_train_and_val(
    #                 base_data_path="data/Surge_Frames/OphNet2024_phase",
    #                 output_base_path=f"data/Surge_Frames/OphNet2024_phase/clips_{window_size}f",
    #                 window_size=window_size
    #             )

    # for window_size in [16, 32, 64, 128]:
    #         print(f"###### 处理EgoSurgery数据集，窗口大小: {window_size}f ######")
    #         process_train_and_val(
    #                 base_data_path="data/Surge_Frames/PolypDiag",
    #                 output_base_path=f"data/Surge_Frames/PolypDiag/clips_{window_size}f",
    #                 window_size=window_size
                # )

    # for window_size in [16, 32, 64, 128]:
    #         print(f"###### 处理EgoSurgery数据集，窗口大小: {window_size}f ######")
    #         process_train_and_val(
    #                 base_data_path="data/Surge_Frames/CATARACTS",
    #                 output_base_path=f"data/Surge_Frames/CATARACTS/clips_{window_size}f",
    #                 window_size=window_size
    #             )

    # for window_size in [16, 32, 64, 128]:
    #         print(f"###### 处理EgoSurgery数据集，窗口大小: {window_size}f ######")
    #         process_train_and_val(
    #                 base_data_path="data/Surge_Frames/JIGSAWS",
    #                 output_base_path=f"data/Surge_Frames/JIGSAWS/clips_{window_size}f",
    #                 window_size=window_size
    #             )

    # for window_size in [16, 32, 64, 128]:
    #         print(f"###### 处理EgoSurgery数据集，窗口大小: {window_size}f ######")
    #         process_train_and_val(
    #                 base_data_path="data/Surge_Frames/AIxsuture",
    #                 output_base_path=f"data/Surge_Frames/AIxsuture/clips_{window_size}f",
    #                 window_size=window_size
    #             )

    # for window_size in [16, 32, 64, 128]:
    #         print(f"###### 处理EgoSurgery数据集，窗口大小: {window_size}f ######")
    #         process_train_and_val(
    #                 base_data_path="data/Surge_Frames/AVOS",
    #                 output_base_path=f"data/Surge_Frames/AVOS/clips_{window_size}f",
    #                 window_size=window_size
    #             )

    for window_size in [64]:
            print(f"###### 处理EgoSurgery数据集，窗口大小: {window_size}f ######")
            process_train_and_val(
                    base_data_path="data/Surge_Frames/GynSurg_Action",
                    output_base_path=f"data/Surge_Frames/GynSurg_Action/clips_{window_size}f",
                    window_size=window_size
                )
