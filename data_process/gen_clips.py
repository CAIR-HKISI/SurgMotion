# import pandas as pd
# import os
# from pathlib import Path
# from datetime import timedelta


# def process_video_csv_dense_sampling(
#     input_csv_path,
#     output_csv_path,
#     clip_info_dir,
#     window_size=16,
#     stride=1,
#     fps=1,
#     base_video_path="/path/to/your/video/frames"
# ):
#     """
#     使用滑动窗口进行dense采样
#     - 不足窗口大小时，用本窗口最后一帧补齐
#     - 打印每个窗口对应的帧索引，方便调试
#     """


#     df = pd.read_csv(input_csv_path)
#     video_groups = df.groupby('Case_ID')
#     all_clips_data = []

#     frames_per_window = window_size * fps
#     frames_per_stride = stride * fps

#     for case_id, video_df in video_groups:
#         video_df = video_df.sort_values('Frame_Path').reset_index(drop=True)
#         total_frames = len(video_df)

#         print(f"\n处理视频 {case_id}:")
#         print(f"  - 总帧数: {total_frames}")
#         print(f"  - 窗口大小: {window_size} 秒 ({frames_per_window} 帧)")
#         print(f"  - 步长: {stride} 秒 ({frames_per_stride} 帧)")

#         clip_count = 0
#         start_idx = 0

#         while start_idx < total_frames:
#             # 窗口结束位置（含）
#             end_idx = start_idx + 1
#             start_idx_for_window = end_idx - frames_per_window

#             if start_idx_for_window < 0:
#                 # 开头不足 window_size
#                 valid_frames = video_df.iloc[0:end_idx].copy()
#                 actual_frames = len(valid_frames)

#                 last_row = valid_frames.iloc[-1]
#                 padding_frames = frames_per_window - actual_frames
#                 padding_data = [last_row.to_dict()] * padding_frames
#                 padding_df = pd.DataFrame(padding_data)

#                 clip_frames = pd.concat([valid_frames, padding_df], ignore_index=True)
#                 is_padded = True
#             else:
#                 # 正常窗口
#                 clip_frames = video_df.iloc[start_idx_for_window:end_idx].copy()
#                 actual_frames = len(clip_frames)
#                 padding_frames = 0
#                 is_padded = False

#             # ====== 打印窗口中的帧下标 ======
#             frame_indices = list(range(max(start_idx_for_window, 0), end_idx))
#             if is_padded:
#                 print(f"片段 {clip_count}: 帧索引 {frame_indices} + 补齐 {padding_frames} 个 (用 {end_idx-1}号帧)")
#             else:
#                 print(f"片段 {clip_count}: 帧索引 {frame_indices}")
#             # ==============================

#             # 标签：窗口最后一帧
#             last_frame = clip_frames.iloc[-1]
#             clip_label = last_frame['Phase_GT']
#             clip_phase_name = last_frame['Phase_Name']

#             # 时间戳
#             clip_start_time = max(start_idx_for_window, 0) / fps
#             clip_end_time = end_idx / fps
#             clip_start_time_str = str(timedelta(seconds=int(clip_start_time)))
#             clip_end_time_str = str(timedelta(seconds=int(clip_end_time)))

#             # 片段ID
#             clip_identifier = (
#                 f"case{case_id}_c{clip_count:03d}_f{start_idx_for_window:06d}-{end_idx:06d}"
#             )
#             if is_padded:
#                 clip_identifier += "_padded"

#             # 保存帧路径到 txt
#             os.makedirs(clip_info_dir, exist_ok=True)
#             clip_frames_file = os.path.join(clip_info_dir, f"{clip_identifier}.txt")
#             with open(clip_frames_file, 'w') as f:
#                 for _, row in clip_frames.iterrows():
#                     full_path = os.path.join(row['Frame_Path'])
#                     f.write(f"{full_path}\n")

#             # 保存 clip 信息
#             clip_info = {
#                 'clip_path': str(clip_frames_file),
#                 'label': clip_label,
#                 'label_name': clip_phase_name,
#                 'case_id': case_id,
#                 'clip_idx': clip_count,
#                 'start_frame': max(start_idx_for_window, 0),
#                 'end_frame': end_idx,
#                 'actual_frames': actual_frames,
#                 'padded_frames': padding_frames,
#                 'start_time': clip_start_time_str,
#                 'end_time': clip_end_time_str,
#                 'duration_seconds': frames_per_window / fps,
#                 'is_padded': is_padded
#             }
#             all_clips_data.append(clip_info)

#             start_idx += frames_per_stride
#             clip_count += 1

#         print(f"  - 生成 {clip_count} 个片段")

#     # 输出
#     output_df = pd.DataFrame(all_clips_data)
#     detailed_path = output_csv_path.replace('.csv', '_detailed.csv')
#     output_df.to_csv(detailed_path, index=True, index_label="Index")

#     print(f"\n=== 处理完成 ===")
#     print(f"总共生成 {len(all_clips_data)} 个片段")
#     print(f"输出文件:")
#     print(f"  - 主文件: {output_csv_path}")
#     print(f"  - 详细信息: {detailed_path}")

#     return output_df



# def process_train_and_val(base_data_path="/data/wjl/vjepa2/data/pitvis",
#                          output_base_path="/data/wjl/vjepa2/data_process",
#                          window_size=16):
#     """
#     对train和val数据集都进行dense采样处理
#     """
    
#     # 处理训练集
#     print("=== 处理训练集 ===")
#     train_df = process_video_csv_dense_sampling(
#         input_csv_path=os.path.join(base_data_path, "train_metadata.csv"),
#         output_csv_path=os.path.join(output_base_path, f"train_dense_{window_size}f.csv"),
#         clip_info_dir=os.path.join(output_base_path, f"clip_dense_{window_size}f_info/train"),
#         window_size=window_size,  # 16秒窗口
#         stride=1,        # 1秒步长
#         fps=1,
#         base_video_path=base_data_path
#     )
    
#     # 处理验证集
#     print("\n\n=== 处理验证集 ===")
#     val_df = process_video_csv_dense_sampling(
#         input_csv_path=os.path.join(base_data_path, "val_metadata.csv"),
#         output_csv_path=os.path.join(output_base_path, f"val_dense_{window_size}f.csv"),
#         clip_info_dir=os.path.join(output_base_path, f"clip_dense_{window_size}f_info/val"),
#         window_size=window_size,  # 16秒窗口
#         stride=1,        # 1秒步长
#         fps=1,
#         base_video_path=base_data_path
#     )

#     # 处理验证集
#     print("\n\n=== 处理测试集 ===")
#     test_df = process_video_csv_dense_sampling(
#         input_csv_path=os.path.join(base_data_path, "test_metadata.csv"),
#         output_csv_path=os.path.join(output_base_path, f"test_dense_{window_size}f.csv"),
#         clip_info_dir=os.path.join(output_base_path, f"clip_dense_{window_size}f_info/test"),
#         window_size=window_size,  # 16秒窗口
#         stride=1,        # 1秒步长
#         fps=1,
#         base_video_path=base_data_path
#     )
    
#     # 打印统计信息
#     print("\n\n=== 总体统计 ===")
#     print(f"训练集片段数: {len(train_df)}")
#     print(f"验证集片段数: {len(val_df)}")
#     print(f"测试集片段数: {len(test_df)}")
    
#     # 统计填充片段
#     if 'is_padded' in train_df.columns:
#         train_padded = train_df['is_padded'].sum()
#         val_padded = val_df['is_padded'].sum()
#         test_padded = test_df['is_padded'].sum()
#         print(f"\n填充片段统计:")
#         print(f"  训练集填充片段: {train_padded}/{len(train_df)} ({train_padded/len(train_df)*100:.1f}%)")
#         print(f"  验证集填充片段: {val_padded}/{len(val_df)} ({val_padded/len(val_df)*100:.1f}%)")
#         print(f"  测试集填充片段: {test_padded}/{len(test_df)} ({test_padded/len(test_df)*100:.1f}%)")
    
#     # 统计每个类别的片段数
#     print("\n训练集各类别分布:")
#     train_label_counts = train_df['label'].value_counts().sort_index()
#     for label, count in train_label_counts.items():
#         label_name = train_df[train_df['label'] == label]['label_name'].iloc[0]
#         print(f"  类别 {label} ({label_name}): {count} 个片段")
    
#     print("\n验证集各类别分布:")
#     val_label_counts = val_df['label'].value_counts().sort_index()
#     for label, count in val_label_counts.items():
#         label_name = val_df[val_df['label'] == label]['label_name'].iloc[0]
#         print(f"  类别 {label} ({label_name}): {count} 个片段")
    
#     print("\n测试集各类别分布:")
#     test_label_counts = test_df['label'].value_counts().sort_index()
#     for label, count in test_label_counts.items():
#         label_name = test_df[test_df['label'] == label]['label_name'].iloc[0]
#         print(f"  类别 {label} ({label_name}): {count} 个片段")


# # 使用示例
# if __name__ == "__main__":
    
#     # for window_size in [16, 32, 64, 128]:
#     #     print(f"###### 处理StrasBypass70数据集，窗口大小: {window_size}f ######")
#     #     process_train_and_val(
#     #             base_data_path="data/Surge_Frames/MultiBypass140/StrasBypass70",
#     #             output_base_path=f"data/Surge_Frames/MultiBypass140/StrasBypass70/clips_{window_size}f",
#     #             window_size=window_size
#     #         )
    
    
#     # for window_size in [16, 32, 64, 128]:
#     #     print(f"###### 处理BernBypass70数据集，窗口大小: {window_size}f ######")
#     #     process_train_and_val(
#     #         base_data_path="data/Surge_Frames/MultiBypass140/BernBypass70",
#     #         output_base_path=f"data/Surge_Frames/MultiBypass140/BernBypass70/clips_{window_size}f",
#     #         window_size=window_size
#     #     )

#     # for window_size in [16, 32, 64, 128]:
#     #     print(f"###### 处理Cholec80数据集，窗口大小: {window_size}f ######")
#     #     process_train_and_val(
#     #         base_data_path="data/Surge_Frames/Cholec80",
#     #         output_base_path=f"data/Surge_Frames/Cholec80/clips_{window_size}f",
#     #         window_size=window_size
#     #     )


#     # for window_size in [16, 32, 64, 128]:
#     #     print(f"###### 处理AutoLaparo数据集，窗口大小: {window_size}f ######")
#     #     process_train_and_val(
#     #         base_data_path="data/Surge_Frames/AutoLaparo",
#     #         output_base_path=f"data/Surge_Frames/AutoLaparo/clips_{window_size}f",
#     #         window_size=window_size
#     #     )

#     # for window_size in [16, 32, 64, 128]:
#     #     print(f"###### 处理M2CAI2016数据集，窗口大小: {window_size}f ######")
#     #     process_train_and_val(
#     #         base_data_path="data/Surge_Frames/M2CAI16",
#     #         output_base_path=f"data/Surge_Frames/M2CAI16/clips_{window_size}f",
#     #         window_size=window_size
#     #     )

#     # for window_size in [16, 32, 64, 128]:
#     #     print(f"###### 处理PitVis数据集，窗口大小: {window_size}f ######")
#     #     process_train_and_val(
#     #             base_data_path="data/Surge_Frames/PitVis",
#     #             output_base_path=f"data/Surge_Frames/PitVis/clips_{window_size}f",
#     #             window_size=window_size
#     #         )

#     # for window_size in [16, 32, 64, 128]:
#     #     print(f"###### 处理EgoSurgery数据集，窗口大小: {window_size}f ######")
#     #     process_train_and_val(
#     #             base_data_path="data/Surge_Frames/EgoSurgery",
#     #             output_base_path=f"data/Surge_Frames/EgoSurgery/clips_{window_size}f",
#     #             window_size=window_size
#     #         )

#     # for window_size in [16, 32, 64, 128]:
#     #     print(f"###### 处理EgoSurgery数据集，窗口大小: {window_size}f ######")
#     #     process_train_and_val(
#     #             base_data_path="data/Surge_Frames/BernBypass70",
#     #             output_base_path=f"data/Surge_Frames/BernBypass70/clips_{window_size}f",
#     #             window_size=window_size
#     #         )

#     # for window_size in [16, 32, 64, 128]:
#     #     print(f"###### 处理EgoSurgery数据集，窗口大小: {window_size}f ######")
#     #     process_train_and_val(
#     #             base_data_path="data/Surge_Frames/StrasBypass70",
#     #             output_base_path=f"data/Surge_Frames/StrasBypass70/clips_{window_size}f",
#     #             window_size=window_size
#     #         )

#     # for window_size in [16, 32, 64, 128]:
#     #     print(f"###### 处理EgoSurgery数据集，窗口大小: {window_size}f ######")
#     #     process_train_and_val(
#     #             base_data_path="data/Surge_Frames/PmLR50",
#     #             output_base_path=f"data/Surge_Frames/PmLR50/clips_{window_size}f",
#     #             window_size=window_size
#     #         )

#     # for window_size in [16, 32, 64, 128]:
#     #         print(f"###### 处理EgoSurgery数据集，窗口大小: {window_size}f ######")
#     #         process_train_and_val(
#     #                 base_data_path="data/Surge_Frames/OphNet2024_phase",
#     #                 output_base_path=f"data/Surge_Frames/OphNet2024_phase/clips_{window_size}f",
#     #                 window_size=window_size
#     #             )

#     # for window_size in [16, 32, 64, 128]:
#     #         print(f"###### 处理EgoSurgery数据集，窗口大小: {window_size}f ######")
#     #         process_train_and_val(
#     #                 base_data_path="data/Surge_Frames/PolypDiag",
#     #                 output_base_path=f"data/Surge_Frames/PolypDiag/clips_{window_size}f",
#     #                 window_size=window_size
#                 # )

#     # for window_size in [16, 32, 64, 128]:
#     #         print(f"###### 处理EgoSurgery数据集，窗口大小: {window_size}f ######")
#     #         process_train_and_val(
#     #                 base_data_path="data/Surge_Frames/CATARACTS",
#     #                 output_base_path=f"data/Surge_Frames/CATARACTS/clips_{window_size}f",
#     #                 window_size=window_size
#     #             )

#     # for window_size in [16, 32, 64, 128]:
#     #         print(f"###### 处理EgoSurgery数据集，窗口大小: {window_size}f ######")
#     #         process_train_and_val(
#     #                 base_data_path="data/Surge_Frames/JIGSAWS",
#     #                 output_base_path=f"data/Surge_Frames/JIGSAWS/clips_{window_size}f",
#     #                 window_size=window_size
#     #             )

#     # for window_size in [16, 32, 64, 128]:
#     #         print(f"###### 处理EgoSurgery数据集，窗口大小: {window_size}f ######")
#     #         process_train_and_val(
#     #                 base_data_path="data/Surge_Frames/AIxsuture",
#     #                 output_base_path=f"data/Surge_Frames/AIxsuture/clips_{window_size}f",
#     #                 window_size=window_size
#     #             )

#     # for window_size in [16, 32, 64, 128]:
#     #         print(f"###### 处理EgoSurgery数据集，窗口大小: {window_size}f ######")
#     #         process_train_and_val(
#     #                 base_data_path="data/Surge_Frames/AVOS",
#     #                 output_base_path=f"data/Surge_Frames/AVOS/clips_{window_size}f",
#     #                 window_size=window_size
#     #             )

#     for window_size in [64]:
#             print(f"###### 处理EgoSurgery数据集，窗口大小: {window_size}f ######")
#             process_train_and_val(
#                     base_data_path="data/Surge_Frames/GynSurg_Action",
#                     output_base_path=f"data/Surge_Frames/GynSurg_Action/clips_{window_size}f",
#                     window_size=window_size
#                 )


import pandas as pd
import os
import argparse
from pathlib import Path
from datetime import timedelta


def process_video_csv_dense_sampling(
    input_csv_path,
    output_csv_path,
    clip_info_dir,
    window_size=16,
    stride=1,
    fps=1,
    base_video_path="/path/to/your/video/frames",
    enable_padding=True
):
    """
    使用滑动窗口进行dense采样
    - 不足窗口大小时，可选是否用本窗口最后一帧补齐
    """
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
        print(f"  - 窗口大小: {window_size} 秒（{frames_per_window} 帧）")
        print(f"  - 步长: {stride} 秒（{frames_per_stride} 帧）")
        print(f"  - 启用 padding: {enable_padding}")

        clip_count = 0
        start_idx = 0

        while start_idx < total_frames:
            end_idx = start_idx + 1
            start_idx_for_window = end_idx - frames_per_window

            if start_idx_for_window < 0:
                valid_frames = video_df.iloc[0:end_idx].copy()
                actual_frames = len(valid_frames)

                if enable_padding:
                    last_row = valid_frames.iloc[-1]
                    padding_frames = frames_per_window - actual_frames
                    padding_data = [last_row.to_dict()] * padding_frames
                    padding_df = pd.DataFrame(padding_data)
                    clip_frames = pd.concat([valid_frames, padding_df], ignore_index=True)
                    is_padded = True
                else:
                    # 不进行padding，则跳过帧不足的片段
                    if actual_frames < frames_per_window:
                        start_idx += frames_per_stride
                        continue
                    clip_frames = valid_frames
                    padding_frames = 0
                    is_padded = False
            else:
                clip_frames = video_df.iloc[start_idx_for_window:end_idx].copy()
                actual_frames = len(clip_frames)
                padding_frames = 0
                is_padded = False

            frame_indices = list(range(max(start_idx_for_window, 0), end_idx))
            if is_padded:
                print(f"片段 {clip_count}: 帧索引 {frame_indices} + 补齐 {padding_frames} 个")
            else:
                print(f"片段 {clip_count}: 帧索引 {frame_indices}")

            last_frame = clip_frames.iloc[-1]
            clip_label = last_frame['Phase_GT']
            clip_phase_name = last_frame['Phase_Name']

            clip_start_time = max(start_idx_for_window, 0) / fps
            clip_end_time = end_idx / fps
            clip_start_time_str = str(timedelta(seconds=int(clip_start_time)))
            clip_end_time_str = str(timedelta(seconds=int(clip_end_time)))

            clip_identifier = (
                f"case{case_id}_c{clip_count:03d}_f{start_idx_for_window:06d}-{end_idx:06d}"
            )
            if is_padded:
                clip_identifier += "_padded"

            os.makedirs(clip_info_dir, exist_ok=True)
            clip_frames_file = os.path.join(clip_info_dir, f"{clip_identifier}.txt")
            with open(clip_frames_file, 'w') as f:
                for _, row in clip_frames.iterrows():
                    full_path = os.path.join(row['Frame_Path'])
                    f.write(f"{full_path}\n")

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

    output_df = pd.DataFrame(all_clips_data)
    detailed_path = output_csv_path.replace('.csv', '_detailed.csv')
    output_df.to_csv(detailed_path, index=True, index_label="Index")

    print(f"\n=== 处理完成 ===")
    print(f"总共生成 {len(all_clips_data)} 个片段")
    print(f"输出文件:")
    print(f"  - 主文件: {output_csv_path}")
    print(f"  - 详细信息: {detailed_path}")

    return output_df


def process_train_and_val(base_data_path, output_base_path, window_size, stride, fps, enable_padding):
    """处理 train / val / test 三个子集"""
    subsets = ["train", "val", "test"]
    for subset in subsets:
        print(f"\n=== 处理 {subset} 集 ===")
        input_csv = os.path.join(base_data_path, f"{subset}_metadata.csv")
        output_csv = os.path.join(output_base_path, f"{subset}_dense_{window_size}f.csv")
        clip_dir = os.path.join(output_base_path, f"clip_dense_{window_size}f_info/{subset}")

        process_video_csv_dense_sampling(
            input_csv_path=input_csv,
            output_csv_path=output_csv,
            clip_info_dir=clip_dir,
            window_size=window_size,
            stride=stride,
            fps=fps,
            base_video_path=base_data_path,
            enable_padding=enable_padding
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dense sampling with controllable stride and padding")
    parser.add_argument("--base_data_path", type=str, default="data/Surge_Frames/GynSurg_Action",
                        help="Base path to dataset containing train/val/test_metadata.csv")
    parser.add_argument("--window_size", type=int, default=64, help="Temporal window size (in seconds or frames/fps)")
    parser.add_argument("--stride", type=int, default=1, help="Stride (in seconds)")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second")
    parser.add_argument("--padding", type=bool, default=True, help="Enable padding for short clips")
    args = parser.parse_args()

    output_dir = os.path.join(args.base_data_path, f"clips_{args.window_size}f")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=== 参数设置 ===")
    print(f"数据路径: {args.base_data_path}")
    print(f"窗口大小: {args.window_size}")
    print(f"步长: {args.stride}")
    print(f"FPS: {args.fps}")
    print(f"启用Padding: {args.padding}")

    process_train_and_val(
        base_data_path=args.base_data_path,
        output_base_path=output_dir,
        window_size=args.window_size,
        stride=args.stride,
        fps=args.fps,
        enable_padding=args.padding
    )
