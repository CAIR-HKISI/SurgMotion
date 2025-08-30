#!/usr/bin/env python3
"""
测试填充功能的脚本
"""

import pandas as pd
import os
import tempfile
from data_process.gen_clips import process_video_csv_dense_sampling

def create_test_data():
    """创建测试数据"""
    # 创建一个测试视频（17帧，窗口大小为16帧，最后一个片段需要填充）
    test_data = []
    for i in range(17):  # 17帧，窗口16帧，最后一个片段需要填充1帧
        test_data.append({
            'Frame_Path': f'frame_{i:06d}.jpg',
            'Phase_GT': 1,
            'Phase_Name': 'Test_Phase',
            'Case_ID': 1
        })
    
    return pd.DataFrame(test_data)

def test_padding():
    """测试填充功能"""
    print("=== 测试填充功能 ===")
    
    # 创建测试数据
    test_df = create_test_data()
    print(f"测试数据: {len(test_df)} 帧")
    
    # 使用当前目录
    test_dir = "test_output"
    os.makedirs(test_dir, exist_ok=True)
    
    # 保存测试数据
    test_csv_path = os.path.join(test_dir, "test_metadata.csv")
    test_df.to_csv(test_csv_path, index=False)
    
    # 设置输出路径
    output_csv_path = os.path.join(test_dir, "test_output.csv")
    clip_info_dir = os.path.join(test_dir, "clip_info")
    
    # 测试填充功能（窗口大小16，17帧数据，最后一个片段需要填充）
    print(f"\n使用窗口大小16帧处理17帧数据...")
    result_df = process_video_csv_dense_sampling(
        input_csv_path=test_csv_path,
        output_csv_path=output_csv_path,
        clip_info_dir=clip_info_dir,
        window_size=16,  # 16帧窗口
        stride=1,        # 1帧步长
        fps=1,
        base_video_path=test_dir
    )
    
    print(f"\n处理结果:")
    print(f"生成片段数: {len(result_df)}")
    
    if len(result_df) > 0:
        # 检查是否有填充片段
        if 'is_padded' in result_df.columns:
            padded_count = result_df['is_padded'].sum()
            print(f"填充片段数: {padded_count}")
            
            if padded_count > 0:
                padded_sample = result_df[result_df['is_padded'] == True].iloc[0]
                print(f"填充片段示例:")
                print(f"  - 实际帧数: {padded_sample['actual_frames']}")
                print(f"  - 填充帧数: {padded_sample['padded_frames']}")
                print(f"  - 片段路径: {padded_sample['clip_path']}")
                
                # 检查生成的txt文件
                if os.path.exists(padded_sample['clip_path']):
                    with open(padded_sample['clip_path'], 'r') as f:
                        frame_paths = f.readlines()
                    print(f"  - txt文件中的帧数: {len(frame_paths)}")
                    print(f"  - 前5帧: {[p.strip() for p in frame_paths[:5]]}")
                    print(f"  - 后5帧: {[p.strip() for p in frame_paths[-5:]]}")
                    
                    # 检查是否有重复帧
                    unique_frames = set([p.strip() for p in frame_paths])
                    print(f"  - 唯一帧数: {len(unique_frames)}")
                    print(f"  - 重复帧数: {len(frame_paths) - len(unique_frames)}")
        
        # 打印所有片段的详细信息
        print(f"\n所有片段详细信息:")
        for i, row in result_df.iterrows():
            print(f"  片段{i}: 实际帧数={row.get('actual_frames', 'N/A')}, 填充帧数={row.get('padded_frames', 'N/A')}, 是否填充={row.get('is_padded', 'N/A')}")
    
    print(f"\n测试完成!")
    print(f"输出文件保存在: {test_dir}")

if __name__ == "__main__":
    test_padding()
