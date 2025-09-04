import cv2 as cv
import ptlflow
import torch
import argparse
import os
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter

def process_video_to_flow(input_video_path, output_video_path, model_name='raft_small', ckpt_path='things'):
    """
    处理输入视频，生成光流视频并保存
    
    Args:
        input_video_path: 输入视频路径
        output_video_path: 输出光流视频路径
        model_name: 光流模型名称
        ckpt_path: 模型检查点路径
    """
    # 获取光流模型
    print(f"正在加载模型: {model_name}")
    model = ptlflow.get_model(model_name, ckpt_path=ckpt_path)
    
    # 打开输入视频
    cap = cv.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开输入视频: {input_video_path}")
    
    # 获取视频属性
    fps = int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息: {width}x{height}, {fps} FPS, {total_frames} 帧")
    
    # 设置输出视频编码器
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 初始化IO适配器
    io_adapter = None
    
    # 读取第一帧
    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("无法读取视频第一帧")
    
    frame_count = 0
    
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        print(f"处理第 {frame_count}/{total_frames} 帧", end='\r')
        
        # 初始化IO适配器（使用第一帧的尺寸）
        if io_adapter is None:
            io_adapter = IOAdapter(model, prev_frame.shape[:2])
        
        # 准备输入
        images = [prev_frame, curr_frame]
        inputs = io_adapter.prepare_inputs(images)
        
        # 前向传播
        with torch.no_grad():
            predictions = model(inputs)
        
        # 获取光流
        flows = predictions['flows']
        
        # 转换为RGB表示
        flow_rgb = flow_utils.flow_to_rgb(flows)
        flow_rgb = flow_rgb[0, 0].permute(1, 2, 0)
        flow_rgb_npy = flow_rgb.detach().cpu().numpy()
        
        # 转换为BGR格式（OpenCV使用）
        flow_bgr_npy = cv.cvtColor(flow_rgb_npy, cv.COLOR_RGB2BGR)
        
        # 写入输出视频
        out.write(flow_bgr_npy)
        
        # 更新前一帧
        prev_frame = curr_frame
    
    # 释放资源
    cap.release()
    out.release()
    print(f"\n光流视频已保存到: {output_video_path}")

def main():
    parser = argparse.ArgumentParser(description='从视频生成光流视频')
    parser.add_argument('--input', '-i', required=True, help='输入视频路径')
    parser.add_argument('--output', '-o', required=True, help='输出光流视频路径')
    parser.add_argument('--model', default='raft_small', help='光流模型名称 (默认: raft_small)')
    parser.add_argument('--ckpt', default='things', help='模型检查点路径 (默认: things)')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入视频文件不存在: {args.input}")
        return
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        process_video_to_flow(args.input, args.output, args.model, args.ckpt)
    except Exception as e:
        print(f"处理过程中出现错误: {e}")

if __name__ == "__main__":
    main()


