import argparse
import os
import cv2
import torch
import numpy as np
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import (
    raft_large, raft_small,
    Raft_Large_Weights, Raft_Small_Weights
)
from torchvision.utils import flow_to_image


def preprocess(img1, img2, weights, device):
    """
    输入: 两帧 numpy 图像 (H, W, 3)
    输出: 预处理后的 tensor batch
    """
    img1 = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
    img2 = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)

    transforms = weights.transforms()
    img1, img2 = transforms(img1, img2)
    return img1.to(device), img2.to(device)


def estimate_optical_flow(video_path, output_path, model, weights, device, comparison=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 获取总帧数用于进度显示
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📊 视频信息: {total_frames} 帧, {fps:.2f} FPS, {width}x{height}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 如果需要对比，则宽度翻倍
    out_width = width * 2 if comparison else width
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_width, height))

    ret, prev_frame = cap.read()
    if not ret:
        print("❌ 视频为空")
        return

    frame_count = 0
    print("🚀 开始处理光流...")
    
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # 打印进度
        if frame_count % 10 == 0 or frame_count == total_frames:  # 每10帧或最后一帧打印一次
            progress = (frame_count / total_frames) * 100
            print(f"📈 处理进度: {frame_count}/{total_frames} ({progress:.1f}%)")

        # 预处理
        img1, img2 = preprocess(prev_frame, curr_frame, weights, device)

        # 光流计算
        with torch.no_grad():
            flow_up = model(img1, img2)[-1]

        # 光流可视化
        flow_img = flow_to_image(flow_up)[0].permute(1, 2, 0).cpu().numpy()
        flow_resized = cv2.resize(flow_img, (width, height))

        if comparison:
            combined = np.hstack([curr_frame, flow_resized])
            writer.write(combined)
        else:
            writer.write(flow_resized)

        prev_frame = curr_frame

    cap.release()
    writer.release()
    print(f"✅ 光流视频已保存: {output_path}")
    print(f"🎉 总共处理了 {frame_count} 帧")


def main():
    parser = argparse.ArgumentParser(description="使用 TorchVision RAFT 进行手术视频光流估计")
    parser.add_argument("--video", type=str, default="data/SurgicalAction160/01_abdominal_access/01_03.mp4", help="输入视频路径")
    parser.add_argument("--output", type=str, default="logs/raft_flow.mp4", help="输出视频路径")
    parser.add_argument("--model", type=str, default="raft_large", choices=["raft_small", "raft_large"], help="RAFT 模型类型")
    parser.add_argument("--comparison", action="store_true", help="拼接对比模式（左边原视频，右边光流）")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")
    print(f"📁 输入视频: {args.video}")
    print(f"📁 输出路径: {args.output}")
    print(f"🤖 选择模型: {args.model}")

    # 加载模型和权重
    print("⏳ 正在加载模型...")
    if args.model == "raft_large":
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights).to(device).eval()
    else:
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights).to(device).eval()
    print("✅ 模型加载完成！")

    print("=" * 50)
    estimate_optical_flow(args.video, args.output, model, weights, device, comparison=args.comparison)
    print("=" * 50)
    print("🎊 所有处理完成！")


if __name__ == "__main__":
    main()
