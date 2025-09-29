import argparse
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from torchvision.models.optical_flow import (
    raft_large, raft_small,
    Raft_Large_Weights, Raft_Small_Weights
)
from torchvision.utils import flow_to_image


# ==================== 预处理 ====================
def preprocess(img1, img2, weights, device):
    """输入两帧 numpy (H, W, 3)，输出预处理后的 tensor，resize 到 640x480"""
    target_h, target_w = 480, 640
    img1_resized = cv2.resize(img1, (target_w, target_h))
    img2_resized = cv2.resize(img2, (target_w, target_h))

    img1 = torch.from_numpy(img1_resized).permute(2, 0, 1).float() / 255.0
    img2 = torch.from_numpy(img2_resized).permute(2, 0, 1).float() / 255.0
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)

    transforms = weights.transforms()
    img1, img2 = transforms(img1, img2)

    return img1.to(device), img2.to(device)


# ==================== 光流提取并保存 ====================
def estimate_optical_flow(video_path, output_base, model, weights, device, target_fps=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return None

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if target_fps and target_fps < orig_fps:
        frame_interval = int(round(orig_fps / target_fps))
        fps = target_fps
    else:
        frame_interval = 1
        fps = orig_fps

    # 输出目录
    basename = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join(output_base, basename)
    os.makedirs(save_dir, exist_ok=True)
    flow_path = os.path.join(save_dir, f"{basename}_flow.mp4")
    comparison_path = os.path.join(save_dir, f"{basename}_comparison.mp4")
    flow_npy_dir = os.path.join(save_dir, f"{basename}_npy")
    os.makedirs(flow_npy_dir, exist_ok=True)

    # 输出视频 writer
    output_width, output_height = 640, 480
    writer_flow = cv2.VideoWriter(flow_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (output_width, output_height))
    writer_cmp = cv2.VideoWriter(comparison_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (output_width * 2, output_height))

    # 第一帧
    ret, prev_frame = cap.read()
    if not ret:
        print("❌ 空视频")
        return None

    frame_count = 0
    with tqdm(total=total_frames, desc=f"处理 {os.path.basename(video_path)}") as pbar:
        while True:
            for _ in range(frame_interval - 1):
                cap.read()
                pbar.update(1)

            ret, curr_frame = cap.read()
            if not ret:
                break
            frame_count += 1
            pbar.update(1)

            img1, img2 = preprocess(prev_frame, curr_frame, weights, device)
            with torch.no_grad():
                flow_up = model(img1, img2)[-1]  # (1, 2, H, W)

            # 保存可视化视频
            flow_img = flow_to_image(flow_up)[0].permute(1, 2, 0).cpu().numpy()
            writer_flow.write(flow_img)

            # 保存真实光流 .npy
            flow_np = flow_up[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 2)
            np.save(os.path.join(flow_npy_dir, f"{basename}_{frame_count:06d}.npy"), flow_np)

            # 保存对比视频
            curr_frame_resized = cv2.resize(curr_frame, (output_width, output_height))
            combined = np.hstack([curr_frame_resized, flow_img])
            writer_cmp.write(combined)

            prev_frame = curr_frame

    cap.release()
    writer_flow.release()
    writer_cmp.release()

    print(f"✅ 光流可视化保存: {flow_path}")
    print(f"✅ 光流数据保存: {flow_npy_dir}")
    print(f"✅ 对比保存: {comparison_path}")
    print(f"🎉 总处理帧数: {frame_count}")

    return {
        "flow_video": os.path.abspath(flow_path),
        "flow_npy_dir": os.path.abspath(flow_npy_dir),
        "comparison_video": os.path.abspath(comparison_path),
        "fps": fps
    }


# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser(description="提取光流并保存 (可视化 + 对比 + .npy)")
    parser.add_argument("--video", type=str, required=True, help="输入视频路径")
    parser.add_argument("--output_dir", type=str, default="logs", help="输出目录")
    parser.add_argument("--model", type=str, default="raft_large", choices=["raft_small", "raft_large"], help="RAFT 模型类型")
    parser.add_argument("--target_fps", type=int, default=None, help="输出视频帧率（可选）")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")
    print(f"🤖 使用模型: {args.model}")

    # 加载模型
    print("⏳ 加载模型中...")
    if args.model == "raft_large":
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights).to(device).eval()
    else:
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights).to(device).eval()
    print("✅ 模型加载完成！")

    # 提取光流
    estimate_optical_flow(args.video, args.output_dir, model, weights, device, args.target_fps)


if __name__ == "__main__":
    main()

