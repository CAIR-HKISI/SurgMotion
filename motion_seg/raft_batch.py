import argparse
import os
import cv2
import torch
import numpy as np
import csv
from tqdm import tqdm
from torchvision.models.optical_flow import (
    raft_large, raft_small,
    Raft_Large_Weights, Raft_Small_Weights
)
from torchvision.utils import flow_to_image


def preprocess(img1, img2, weights, device):
    img1 = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
    img2 = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)
    transforms = weights.transforms()
    img1, img2 = transforms(img1, img2)
    return img1.to(device), img2.to(device)


def estimate_optical_flow(video_path, output_base, model, weights, device, target_fps=None, input_dir=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return None

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if target_fps and target_fps < orig_fps:
        frame_interval = int(round(orig_fps / target_fps))
        fps = target_fps
    else:
        frame_interval = 1
        fps = orig_fps

    # 保持目录结构
    if input_dir:
        rel_path = os.path.relpath(video_path, input_dir)
        rel_dir = os.path.dirname(rel_path)
    else:
        rel_dir = ""
    save_dir = os.path.join(output_base, rel_dir)

    basename = os.path.splitext(os.path.basename(video_path))[0]
    flow_path = os.path.join(save_dir, f"{basename}_flow.mp4")
    comparison_path = os.path.join(save_dir, f"{basename}_comparison.mp4")
    os.makedirs(save_dir, exist_ok=True)

    writer_flow = cv2.VideoWriter(flow_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    writer_cmp = cv2.VideoWriter(comparison_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * 2, height))

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
                flow_up = model(img1, img2)[-1]

            flow_img = flow_to_image(flow_up)[0].permute(1, 2, 0).cpu().numpy()
            flow_resized = cv2.resize(flow_img, (width, height))

            writer_flow.write(flow_resized)
            combined = np.hstack([curr_frame, flow_resized])
            writer_cmp.write(combined)
            prev_frame = curr_frame

    cap.release()
    writer_flow.release()
    writer_cmp.release()

    print(f"✅ 光流保存: {flow_path}")
    print(f"✅ 对比保存: {comparison_path}")
    print(f"🎉 总处理帧数: {frame_count}")

    return {
        "original_video": os.path.abspath(video_path),
        "flow_video": os.path.abspath(flow_path),
        "comparison_video": os.path.abspath(comparison_path),
        "fps": fps
    }


def main():
    parser = argparse.ArgumentParser(description="使用 TorchVision RAFT 批量处理视频光流")
    parser.add_argument("--video", type=str, help="单个输入视频路径")
    parser.add_argument("--input_dir", type=str, help="输入视频文件夹")
    parser.add_argument("--output_dir", type=str, default="logs", help="输出目录")
    parser.add_argument("--model", type=str, default="raft_large", choices=["raft_small", "raft_large"], help="RAFT 模型类型")
    parser.add_argument("--target_fps", type=int, default=None, help="输出视频帧率（可选）")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")
    print(f"🤖 使用模型: {args.model}")

    print("⏳ 加载模型中...")
    if args.model == "raft_large":
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights).to(device).eval()
    else:
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights).to(device).eval()
    print("✅ 模型加载完成！")

    video_list = []
    if args.video:
        video_list.append(args.video)
        input_dir = None
    elif args.input_dir:
        input_dir = args.input_dir
        for root, _, files in os.walk(input_dir):
            for fname in files:
                if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    video_list.append(os.path.join(root, fname))
    else:
        print("❌ 需要指定 --video 或 --input_dir")
        return

    records = []
    print("=" * 60)
    for vpath in video_list:
        record = estimate_optical_flow(vpath, args.output_dir, model, weights, device,
                                       target_fps=args.target_fps, input_dir=args.input_dir)
        if record:
            records.append(record)
        print("-" * 60)

    if args.input_dir and records:
        csv_path = os.path.join(args.output_dir, "summary.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        print(f"📑 CSV 已保存: {csv_path}")

    print("🎊 全部视频处理完成！")


if __name__ == "__main__":
    main()