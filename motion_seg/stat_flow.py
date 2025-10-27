import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from tqdm import tqdm


def analyze_flow_video(video_path, output_dir, bins=100):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mags = []

    for _ in tqdm(range(frame_count), desc=f"读取 {os.path.basename(video_path)}"):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 灰度 ≈ 强度
        mags.extend(gray.flatten())

    cap.release()
    mags = np.array(mags)

    mean_val, var_val = np.mean(mags), np.var(mags)

    # 保存直方图
    plt.figure(figsize=(8, 5))
    plt.hist(mags, bins=bins, density=True, color="blue", alpha=0.7)
    plt.xlabel("灰度值 (近似光流强度)")
    plt.ylabel("概率密度")
    plt.title(f"光流强度分布 (近似): {os.path.basename(video_path)}")
    plt.grid(True, linestyle="--", alpha=0.5)
    hist_path = os.path.join(output_dir, f"{os.path.basename(video_path)}_hist.png")
    plt.savefig(hist_path)
    plt.close()

    return {"Video": os.path.basename(video_path), "Mean": mean_val, "Variance": var_val, "Hist": hist_path}


def main():
    parser = argparse.ArgumentParser(description="基于光流视频近似分析运动强度分布")
    parser.add_argument("--videos", type=str, nargs="+", required=True, help="一个或多个光流视频路径 (flow.mp4)")
    parser.add_argument("--output_dir", type=str, default="analysis_from_video", help="输出目录")
    parser.add_argument("--bins", type=int, default=100, help="直方图分箱数")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for v in args.videos:
        stats = analyze_flow_video(v, args.output_dir, args.bins)
        if stats:
            results.append(stats)
            print(f"🎉 {stats['Video']} -> 均值={stats['Mean']:.4f}, 方差={stats['Variance']:.4f}")

    if results:
        csv_path = os.path.join(args.output_dir, "flow_video_stats.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Video", "Mean", "Variance", "Hist"])
            writer.writeheader()
            writer.writerows(results)
        print(f"📑 统计结果保存到: {csv_path}")


if __name__ == "__main__":
    main()

