import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from tqdm import tqdm


def sample_video_frames(video_path, fps_target=25):
    """从视频中按1fps抽取灰度像素，返回展平数组"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # 容错

    step = int(round(fps / fps_target))  # 每秒取1帧
    mags = []

    for i in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 灰度 ≈ 光流强度
        mags.extend(gray.flatten())

    cap.release()
    return np.array(mags)


def analyze_video(video_path, output_dir, bins=100):
    """统计单个视频，返回像素分布特征"""
    mags = sample_video_frames(video_path, fps_target=1)
    if mags is None or len(mags) == 0:
        return None

    mean_val, var_val = np.mean(mags), np.var(mags)

    # 单视频直方图
    plt.figure(figsize=(8, 5))
    plt.hist(mags, bins=bins, density=True, color="blue", alpha=0.7)
    plt.xlabel("灰度值 (近似光流强度)")
    plt.ylabel("概率密度")
    plt.title(f"光流强度分布 (1fps) — {os.path.basename(video_path)}")
    plt.grid(True, linestyle="--", alpha=0.5)
    hist_path = os.path.join(output_dir, f"{os.path.basename(video_path)}_hist.png")
    plt.savefig(hist_path)
    plt.close()

    return {"Video": os.path.basename(video_path), "Mean": mean_val, "Variance": var_val, "Mags": mags, "Hist": hist_path}


def main():
    parser = argparse.ArgumentParser(description="多个视频的光流统计 (1fps)")
    parser.add_argument("--videos", type=str, nargs="+", required=True, help="多个视频文件路径 (每个视频作为一个数据集)")
    parser.add_argument("--output_dir", type=str, default="analysis_from_videos", help="输出目录")
    parser.add_argument("--bins", type=int, default=100, help="直方图分箱数")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    for v in args.videos:
        stats = analyze_video(v, args.output_dir, args.bins)
        if stats:
            results.append(stats)
            print(f"🎉 {stats['Video']} -> 均值={stats['Mean']:.4f}, 方差={stats['Variance']:.4f}")

    # 保存 CSV
    if results:
        csv_path = os.path.join(args.output_dir, "videos_flow_stats.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Video", "Mean", "Variance", "Hist"])
            writer.writeheader()
            for r in results:
                writer.writerow({k: r[k] for k in ["Video", "Mean", "Variance", "Hist"]})
        print(f"📑 统计结果保存到: {csv_path}")

        # 绘制对比图
        plt.figure(figsize=(10, 6))
        for r in results:
            plt.hist(r["Mags"], bins=args.bins, density=True, alpha=0.5, label=r["Video"])
        plt.xlabel("灰度值 (近似光流强度)")
        plt.ylabel("概率密度")
        plt.title("各视频光流强度分布对比 (1fps)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        compare_path = os.path.join(args.output_dir, "videos_comparison.png")
        plt.savefig(compare_path)
        plt.close()
        print(f"📊 分布对比图保存到: {compare_path}")


if __name__ == "__main__":
    main()

