#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_and_extract.py
-------------------------------------
1. 遍历 data/Landscopy/[name]_dataset/*/*.mp4
2. 将视频复制并重命名为连续数字：
   data/Landscopy/[name]_dataset_renumbered/{子文件夹}/{00001.mp4, 00002.mp4, ...}
3. 使用更兼容的 ffmpeg 参数进行抽帧。
"""

import subprocess
from pathlib import Path
import shutil
from tqdm import tqdm
import argparse


# 🧩 Step 1: 拷贝并重命名视频
def clean_videos(
    src_root="data/Landscopy/GynSurg_Action_Segments",
    dst_root="data/Landscopy/GynSurg_Action_Segments_Clean"
):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    video_files = list(src_root.rglob("*.mp4"))
    video_files.sort()

    print(f"🎥 检测到 {len(video_files)} 个视频文件，开始拷贝并重命名...")

    # 按子目录分别处理
    for folder in sorted({v.parent for v in video_files}):
        rel = folder.relative_to(src_root)
        out_subdir = dst_root / rel
        out_subdir.mkdir(parents=True, exist_ok=True)

        vids = sorted(folder.glob("*.mp4"))
        for idx, vid in enumerate(vids, start=1):
            new_name = f"{idx:05d}.mp4"
            new_path = out_subdir / new_name
            shutil.copy2(vid, new_path)
        print(f"✅ {rel}: {len(vids)} 个视频已复制重命名。")

    print("🎉 所有视频文件已复制并按数字命名。")
    return dst_root


# 🧩 Step 2: 抽帧
def videos_to_frames(input_path,
                     output_path,
                     fps=30,
                     pattern="*.mp4",
                     debug=False,
                     save_failed=True):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    video_files = list(input_path.rglob(pattern))
    video_files.sort()

    if not video_files:
        print(f"⚠️ 未在 {input_path} 下找到匹配 {pattern} 的视频文件。")
        return

    print(f"\n🎞️ 共检测 {len(video_files)} 个视频，开始抽帧...\n")

    failed_videos = []

    for i, vid_path in enumerate(tqdm(video_files, desc="Extracting frames")):
        rel_path = vid_path.relative_to(input_path).parent
        out_folder = output_path / rel_path / vid_path.stem
        out_folder.mkdir(parents=True, exist_ok=True)
        output_pattern = out_folder / f"{vid_path.stem}_%05d.jpg"

        # ✅ 全兼容版 FFmpeg 命令
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i", str(vid_path.resolve()),   # 输入文件要放在 -safe 之前！
            "-safe", "0",
            "-vf", f"fps={fps},scale=512:-1:flags=bicubic",
            "-vsync", "2",
            "-qscale:v", "2",
            str(output_pattern)
        ]

        if debug:
            print("🔍 FFmpeg 命令:", " ".join(ffmpeg_cmd))

        try:
            result = subprocess.run(
                ffmpeg_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # 调试输出（前200字符）
            if debug:
                print(result.stderr.decode("utf-8", errors="ignore")[:200])

        except subprocess.CalledProcessError as e:
            log = e.stderr.decode("utf-8", errors="ignore")
            print(f"\n❌ 抽帧失败: {vid_path}")
            if "Invalid data found" in log:
                print("⚠️ 视频损坏或无法解析")
            elif "moov atom not found" in log:
                print("⚠️ 视频文件不完整（缺少索引）")
            elif "Error while opening filter" in log:
                print("⚠️ 滤镜错误，请检查视频宽高")
            else:
                if debug:
                    print("详细错误:\n", log[:400])
            failed_videos.append(str(vid_path))
            continue

    print("\n🎉 抽帧任务完成。")

    # 保存失败列表
    if save_failed and failed_videos:
        fail_log = output_path / "failed_videos.txt"
        with fail_log.open("w", encoding="utf-8", errors="ignore") as f:
            f.write("\n".join(failed_videos))
        print(f"⚠️ 共 {len(failed_videos)} 个视频抽帧失败，详情见: {fail_log}")


# 🧩 Step 3: 主逻辑
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GynSurg dataset clean and frame extraction")
    parser.add_argument('--name', type=str, default='GynSurg_action', help='数据集名 (例如 GynSurg_action)')
    parser.add_argument('--fps', type=int, default=30, help='抽帧帧率')
    parser.add_argument('--debug', action='store_true', help='打印调试信息')
    args = parser.parse_args()

    name = args.name
    SRC = f"data/Landscopy/{name}_dataset"
    DST_CLEAN = f"data/Landscopy/{name}_dataset_renumbered"
    DST_FRAMES = f"data/Surge_Frames/{name}/frames"

    # ✅ ① 拷贝并重命名视频
    clean_videos(SRC, DST_CLEAN)

    # ✅ ② 执行抽帧
    videos_to_frames(
        input_path=DST_CLEAN,
        output_path=DST_FRAMES,
        fps=args.fps,
        debug=args.debug
    )

