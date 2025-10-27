import os
import argparse
from tqdm import tqdm
from pathlib import Path


def videos_to_imgs(input_path="/Videos/input",
                   output_path="/Videos/output",
                   fps=1,
                   pattern="**/*.mp4"):
    """
    Converts videos into individual frames using ffmpeg.
    Each video will have its own folder under the mirrored structure.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # 匹配符合 pattern 的视频文件（支持递归搜索）
    dirs = list(input_path.glob(pattern))
    dirs.sort()

    print(f"🎥 Found {len(dirs)} video(s) under {input_path}")

    for i, vid_path in enumerate(tqdm(dirs, desc="Extracting frames")):
        file_name = vid_path.stem  # 不带扩展名的文件名

        # ✅ 获取相对于输入根目录的路径层次（不含输入根目录本身）
        rel_path = vid_path.relative_to(input_path).parent

        # ✅ 每个视频一个文件夹，保持上层目录结构
        out_folder = output_path / rel_path / file_name
        out_folder.mkdir(parents=True, exist_ok=True)

        # ✅ 输出文件名前缀，例如：00000001.jpg
        output_pattern = out_folder / f"{file_name}_%08d.jpg"

        # ✅ 调用 ffmpeg 提取帧并缩放
        ffmpeg_cmd = (
            f'ffmpeg -y -i "{vid_path}" '
            f'-vf "fps={fps},scale=\'if(gte(iw,ih),512,-1)\':\'if(gte(ih,iw),512,-1)\':'
            f'force_original_aspect_ratio=decrease" "{output_pattern}"'
        )
        os.system(ffmpeg_cmd)

        print(f"✅ [{i+1}/{len(dirs)}] Done extracting frames from: {vid_path.name}")

    print("🎉 All videos processed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames from videos (each video gets its own folder)."
    )
    parser.add_argument(
        "--video_path",
        required=True,
        type=str,
        help="Path to input video directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Path to output directory for extracted frames.",
    )
    parser.add_argument(
        "--fps",
        required=False,
        type=int,
        default=1,
        help="Frames per second for extraction (default: 1)",
    )
    parser.add_argument(
        "--pattern",
        required=False,
        type=str,
        default="*.mp4",
        help="Glob pattern for locating video files (default: '**/*.mp4'). Example: '*.avi'",
    )

    args = parser.parse_args()

    videos_to_imgs(
        input_path=args.video_path,
        output_path=args.output,
        fps=args.fps,
        pattern=args.pattern
    )