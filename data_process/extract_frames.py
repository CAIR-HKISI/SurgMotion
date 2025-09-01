import os
import argparse
from tqdm import tqdm
from pathlib import Path


def videos_to_imgs(input_path="/Videos/input",
                   output_path="/Videos/output",
                   fps=1,
                   pattern="*/*.mp4"):
    
    output_path = Path(output_path)
    input_path = Path(input_path)

    output_path.mkdir(exist_ok=True)

    dirs = list(input_path.glob(pattern))
    dirs.sort()

    for i, vid_path in enumerate(tqdm(dirs)): 
        file_name = vid_path.stem 
        out_folder = output_path / file_name.split('-')[0]
        out_folder.mkdir(exist_ok=True)

        # os.system(command) method executes the command (a string) in a subshell; here, the command converts each video into images, by filtering
        os.system(f'ffmpeg -i {vid_path} -vf "fps={fps}" {out_folder/file_name}_%08d.jpg') 
        print("Done extracting: {}".format(i + 1))

if __name__ == "__main__":
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Utility to download annotations on DONE videos in MOSaiC ext stack"
    )
    parser.add_argument(
        "--video_path",
        required=True,
        type=str,
        help="Define input video dir",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Define output dir to save frames extracted",
    )
    parser.add_argument(
        "--fps",
        required=False,
        type=int,
        default=1,
        help="Define fps for frames extraction",
    )
    arguments = parser.parse_args()

    videos_to_imgs(input_path=arguments.video_path, output_path=arguments.output, fps=arguments.fps)

