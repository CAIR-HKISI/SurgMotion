import os, cv2
import numpy as np
import warnings
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from crop_black import black_crop

input_dir = "/mnt/tqy/AutoLaparo/AutoLaparo_Task1/frames/"
output_dir = "/mnt/tqy/AutoLaparo/AutoLaparo_Task1/frames_resized/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_image(image_path, output_folder, image_file):
    try:
        x, y, w, h = None, None, None, None
        with Image.open(image_path) as img:
            # width, height = img.size
            img = np.array(img)
            if x is None:
                x, y, w, h = black_crop(img)
            img = img[y:y+h, x:x+w]
            img = Image.fromarray(img)

            new_width = 480
            new_height = 270

            img_resized = img.resize((new_width, new_height))

            if img_resized.mode == "RGBA":
                img_resized = img_resized.convert("RGB")

            output_image_path = os.path.join(output_folder, image_file.replace(".png", ".jpg"))
            img_resized.save(output_image_path, "JPEG")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def process_folder(folder):
    folder_path = os.path.join(input_dir, folder)
    
    if os.path.isdir(folder_path):
        output_folder = os.path.join(output_dir, folder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        image_files = [image_file for image_file in os.listdir(folder_path) if image_file.endswith(".png")]
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            for _ in tqdm(executor.map(lambda image_file: process_image(
                os.path.join(folder_path, image_file), output_folder, image_file), image_files), 
                total=len(image_files), desc=f"Processing {folder}"):
                pass

def main():
    folders = os.listdir(input_dir)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        for folder in tqdm(folders, desc="Processing"):
            if os.path.isdir(os.path.join(input_dir, folder)):
                executor.submit(process_folder, folder)

    print("Done！")

if __name__ == "__main__":
    main()
