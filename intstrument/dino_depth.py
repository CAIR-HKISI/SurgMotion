# ## 用Depth Anything模型，对pitvis的frame 进行预测，并保存为depth map

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# import torchvision.transforms.functional as TF
# import cv2
# import numpy as np
# from PIL import Image
# import os
# from pathlib import Path

# # 加载Depth Anything模型
# from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# # 加载模型和处理器
# model_name = "LiheYoung/depth_anything_vitl14"
# processor = AutoImageProcessor.from_pretrained(model_name)
# model = AutoModelForDepthEstimation.from_pretrained(model_name)
# model.eval()
# model.to('cuda')

# # 图像预处理
# def preprocess_image(image_path, target_size=(518, 518)):
#     """预处理图像用于Depth Anything模型"""
#     # 读取图像
#     if isinstance(image_path, str):
#         image = Image.open(image_path).convert('RGB')
#     else:
#         image = image_path
    
#     # 调整图像大小
#     image = image.resize(target_size, Image.Resampling.LANCZOS)
    
#     return image

# def predict_depth(image):
#     """使用Depth Anything模型预测深度"""
#     with torch.no_grad():
#         # 使用处理器预处理图像
#         inputs = processor(images=image, return_tensors="pt")
#         inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
#         # 预测深度
#         outputs = model(**inputs)
#         predicted_depth = outputs.predicted_depth
        
#         # 后处理深度图
#         depth_map = torch.nn.functional.interpolate(
#             predicted_depth.unsqueeze(1),
#             size=image.size[::-1],  # (height, width)
#             mode="bicubic",
#             align_corners=False,
#         ).squeeze()
        
#         return depth_map

# def save_depth_map(depth_map, output_path, original_size=None):
#     """保存深度图为图像文件"""
#     # 转换为numpy数组
#     depth_np = depth_map.cpu().numpy()[0]  # 移除batch维度
    
#     # 转换为0-255范围
#     depth_np = (depth_np * 255).astype(np.uint8)
    
#     # 如果需要，调整回原始尺寸
#     if original_size:
#         depth_np = cv2.resize(depth_np, original_size, interpolation=cv2.INTER_LINEAR)
    
#     # 保存为图像
#     cv2.imwrite(output_path, depth_np)
#     print(f"深度图已保存到: {output_path}")

# def process_single_image(image_path, output_dir="./depth_maps"):
#     """处理单张图像"""
#     # 创建输出目录
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 获取原始图像尺寸
#     original_image = Image.open(image_path)
#     original_size = original_image.size[::-1]  # (height, width)
    
#     # 预处理图像
#     image = preprocess_image(image_path)
    
#     # 预测深度
#     depth_map = predict_depth(image)
    
#     # 生成输出文件名
#     image_name = Path(image_path).stem
#     output_path = os.path.join(output_dir, f"{image_name}_depth.png")
    
#     # 保存深度图
#     save_depth_map(depth_map, output_path, original_size)
    
#     return output_path

# def process_video_frames(video_path, output_dir="./depth_maps", frame_interval=1):
#     """处理视频的所有帧"""
#     cap = cv2.VideoCapture(video_path)
#     frame_count = 0
#     processed_count = 0
    
#     os.makedirs(output_dir, exist_ok=True)
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         if frame_count % frame_interval == 0:
#             # 转换BGR到RGB
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame_pil = Image.fromarray(frame_rgb)
            
#             # 预处理
#             image = preprocess_image(frame_pil)
            
#             # 预测深度
#             depth_map = predict_depth(image)
            
#             # 保存深度图
#             output_path = os.path.join(output_dir, f"frame_{frame_count:06d}_depth.png")
#             save_depth_map(depth_map, output_path, frame.shape[:2])
            
#             processed_count += 1
#             print(f"已处理第 {frame_count} 帧")
        
#         frame_count += 1
    
#     cap.release()
#     print(f"总共处理了 {processed_count} 帧")

# # 示例使用
# if __name__ == "__main__":
#     # 处理指定目录下的所有图像
#     import glob
    
#     dirrs = "/data/wjl/vjepa2/data/pitvis/video_01/"
    
#     # 支持的图像格式
#     image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    
#     # 获取所有图像文件
#     image_files = []
#     for ext in image_extensions:
#         image_files.extend(glob.glob(os.path.join(dirrs, ext)))
#         image_files.extend(glob.glob(os.path.join(dirrs, ext.upper())))
    
#     print(f"找到 {len(image_files)} 个图像文件")
    
#     # 处理每个图像
#     for i, image_path in enumerate(image_files):
#         try:
#             print(f"正在处理第 {i+1}/{len(image_files)} 个图像: {os.path.basename(image_path)}")
#             process_single_image(image_path, output_dir="./depth_maps")
#         except Exception as e:
#             print(f"处理图像 {image_path} 时出错: {e}")
#             continue
            
#     print("所有图像处理完成！")
    




from transformers import pipeline
from PIL import Image

img1 = "/data/wjl/vjepa2/data/pitvis/video_03/frame_001070.jpg" # 有器械
img2 = "/data/wjl/vjepa2/data/pitvis/video_03/frame_001342.jpg" # 没有器械

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
image = Image.open('your/image/path')
depth = pipe(image)["depth"]


