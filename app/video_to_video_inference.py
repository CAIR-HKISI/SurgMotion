# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
import yaml
from tqdm import tqdm
import logging

# 添加项目路径
sys.path.append('.')

from src.models.vjepa import init_video_model
from src.utils.checkpoint_loader import robust_checkpoint_loader
from app.vjepa.transforms import make_transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoToVideoInference:
    """视频到视频的推理类"""
    
    def __init__(self, config_path, checkpoint_path, device="cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.config = self.load_config(config_path)
        self.model = self.load_model(checkpoint_path)
        self.transform = self.setup_transforms()
        
    def load_config(self, config_path):
        """加载配置文件"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def load_model(self, checkpoint_path):
        """加载预训练模型"""
        logger.info(f"Loading model from {checkpoint_path}")
        
        # 从配置中获取模型参数
        model_config = self.config.get('model', {})
        data_config = self.config.get('data', {})
        
        # 初始化模型
        encoder, predictor = init_video_model(
            uniform_power=model_config.get('uniform_power', True),
            use_mask_tokens=model_config.get('use_mask_tokens', True),
            num_mask_tokens=10,
            zero_init_mask_tokens=model_config.get('zero_init_mask_tokens', True),
            device=self.device,
            patch_size=data_config.get('patch_size', 16),
            max_num_frames=data_config.get('dataset_fpcs', [16])[0],
            tubelet_size=data_config.get('tubelet_size', 2),
            model_name=model_config.get('model_name', 'vit_huge'),
            crop_size=data_config.get('crop_size', 256),
            pred_depth=model_config.get('pred_depth', 12),
            pred_num_heads=model_config.get('pred_num_heads', 12),
            pred_embed_dim=model_config.get('pred_embed_dim', 384),
            use_sdpa=model_config.get('use_sdpa', True),
            use_silu=model_config.get('use_silu', False),
            use_pred_silu=model_config.get('use_pred_silu', False),
            wide_silu=model_config.get('wide_silu', True),
            use_rope=model_config.get('use_rope', True),
            use_activation_checkpointing=model_config.get('use_activation_checkpointing', True),
        )
        
        # 加载预训练权重
        checkpoint = robust_checkpoint_loader(checkpoint_path, map_location=self.device)
        
        # 加载编码器权重
        if 'encoder' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder'], strict=False)
            logger.info("Loaded encoder weights")
        
        # 加载预测器权重
        if 'predictor' in checkpoint:
            predictor.load_state_dict(checkpoint['predictor'], strict=False)
            logger.info("Loaded predictor weights")
        
        encoder.eval()
        predictor.eval()
        
        return {'encoder': encoder, 'predictor': predictor}
    
    def setup_transforms(self):
        """设置数据变换"""
        data_aug_config = self.config.get('data_aug', {})
        data_config = self.config.get('data', {})
        
        transform = make_transforms(
            random_horizontal_flip=False,  # 推理时不进行随机翻转
            random_resize_aspect_ratio=data_aug_config.get('random_resize_aspect_ratio', [3/4, 4/3]),
            random_resize_scale=data_aug_config.get('random_resize_scale', [0.3, 1.0]),
            reprob=data_aug_config.get('reprob', 0.0),
            auto_augment=data_aug_config.get('auto_augment', False),
            motion_shift=data_aug_config.get('motion_shift', False),
            crop_size=data_config.get('crop_size', 256),
        )
        
        return transform
    
    def load_video(self, video_path, max_frames=None):
        """加载视频文件"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 转换为RGB格式
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                if max_frames and len(frames) >= max_frames:
                    break
        finally:
            cap.release()
        
        if not frames:
            raise ValueError(f"No frames loaded from video: {video_path}")
        
        logger.info(f"Loaded {len(frames)} frames from {video_path}")
        return frames
    
    def preprocess_frames(self, frames, target_size=256):
        """预处理帧"""
        processed_frames = []
        
        for frame in frames:
            # 调整大小
            if frame.shape[:2] != (target_size, target_size):
                frame = cv2.resize(frame, (target_size, target_size))
            
            # 转换为PIL Image格式（transform需要）
            from PIL import Image
            pil_image = Image.fromarray(frame)
            
            # 应用变换
            if self.transform:
                processed_frame = self.transform(pil_image)
            else:
                # 如果没有transform，直接转换为tensor
                processed_frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            
            processed_frames.append(processed_frame)
        
        # 堆叠为 [T, C, H, W] 格式
        video_tensor = torch.stack(processed_frames, dim=0)
        
        # 添加batch维度 [1, T, C, H, W]
        video_tensor = video_tensor.unsqueeze(0)
        
        return video_tensor.to(self.device)
    
    def postprocess_frames(self, output_tensor, original_frames):
        """后处理输出帧"""
        # 移除batch维度
        if output_tensor.dim() == 5:
            output_tensor = output_tensor.squeeze(0)
        
        # 转换为numpy格式
        output_frames = []
        for i in range(output_tensor.shape[0]):
            frame = output_tensor[i].cpu().numpy()
            
            # 如果是 [C, H, W] 格式，转换为 [H, W, C]
            if frame.shape[0] == 3:
                frame = np.transpose(frame, (1, 2, 0))
            
            # 归一化到0-255范围
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
            
            # 确保是RGB格式
            if frame.shape[2] == 3:
                output_frames.append(frame)
            else:
                # 如果是单通道，转换为RGB
                output_frames.append(np.stack([frame[:, :, 0]] * 3, axis=2))
        
        return output_frames
    
    def save_video(self, frames, output_path, fps=30):
        """保存视频"""
        if not frames:
            logger.error("No frames to save")
            return
        
        height, width = frames[0].shape[:2]
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            for frame in frames:
                # 转换为BGR格式（OpenCV需要）
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
        finally:
            out.release()
        
        logger.info(f"Saved video to {output_path}")
    
    def inference(self, input_video_path, output_video_path, max_frames=None):
        """执行视频到视频的推理"""
        logger.info(f"Starting inference: {input_video_path} -> {output_video_path}")
        
        # 1. 加载输入视频
        frames = self.load_video(input_video_path, max_frames)
        
        # 2. 预处理帧
        video_tensor = self.preprocess_frames(frames)
        
        # 3. 模型推理
        with torch.no_grad():
            # 获取编码器特征
            encoder_output = self.model['encoder'](video_tensor)
            
            # 获取预测器输出
            # 注意：这里需要根据实际的mask策略来调整
            # 为了简化，我们直接使用编码器输出
            output_tensor = encoder_output
            
            # 如果输出是列表，取第一个元素
            if isinstance(output_tensor, list):
                output_tensor = output_tensor[0]
        
        # 4. 后处理输出
        output_frames = self.postprocess_frames(output_tensor, frames)
        
        # 5. 保存输出视频
        self.save_video(output_frames, output_video_path)
        
        logger.info("Inference completed successfully")
        
        return output_frames

def main():
    parser = argparse.ArgumentParser(description="Video to Video Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, required=True, help="Path to output video")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    
    args = parser.parse_args()
    
    # 创建推理器
    inference = VideoToVideoInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # 执行推理
    try:
        inference.inference(
            input_video_path=args.input,
            output_video_path=args.output,
            max_frames=args.max_frames
        )
        logger.info("Video to video inference completed successfully!")
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise

if __name__ == "__main__":
    main()
