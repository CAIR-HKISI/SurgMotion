import cv2
import numpy as np
import pandas as pd
import os
import random
import glob
import torch
import torch.nn.functional as F

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_motion_target_heatmap(clips):
    """
    计算 Motion Heatmap，参考 app/vjepa_pred/train.py 中的逻辑。
    返回未下采样的全分辨率 Heatmap (B, T, H, W)
    """
    # clips: [B, C, T, H, W], 已经归一化到 [0, 1]
    
    B, C, T, H, W = clips.shape
    
    # 1. 时间梯度 (I_t): 帧差 [B, C, T, H, W]
    diffs = torch.abs(clips[:, :, 1:] - clips[:, :, :-1])
    diffs = torch.cat([diffs[:, :, :1], diffs], dim=2) # 补齐时间维
    
    # 2. 空间梯度 (I_x, I_y): 纹理强度 [B, C, T, H, W]
    # 计算 x 方向梯度: I(x+1) - I(x)
    dx = clips[:, :, :, :, 1:] - clips[:, :, :, :, :-1]
    dx = torch.cat([dx, dx[:, :, :, :, -1:]], dim=4) # Padding
    
    # 计算 y 方向梯度: I(y+1) - I(y)
    dy = clips[:, :, :, 1:, :] - clips[:, :, :, :-1, :]
    dy = torch.cat([dy, dy[:, :, :, -1:, :]], dim=3) # Padding
    
    # 空间梯度模长 (Texture Strength)
    spatial_grad = torch.sqrt(dx**2 + dy**2 + 1e-6)
    
    # 3. 计算物理速度 (Speed)
    alpha = 0.05 
    speed_map = diffs / (spatial_grad + alpha)
    
    # 过滤低纹理区域
    valid_texture = spatial_grad > 0.01
    speed_map = speed_map * valid_texture.float()
    
    # --- Global Motion Estimation & Removal ---
    # 计算每帧的全局运动基准 (Median over H, W)
    flat_speed = speed_map.view(B, C, T, -1) # [B, C, T, H*W]
    global_motion_est = torch.quantile(flat_speed, 0.5, dim=-1, keepdim=True) # [B, C, T, 1]
    global_motion_est = global_motion_est.view(B, C, T, 1, 1)
    
    # Global Base Compression
    global_base_map = torch.tanh(global_motion_est * 5.0).clamp(max=0.3)
    global_base_map = global_base_map.expand_as(speed_map)
    
    # Local Motion Compression
    local_motion_map = torch.relu(speed_map - (global_motion_est * 1.2))
    local_motion_map = torch.tanh(local_motion_map * 5.0) * 0.7
    
    # 合成 Target
    combined_map = local_motion_map + global_base_map
    
    # 使用 combined_map 进行后续的平均
    combined_map = combined_map.mean(dim=1, keepdim=False) # [B, T, H, W]

    return combined_map

def apply_heatmap_overlay(img_bgr, heatmap_val, alpha=0.5):
    """
    将单通道 heatmap 值 (0~1) 叠加到 img_bgr 上
    """
    # 映射 heatmap 到 0-255 并转为 uint8
    heatmap_norm = (heatmap_val * 255).clip(0, 255).astype(np.uint8)
    
    # 应用颜色映射
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    
    # 叠加
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

def main():
    # 配置文件路径
    csv_path = 'data/Surge_Frames/AIxsuture/clips_16f/train_dense_16f_detailed.csv'
    output_dir = 'vis_results_jepa_motion'
    sample_count = 5  # 随机采样的数量
    
    ensure_dir(output_dir)
    
    print(f"正在读取CSV文件: {csv_path} ...")
    if not os.path.exists(csv_path):
        print(f"错误: 找不到文件 {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # 随机采样
    samples = df.sample(n=min(sample_count, len(df)))
    print(f"已采样 {len(samples)} 个案例，开始处理...")
    
    for idx, row in samples.iterrows():
        clip_info_path = row['clip_path']
        case_id = row['case_id']
        clip_idx = row['clip_idx']
        
        # 读取包含图片路径的txt文件
        if not os.path.exists(clip_info_path):
            print(f"警告: 找不到Clip信息文件 {clip_info_path}，跳过。")
            continue
            
        with open(clip_info_path, 'r') as f:
            frame_paths = [line.strip() for line in f.readlines()]
            
        if len(frame_paths) == 0:
            print(f"警告: Clip {clip_info_path} 为空。")
            continue
            
        print(f"处理 Case {case_id} Clip {clip_idx} ({len(frame_paths)} frames)...")
        
        # 创建该case的输出目录
        case_out_dir = os.path.join(output_dir, f"case{case_id}_clip{clip_idx}")
        ensure_dir(case_out_dir)
        
        # 加载所有图片并准备为 Tensor
        # 模型期望输入: [B, C, T, H, W]，归一化到 [0, 1]
        imgs = []
        original_imgs = []
        
        for p in frame_paths:
            if not os.path.exists(p):
                # 简单填充上一帧或黑帧
                if len(imgs) > 0:
                    img = imgs[-1].copy()
                else:
                    img = np.zeros((224, 224, 3), dtype=np.uint8) # 假设尺寸，后面resize
            else:
                img = cv2.imread(p)
                if img is None:
                    continue
            
            original_imgs.append(img)
            # 转 RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img_rgb)
        
        if not imgs:
            continue

        # 转换为 Tensor
        # Stack -> [T, H, W, C]
        clip_np = np.stack(imgs)
        # Normalize to [0, 1]
        clip_np = clip_np.astype(np.float32) / 255.0
        # To Torch: [T, H, W, C] -> [T, C, H, W] -> [1, C, T, H, W]
        clip_tensor = torch.from_numpy(clip_np).permute(3, 0, 1, 2).unsqueeze(0)
        
        # 计算 Motion Heatmap
        with torch.no_grad():
            # heatmap: [B, T, H, W] -> [1, T, H, W]
            heatmap_tensor = get_motion_target_heatmap(clip_tensor)
            heatmap_np = heatmap_tensor.squeeze(0).cpu().numpy() # [T, H, W]

        # 保存可视化结果
        vis_files = []
        for i in range(len(original_imgs)):
            orig_img = original_imgs[i]
            heatmap_val = heatmap_np[i]
            
            # heatmap 可能和原图尺寸不一致(如果之前resize过)，这里确保一致
            # heatmap_val is H, W from clip_tensor, which is from imgs
            # So sizes should match unless original images varied in size (unlikely for a clip)
            
            overlay = apply_heatmap_overlay(orig_img, heatmap_val, alpha=0.6)
            
            save_name = f"motion_jepa_{i:03d}.jpg"
            save_path = os.path.join(case_out_dir, save_name)
            cv2.imwrite(save_path, overlay)
            vis_files.append(save_path)
            
        print(f"结果已保存至: {case_out_dir}")
        
        # 生成GIF
        try:
            import imageio
            gif_images = []
            for filename in vis_files:
                gif_images.append(imageio.imread(filename))
            if gif_images:
                imageio.mimsave(os.path.join(case_out_dir, 'motion_heatmap.gif'), gif_images, fps=5)
        except ImportError:
            pass
        except Exception as e:
            print(f"生成GIF出错: {e}")

    print("全部处理完成。")

if __name__ == "__main__":
    main()
