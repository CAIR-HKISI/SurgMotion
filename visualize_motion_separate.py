import cv2
import numpy as np
import pandas as pd
import os
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
    combined_map = combined_map.mean(dim=1, keepdim=True) # [B, 1, T, H, W]

    # --- Downsample & Smooth (Simulate Model Target) ---
    patch_size = 16
    tubelet_size = 2
    sigma = 1.0
    
    # Grid sizes
    t_grid = T // tubelet_size
    h_grid = H // patch_size
    w_grid = W // patch_size
    
    # Downsample
    motion_target = F.adaptive_avg_pool3d(
        combined_map, 
        output_size=(t_grid, h_grid, w_grid)
    ) # [B, 1, t_grid, h_grid, w_grid]
    
    # Gaussian Smoothing
    if sigma > 0:
        k_size = int(4 * sigma + 1)
        if k_size % 2 == 0:
            k_size += 1

        # Create Gaussian kernel
        x = torch.arange(k_size, device=clips.device, dtype=clips.dtype) - (k_size - 1) / 2
        k = torch.exp(-0.5 * (x / sigma)**2)
        k = k / k.sum()

        k_3d = k[:, None, None] * k[None, :, None] * k[None, None, :]
        k_3d = k_3d[None, None, ...] # [1, 1, k, k, k]
        
        # Padding
        padding = k_size // 2
        motion_target = F.conv3d(motion_target, k_3d, padding=padding)

    # Upsample back to original size for visualization
    # Trilinear interpolation
    heatmap_high_res = F.interpolate(motion_target, size=(T, H, W), mode='trilinear', align_corners=False)
    
    # 6. 归一化到 [0, 1]
    heatmap_high_res = heatmap_high_res.clamp(0.0, 1.0)
    
    return heatmap_high_res.squeeze(1) # [B, T, H, W]

def get_top3_patch_soft_heatmap(clips, patch_size=16, top_k=3, gaussian_sigma=2.0, tubelet_size=2):
    """
    计算基于 patch 的 top-3 运动区域（时空立方体），以 tubelet 为单位计算和显示。
    每个 tubelet（2帧）内找 top-3 的空间位置，同一 tubelet 内的帧共享相同的位置。
    
    Args:
        clips: [B, C, T, H, W], 已经归一化到 [0, 1]
        patch_size: patch 大小，默认 16
        top_k: 取运动最大的前 k 个 patch，默认 3
        gaussian_sigma: 高斯分布的 sigma（以 patch 为单位），默认 2.0
        tubelet_size: 时间维度的 tube 大小，默认 2
    
    Returns:
        soft_heatmap: [B, T, H, W] soft attention map (同一 tubelet 内共享相同的空间位置)
    """
    B, C, T, H, W = clips.shape
    
    # 1. 计算帧差 (运动强度)
    diffs = torch.abs(clips[:, :, 1:] - clips[:, :, :-1])
    diffs = torch.cat([diffs[:, :, :1], diffs], dim=2)  # 补齐时间维
    
    # 2. 计算空间梯度
    dx = clips[:, :, :, :, 1:] - clips[:, :, :, :, :-1]
    dx = torch.cat([dx, dx[:, :, :, :, -1:]], dim=4)
    dy = clips[:, :, :, 1:, :] - clips[:, :, :, :-1, :]
    dy = torch.cat([dy, dy[:, :, :, -1:, :]], dim=3)
    spatial_grad = torch.sqrt(dx**2 + dy**2 + 1e-6)
    
    # 3. 计算物理速度
    alpha = 0.05
    speed_map = diffs / (spatial_grad + alpha)
    valid_texture = spatial_grad > 0.01
    speed_map = speed_map * valid_texture.float()
    
    # 4. 全局运动移除
    flat_speed = speed_map.view(B, C, T, -1)
    global_motion_est = torch.quantile(flat_speed, 0.5, dim=-1, keepdim=True)
    global_motion_est = global_motion_est.view(B, C, T, 1, 1)
    local_motion_map = torch.relu(speed_map - (global_motion_est * 1.2))
    
    # 5. 平均通道
    motion_map = local_motion_map.mean(dim=1)  # [B, T, H, W]
    
    # 6. 下采样到 patch 级别（使用 adaptive_avg_pool 支持任意尺寸）
    h_grid = (H + patch_size - 1) // patch_size  # 向上取整
    w_grid = (W + patch_size - 1) // patch_size
    
    # 使用 adaptive_avg_pool2d: [B, T, H, W] -> [B*T, 1, H, W] -> pool -> [B, T, h_grid, w_grid]
    motion_map_flat = motion_map.view(B * T, 1, H, W)
    patch_motion_flat = F.adaptive_avg_pool2d(motion_map_flat, (h_grid, w_grid))
    patch_motion = patch_motion_flat.view(B, T, h_grid, w_grid)
    
    # 7. 按 tubelet 分组：每 tubelet_size 帧为一组
    num_tubelets = T // tubelet_size
    # [B, T, h_grid, w_grid] -> [B, num_tubelets, tubelet_size, h_grid, w_grid]
    patch_motion_tubelets = patch_motion.view(B, num_tubelets, tubelet_size, h_grid, w_grid)
    
    # 8. 每个 tubelet 内累加运动量
    # [B, num_tubelets, tubelet_size, h_grid, w_grid] -> [B, num_tubelets, h_grid, w_grid]
    tubelet_patch_motion = patch_motion_tubelets.sum(dim=2)
    
    # 9. 创建坐标网格 (patch 级别)
    grid_y = torch.arange(h_grid, device=clips.device, dtype=clips.dtype)
    grid_x = torch.arange(w_grid, device=clips.device, dtype=clips.dtype)
    grid_yy, grid_xx = torch.meshgrid(grid_y, grid_x, indexing='ij')  # [h_grid, w_grid]
    
    # 10. 对每个 tubelet 找 top-k 并生成高斯 soft heatmap
    # [B, num_tubelets, h_grid * w_grid]
    tubelet_patch_flat = tubelet_patch_motion.view(B, num_tubelets, -1)
    
    # 获取每个 tubelet 的 top-k 索引和值
    topk_values, topk_indices = torch.topk(tubelet_patch_flat, k=top_k, dim=-1)  # [B, num_tubelets, top_k]
    
    # 初始化每个 tubelet 的 soft heatmap
    soft_heatmap_tubelets = torch.zeros(B, num_tubelets, h_grid, w_grid, device=clips.device, dtype=clips.dtype)
    
    for b in range(B):
        for tube_idx in range(num_tubelets):
            for k in range(top_k):
                idx = topk_indices[b, tube_idx, k].item()
                val = topk_values[b, tube_idx, k].item()
                
                # 将 flat index 转换为 2D 坐标
                cy = idx // w_grid
                cx = idx % w_grid
                
                # 计算高斯分布
                gaussian = torch.exp(-((grid_yy - cy)**2 + (grid_xx - cx)**2) / (2 * gaussian_sigma**2))
                
                # 用运动强度作为权重
                soft_heatmap_tubelets[b, tube_idx] += gaussian * val
    
    # 11. 对每个 tubelet 归一化到 [0, 1]
    for b in range(B):
        for tube_idx in range(num_tubelets):
            max_val = soft_heatmap_tubelets[b, tube_idx].max()
            if max_val > 0:
                soft_heatmap_tubelets[b, tube_idx] = soft_heatmap_tubelets[b, tube_idx] / max_val
    
    # 12. 扩展每个 tubelet 的 heatmap 到对应的帧
    # [B, num_tubelets, h_grid, w_grid] -> [B, num_tubelets, tubelet_size, h_grid, w_grid] -> [B, T, h_grid, w_grid]
    soft_heatmap_expanded = soft_heatmap_tubelets.unsqueeze(2).expand(B, num_tubelets, tubelet_size, h_grid, w_grid)
    soft_heatmap_patch = soft_heatmap_expanded.reshape(B, T, h_grid, w_grid)
    
    # 13. 上采样回原图尺寸
    # [B, T, h_grid, w_grid] -> [B, 1, T, h_grid, w_grid] -> interpolate -> [B, 1, T, H, W]
    soft_heatmap_5d = soft_heatmap_patch.unsqueeze(1)  # [B, 1, T, h_grid, w_grid]
    soft_heatmap_high_res = F.interpolate(
        soft_heatmap_5d, 
        size=(T, H, W), 
        mode='trilinear', 
        align_corners=False
    )
    
    return soft_heatmap_high_res.squeeze(1)  # [B, T, H, W]


def heatmap_to_colormap(heatmap_val):
    """
    将单通道 heatmap 值 (0~1) 转换为彩色图
    """
    # 映射 heatmap 到 0-255 并转为 uint8
    heatmap_norm = (heatmap_val * 255).clip(0, 255).astype(np.uint8)
    
    # 应用颜色映射
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    return heatmap_color

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
    import argparse
    parser = argparse.ArgumentParser(description='Motion Visualization')
    parser.add_argument('--csv_path', type=str, default=None,
                        help='直接指定CSV文件路径（优先级最高）')
    parser.add_argument('--dataset', type=str, default='Cholec80',
                        help='数据集名称，如 Cholec80, AutoLaparo, M2CAI16 等')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='每个clip的帧数 (16, 32, 64, 128 等)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='数据集划分 (train/val/test)')
    parser.add_argument('--sample_count', type=int, default=5,
                        help='可视化的样本数量')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（默认自动生成）')
    args = parser.parse_args()
    
    # 确定CSV路径
    if args.csv_path:
        csv_path = args.csv_path
    else:
        csv_path = f'data/Surge_Frames/{args.dataset}/clips_{args.num_frames}f/{args.split}_dense_{args.num_frames}f_detailed.csv'
    
    # 确定输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        dataset_name = args.dataset.lower() if not args.csv_path else os.path.basename(os.path.dirname(os.path.dirname(args.csv_path)))
        output_dir = f'vis_results_jepa_motion_separate/{dataset_name}_{args.num_frames}f'
    
    sample_count = args.sample_count
    
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
        original_dir = os.path.join(case_out_dir, "original")
        motion_dir = os.path.join(case_out_dir, "motion")
        overlay_dir = os.path.join(case_out_dir, "overlay")
        top3_dir = os.path.join(case_out_dir, "top3_soft")
        top3_overlay_dir = os.path.join(case_out_dir, "top3_overlay")
        ensure_dir(original_dir)
        ensure_dir(motion_dir)
        ensure_dir(overlay_dir)
        ensure_dir(top3_dir)
        ensure_dir(top3_overlay_dir)
        
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
            
            # 计算 Top-3 Patch Soft Heatmap
            top3_soft_tensor = get_top3_patch_soft_heatmap(clip_tensor, patch_size=16, top_k=3, gaussian_sigma=2.0)
            top3_soft_np = top3_soft_tensor.squeeze(0).cpu().numpy()  # [T, H, W]

        # 保存可视化结果
        original_files = []
        motion_files = []
        overlay_files = []
        top3_files = []
        top3_overlay_files = []
        
        for i in range(len(original_imgs)):
            orig_img = original_imgs[i]
            heatmap_val = heatmap_np[i]
            top3_val = top3_soft_np[i]
            
            # 保存原始帧
            orig_save_name = f"frame_{i:03d}.jpg"
            orig_save_path = os.path.join(original_dir, orig_save_name)
            cv2.imwrite(orig_save_path, orig_img)
            original_files.append(orig_save_path)
            
            # 保存 motion heatmap 帧
            motion_color = heatmap_to_colormap(heatmap_val)
            motion_save_name = f"motion_{i:03d}.jpg"
            motion_save_path = os.path.join(motion_dir, motion_save_name)
            cv2.imwrite(motion_save_path, motion_color)
            motion_files.append(motion_save_path)
            
            # 保存叠加帧 (motion heatmap 叠加到原图)
            overlay_img = apply_heatmap_overlay(orig_img, heatmap_val, alpha=0.5)
            overlay_save_name = f"overlay_{i:03d}.jpg"
            overlay_save_path = os.path.join(overlay_dir, overlay_save_name)
            cv2.imwrite(overlay_save_path, overlay_img)
            overlay_files.append(overlay_save_path)
            
            # 保存 Top-3 Soft Heatmap 帧
            top3_color = heatmap_to_colormap(top3_val)
            top3_save_name = f"top3_{i:03d}.jpg"
            top3_save_path = os.path.join(top3_dir, top3_save_name)
            cv2.imwrite(top3_save_path, top3_color)
            top3_files.append(top3_save_path)
            
            # 保存 Top-3 Soft 叠加帧
            top3_overlay_img = apply_heatmap_overlay(orig_img, top3_val, alpha=0.5)
            top3_overlay_save_name = f"top3_overlay_{i:03d}.jpg"
            top3_overlay_save_path = os.path.join(top3_overlay_dir, top3_overlay_save_name)
            cv2.imwrite(top3_overlay_save_path, top3_overlay_img)
            top3_overlay_files.append(top3_overlay_save_path)
            
        print(f"原始帧已保存至: {original_dir}")
        print(f"Motion帧已保存至: {motion_dir}")
        print(f"叠加帧已保存至: {overlay_dir}")
        print(f"Top-3 Soft帧已保存至: {top3_dir}")
        print(f"Top-3 叠加帧已保存至: {top3_overlay_dir}")
        
        # 生成GIF
        try:
            import imageio
            
            # 原始帧 GIF
            gif_original = []
            for filename in original_files:
                gif_original.append(imageio.imread(filename))
            if gif_original:
                imageio.mimsave(os.path.join(case_out_dir, 'original.gif'), gif_original, fps=5)
            
            # Motion heatmap GIF
            gif_motion = []
            for filename in motion_files:
                gif_motion.append(imageio.imread(filename))
            if gif_motion:
                imageio.mimsave(os.path.join(case_out_dir, 'motion.gif'), gif_motion, fps=5)
            
            # Overlay GIF (叠加)
            gif_overlay = []
            for filename in overlay_files:
                gif_overlay.append(imageio.imread(filename))
            if gif_overlay:
                imageio.mimsave(os.path.join(case_out_dir, 'overlay.gif'), gif_overlay, fps=5)
            
            # Top-3 Soft GIF
            gif_top3 = []
            for filename in top3_files:
                gif_top3.append(imageio.imread(filename))
            if gif_top3:
                imageio.mimsave(os.path.join(case_out_dir, 'top3_soft.gif'), gif_top3, fps=5)
            
            # Top-3 Overlay GIF
            gif_top3_overlay = []
            for filename in top3_overlay_files:
                gif_top3_overlay.append(imageio.imread(filename))
            if gif_top3_overlay:
                imageio.mimsave(os.path.join(case_out_dir, 'top3_overlay.gif'), gif_top3_overlay, fps=5)
                
            print(f"GIF已保存至: {case_out_dir}")
            
        except ImportError:
            print("提示: 安装 imageio 可生成GIF (pip install imageio)")
        except Exception as e:
            print(f"生成GIF出错: {e}")

    print("全部处理完成。")

if __name__ == "__main__":
    main()
