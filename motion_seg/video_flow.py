import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, median_filter

def preprocess_surgical_frame(frame, denoise_strength=1.0):
    """
    针对手术视频的预处理降噪，优化条状金属手术器械的可见性
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()
    
    # 1. 高斯滤波去除高频噪声
    sigma = 0.3 * denoise_strength
    gray = gaussian_filter(gray, sigma=sigma)
    
    # 2. 双边滤波，保留器械边缘
    gray = cv2.bilateralFilter(gray, 9, 50*denoise_strength, 50*denoise_strength)
    
    # 3. 形态学操作去除小噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    
    # 4. 对比度增强（降低 clipLimit 避免金属高光过曝）
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # 5. 轻度平滑，抑制 CLAHE 放大的高光点
    gray = cv2.GaussianBlur(gray, (3,3), sigmaX=0.5)
    
    return gray


def filter_surgical_optical_flow(flow, confidence_threshold=0.05):
    """
    针对手术视频的光流滤波，优化手术器械的运动感知
    
    Args:
        flow (np.ndarray): 原始光流，形状为(H, W, 2)
        confidence_threshold (float): 置信度阈值
    
    Returns:
        np.ndarray: 滤波后的光流
    """
    # 计算光流的大小
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    
    # 1. 轻度中值滤波去除异常值，保持手术器械的运动轨迹
    flow_filtered = np.zeros_like(flow)
    flow_filtered[..., 0] = median_filter(flow[..., 0], size=3)
    flow_filtered[..., 1] = median_filter(flow[..., 1], size=3)
    
    # 2. 轻度高斯滤波平滑，不破坏手术器械的精细运动
    flow_filtered[..., 0] = gaussian_filter(flow_filtered[..., 0], sigma=0.5)
    flow_filtered[..., 1] = gaussian_filter(flow_filtered[..., 1], sigma=0.5)
    
    # 3. 基于置信度的掩码过滤
    confidence = calculate_surgical_flow_confidence(flow_filtered)
    mask = confidence > confidence_threshold
    
    # 对低置信度区域进行插值，保持手术器械区域的连续性
    flow_filtered = interpolate_surgical_regions(flow_filtered, mask)
    
    return flow_filtered

def calculate_surgical_flow_confidence(flow):
    """
    计算光流置信度，结合条状金属器械的方向一致性
    """
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    angle = np.arctan2(flow[..., 1], flow[..., 0])
    
    # 局部方差（原有方法）
    kernel = np.ones((3, 3)) / 9
    local_mean = cv2.filter2D(magnitude, -1, kernel)
    local_var = cv2.filter2D(magnitude**2, -1, kernel) - local_mean**2
    confidence_mag = 1.0 / (1.0 + local_var)
    
    # 新增：方向一致性
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    mean_cos = cv2.filter2D(cos_angle, -1, kernel)
    mean_sin = cv2.filter2D(sin_angle, -1, kernel)
    dir_consistency = np.sqrt(mean_cos**2 + mean_sin**2)  # 越接近1说明方向越一致
    
    # 综合置信度：幅值一致性 × 方向一致性
    confidence = confidence_mag * (0.5 + 0.5 * dir_consistency)
    
    # 增强高运动区域的置信度
    high_motion_mask = magnitude > np.percentile(magnitude, 70)
    confidence[high_motion_mask] *= 1.2
    
    return np.clip(confidence, 0, 1)

def interpolate_surgical_regions(flow, mask):
    """
    对手术视频的低置信度区域进行插值，保持手术器械的连续性
    
    Args:
        flow (np.ndarray): 光流数组
        mask (np.ndarray): 置信度掩码
    
    Returns:
        np.ndarray: 插值后的光流
    """
    flow_interpolated = flow.copy()
    
    # 对低置信度区域进行插值
    for i in range(flow.shape[0]):
        for j in range(flow.shape[1]):
            if not mask[i, j]:
                # 使用周围高置信度区域的平均值进行插值
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < flow.shape[0] and 
                            0 <= nj < flow.shape[1] and 
                            mask[ni, nj]):
                            neighbors.append(flow[ni, nj])
                
                if neighbors:
                    flow_interpolated[i, j] = np.mean(neighbors, axis=0)
    
    return flow_interpolated

def flow_to_rgb(flow):
    """
    将光流转换为RGB可视化图像
    
    Args:
        flow (np.ndarray): 光流数组，形状为(H, W, 2)
    
    Returns:
        np.ndarray: RGB图像，形状为(H, W, 3)
    """
    # 计算光流的大小和方向
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # 归一化大小到0-1范围
    magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
    
    # 将角度转换为HSV色彩空间的色调
    hue = angle * 180 / np.pi / 2
    saturation = np.ones_like(magnitude)
    value = magnitude
    
    # 转换为HSV图像
    hsv = np.stack([hue, saturation * 255, value * 255], axis=-1).astype(np.uint8)
    
    # 转换为BGR图像
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return bgr


def calculate_optical_flow(video_path, output_path=None, enable_denoising=True, 
                          denoise_strength=0.7, confidence_threshold=0.12, 
                          save_comparison=True):
    """
    针对条状金属手术器械优化的光流计算
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息: {width}x{height}, {fps}fps, 总帧数: {total_frames}")
    
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"optical_flow_{base_name}.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    comparison_out = None
    if save_comparison:
        comparison_path = output_path.replace('.mp4', '_comparison.mp4')
        comparison_out = cv2.VideoWriter(comparison_path, fourcc, fps, (width*2, height))
    
    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("无法读取视频第一帧")
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # 🔧 新参数：条状金属器械优化
    flow_params = {
        'pyr_scale': 0.5,
        'levels': 4,       # 更多层，适合快速运动
        'winsize': 9,      # 更小窗口，适合细长器械
        'iterations': 5,   # 更多迭代，提高精度
        'poly_n': 7,       # 条状边缘更鲁棒
        'poly_sigma': 1.5, # 平滑噪声
        'flags': 0
    }
    
    print("开始计算手术视频光流...")
    
    for frame_idx in tqdm(range(1, total_frames), desc="计算光流"):
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        if enable_denoising:
            curr_gray = preprocess_surgical_frame(curr_frame, denoise_strength)
        else:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **flow_params)
        original_flow = flow.copy()
        
        if enable_denoising:
            flow = filter_surgical_optical_flow(flow, confidence_threshold)
        
        flow_visualization = flow_to_rgb(flow)
        out.write(flow_visualization)
        
        if save_comparison and comparison_out is not None:
            original_flow_viz = flow_to_rgb(original_flow)
            comparison_frame = np.hstack([original_flow_viz, flow_visualization])
            comparison_out.write(comparison_frame)
        
        prev_gray = curr_gray.copy()
    
    cap.release()
    out.release()
    if comparison_out is not None:
        comparison_out.release()
    
    print(f"光流计算完成，结果保存至: {output_path}")
    if save_comparison:
        print(f"对比视频保存至: {output_path.replace('.mp4', '_comparison.mp4')}")
    return output_path

def process_autolaparo_videos(input_dir, 
                             output_dir="logs/video_flow/autolaparo", 
                             enable_denoising=True,
                             denoise_strength=0.7, 
                             confidence_threshold=0.12,
                             save_comparison=True):
    """
    批量处理指定目录下的所有视频文件，提取光流
    
    Args:
        input_dir (str): 输入视频目录
        output_dir (str): 输出光流目录
        enable_denoising (bool): 是否启用降噪
        denoise_strength (float): 降噪强度
        confidence_threshold (float): 置信度阈值
        save_comparison (bool): 是否保存对比视频
    """
    import glob
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的视频格式
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv']
    
    # 获取所有视频文件
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))
        video_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    if not video_files:
        print(f"在目录 {input_dir} 中没有找到视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件:")
    for video_file in video_files:
        print(f"  - {os.path.basename(video_file)}")
    
    print(f"\n开始批量处理...")
    
    # 处理每个视频文件
    for i, video_path in enumerate(video_files, 1):
        try:
            print(f"\n[{i}/{len(video_files)}] 处理: {os.path.basename(video_path)}")
            
            # 生成输出文件名
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_flow.mp4")
            
            # 计算光流
            calculate_optical_flow(
                video_path=video_path,
                output_path=output_path,
                enable_denoising=enable_denoising,
                denoise_strength=denoise_strength,
                confidence_threshold=confidence_threshold,
                save_comparison=save_comparison
            )
            
            print(f"✓ 完成: {os.path.basename(output_path)}")
            
        except Exception as e:
            print(f"✗ 处理失败: {os.path.basename(video_path)} - {str(e)}")
            continue
    
    print(f"\n批量处理完成！共处理 {len(video_files)} 个视频文件")
    print(f"输出目录: {output_dir}")

def process_single_video(video_path, output_path, args):
    """处理单个视频"""
    if not os.path.exists(video_path):
        print(f"❌ 错误: 视频文件不存在: {video_path}")
        return
    
    try:
        output_path = calculate_optical_flow(
            video_path=video_path,
            output_path=output_path,
            enable_denoising=not args.no_denoise,
            denoise_strength=args.denoise_strength,
            confidence_threshold=args.confidence_threshold,
            save_comparison=not args.no_comparison
        )
        print(f"✅ 成功完成光流计算！输出文件: {output_path}")
    except Exception as e:
        print(f"❌ 处理过程中出现错误 ({video_path}): {str(e)}")


def main():
    """主函数，支持单个视频和批量处理"""
    parser = argparse.ArgumentParser(description="计算手术视频光流（支持手术专用降噪）")
    parser.add_argument("--video", type=str, default=None, help="输入视频路径（单个视频）")
    parser.add_argument("--output", type=str, default=None, help="输出视频路径（仅单个视频时可用）")
    parser.add_argument("--dir", type=str, default=None, help="输入目录，批量处理该目录下所有视频")
    parser.add_argument("--outdir", type=str, default="logs/video_flow", help="输出目录（批量时必用，单个视频时可选）")
    parser.add_argument("--no-denoise", action="store_true", help="禁用手术专用降噪")
    parser.add_argument("--denoise-strength", type=float, default=0.7, help="去噪强度 (0.5-2.0)")
    parser.add_argument("--confidence-threshold", type=float, default=0.12, help="置信度阈值 (0.05-0.3)")
    parser.add_argument("--no-comparison", action="store_true", help="不保存对比视频")
    args = parser.parse_args()

    # 默认路径（兼容旧版）
    default_video_path = "data/SurgicalAction160/02_injection/02_05.mp4"
    default_output_path = os.path.join("logs/video_flow", "SurgicalAction160_02_05_flow.mp4")

    # 批量模式
    if args.dir is not None:
        input_dir = args.dir
        if not os.path.isdir(input_dir):
            print(f"❌ 错误: 输入目录不存在: {input_dir}")
            return
        os.makedirs(args.outdir, exist_ok=True)
        
        video_files = glob.glob(os.path.join(input_dir, "*.mp4"))
        if not video_files:
            print(f"⚠️ 没有找到视频文件: {input_dir}")
            return
        
        print(f"📂 批量处理目录: {input_dir}, 共 {len(video_files)} 个视频")
        for video_path in video_files:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(args.outdir, f"{base_name}_flow.mp4")
            process_single_video(video_path, output_path, args)
        return
    
    # 单个视频模式
    video_path = args.video if args.video is not None else default_video_path
    
    if args.output is not None:
        # 用户显式指定了输出文件
        output_path = args.output
    elif args.outdir is not None:
        # 用户指定了输出目录
        os.makedirs(args.outdir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(args.outdir, f"{base_name}_flow.mp4")
    else:
        # 默认路径（兼容旧版）
        output_path = default_output_path

    process_single_video(video_path, output_path, args)


if __name__ == "__main__":
    main()


