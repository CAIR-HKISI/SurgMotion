import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import random

class FrameDiffStatistic:
    def __init__(self, data_dir, image_root="/data/hl/vjepa2"):
        """初始化帧差统计工具"""
        self.data_dir = data_dir
        self.image_root = image_root  # 图像根目录
        self.txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
        self.frame_extension = ".png"  # 帧图像扩展名
        
        if not self.txt_files:
            raise ValueError(f"在目录 {data_dir} 中未找到任何txt文件")
            
        print(f"找到 {len(self.txt_files)} 个txt文件")

    def select_random_txt(self):
        """随机选择一个txt文件"""
        return random.choice(self.txt_files)

    def read_frame_paths(self, txt_file):
        """读取txt文件中的帧路径"""
        with open(txt_file, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"文件 {os.path.basename(txt_file)} 包含 {len(lines)} 帧")
        
        # 构建完整路径
        frame_paths = []
        for line in lines:
            if os.path.isabs(line):
                path = line
            else:
                path = os.path.join(self.image_root, line)
            
            # 补充扩展名
            if not os.path.splitext(path)[1]:
                path += self.frame_extension
                
            frame_paths.append(path)
        
        # 验证路径有效性
        valid_paths = []
        for path in frame_paths:
            if os.path.exists(path):
                valid_paths.append(path)
            else:
                print(f"警告: 帧文件不存在: {path}")
        
        print(f"有效帧路径数量: {len(valid_paths)}/{len(frame_paths)}")
        return valid_paths

    def load_frames(self, frame_paths):
        """加载帧图像"""
        frames = []
        for path in frame_paths:
            frame = cv2.imread(path)
            if frame is not None:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # 转RGB用于显示
            else:
                print(f"警告: 无法加载帧 {path}")
        print(f"成功加载帧数量: {len(frames)}/{len(frame_paths)}")
        return frames

    def calculate_frame_diffs(self, frames):
        """仅计算帧差（不做回归分析）"""
        if len(frames) < 2:
            return []
            
        frame_diffs = []  # 存储帧差数据
        diff_images = []  # 存储帧差图像（用于可视化）
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
        
        for i in range(1, len(frames)):
            current_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            
            # 计算帧差（绝对值）
            diff = cv2.absdiff(prev_gray, current_gray)
            frame_diffs.append(np.sum(diff))  # 帧差总和
            diff_images.append(diff)  # 保存帧差图像
            
            prev_gray = current_gray
        
        return frame_diffs, diff_images

    def visualize(self, frames, frame_diffs, diff_images, txt_filename):
        """可视化帧、帧差数据和统计信息"""
        if not frames or not frame_diffs:
            print("无足够数据进行可视化")
            return
            
        n_frames = len(frames)
        n_diffs = len(frame_diffs)
        
        # 按8帧一组划分
        frame_groups = [frames[i:i+8] for i in range(0, n_frames, 8)]
        # 帧差对应帧组（第i组帧差对应第i组的后7帧与前7帧的差异）
        diff_groups = [frame_diffs[i:i+7] for i in range(0, n_diffs, 7)]
        
        n_groups = len(frame_groups)
        total_rows = n_groups * 3  # 每组：1行原图 + 1行帧差图 + 1行统计
        
        fig = plt.figure(figsize=(20, total_rows * 3))
        fig.suptitle(f'帧差统计分析: {os.path.basename(txt_filename)}', fontsize=16, y=0.99)
        
        for group_idx in range(n_groups):
            # ---------------------- 1. 显示原始帧 ----------------------
            img_group = frame_groups[group_idx]
            img_row = group_idx * 3  # 原始帧行索引
            
            for col in range(8):
                if col >= len(img_group):
                    ax = plt.subplot(total_rows, 8, img_row * 8 + col + 1)
                    ax.axis('off')
                    continue
                
                ax = plt.subplot(total_rows, 8, img_row * 8 + col + 1)
                ax.imshow(img_group[col])
                ax.set_title(f'帧 {group_idx*8 + col}', fontsize=8)
                ax.axis('off')
            
            # ---------------------- 2. 显示帧差图 ----------------------
            diff_row = img_row + 1  # 帧差图行索引
            start_diff_idx = group_idx * 7
            end_diff_idx = start_diff_idx + min(7, len(diff_images) - start_diff_idx)
            
            for col in range(7):  # 8帧对应7个帧差
                diff_idx = start_diff_idx + col
                if diff_idx >= len(diff_images):
                    ax = plt.subplot(total_rows, 8, diff_row * 8 + col + 1)
                    ax.axis('off')
                    continue
                
                ax = plt.subplot(total_rows, 8, diff_row * 8 + col + 1)
                ax.imshow(diff_images[diff_idx], cmap='gray')
                ax.set_title(f'差 {diff_idx}\n值: {frame_diffs[diff_idx]:.0f}', fontsize=7)
                ax.axis('off')
            
            # 第8列显示该组帧差统计
            ax_stats = plt.subplot(total_rows, 8, diff_row * 8 + 8)
            group_diff_data = diff_groups[group_idx] if group_idx < len(diff_groups) else []
            if group_diff_data:
                ax_stats.bar(range(len(group_diff_data)), group_diff_data, color='skyblue')
                ax_stats.set_title(f'组{group_idx+1}帧差分布\n均值: {np.mean(group_diff_data):.0f}\n最大: {np.max(group_diff_data):.0f}', fontsize=7)
                ax_stats.tick_params(axis='x', labelsize=6)
                ax_stats.tick_params(axis='y', labelsize=6)
            else:
                ax_stats.text(0.5, 0.5, '无数据', ha='center', va='center')
                ax_stats.axis('off')
            
            # ---------------------- 3. 显示帧差时序曲线 ----------------------
            curve_row = img_row + 2  # 曲线行索引
            ax_curve = plt.subplot(total_rows, 1, curve_row + 1)
            
            if group_diff_data:
                ax_curve.plot(range(len(group_diff_data)), group_diff_data, 'b-o', markersize=4)
                ax_curve.set_title(f'组{group_idx+1}帧差时序变化', fontsize=9)
                ax_curve.set_xlabel('帧差索引', fontsize=8)
                ax_curve.set_ylabel('帧差总和', fontsize=8)
                ax_curve.tick_params(axis='both', labelsize=7)
                ax_curve.grid(alpha=0.3)
            else:
                ax_curve.text(0.5, 0.5, '无帧差数据', ha='center', va='center')
                ax_curve.axis('off')
        
        plt.tight_layout()
        
        # 保存结果
        output_dir = "frame_diff_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.splitext(os.path.basename(txt_filename))[0] + '.png'
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可视化结果已保存至: {output_path}")
        return output_path

    def run_analysis(self):
        """运行完整分析流程"""
        selected_txt = self.select_random_txt()
        print(f"随机选择的文件: {selected_txt}")
        
        # 读取帧路径并加载帧
        frame_paths = self.read_frame_paths(selected_txt)
        frames = self.load_frames(frame_paths)
        if len(frames) < 2:
            print("帧数量不足，无法计算帧差")
            return
        
        # 仅计算帧差（不做回归）
        frame_diffs, diff_images = self.calculate_frame_diffs(frames)
        if not frame_diffs:
            print("无法计算帧差")
            return
        
        # 可视化
        self.visualize(frames, frame_diffs, diff_images, selected_txt)

if __name__ == "__main__":
    # 配置路径
    data_dir = "/data/wjl/vjepa2/data/Surge_Frames/AutoLaparo/clips_32f/clip_dense_32f_info/train"
    image_root = "/data/wjl/vjepa2"  # 图像根目录
    
    try:
        analyzer = FrameDiffStatistic(data_dir, image_root)
        analyzer.run_analysis()
    except Exception as e:
        print(f"分析出错: {str(e)}")
    