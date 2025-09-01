# 视频光流计算工具

这个工具用于计算视频的密集光流（Dense Optical Flow），特别适用于手术视频等医学影像分析。

## 功能特点

- 读取长视频文件（如手术视频）
- 使用Farneback算法计算密集光流
- 将光流结果可视化并保存为视频
- 支持进度条显示处理进度
- 自动处理视频格式和参数

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本使用

```bash
python motion_seg/video_flow.py
```

### 自定义使用

```python
from motion_seg.video_flow import calculate_optical_flow

# 计算光流并保存
output_path = calculate_optical_flow(
    video_path="your_video.mp4",
    output_path="output_flow.mp4"
)
```

## 参数说明

### 光流计算参数

- `pyr_scale`: 金字塔缩放比例 (默认: 0.5)
- `levels`: 金字塔层数 (默认: 3)
- `winsize`: 窗口大小 (默认: 15)
- `iterations`: 迭代次数 (默认: 3)
- `poly_n`: 多项式展开参数 (默认: 5)
- `poly_sigma`: 高斯标准差 (默认: 1.2)

### 输出说明

- 输出视频将保存为 `optical_flow_[原文件名].mp4`
- 光流可视化使用HSV色彩空间：
  - 色调(Hue): 表示运动方向
  - 饱和度(Saturation): 固定为最大值
  - 明度(Value): 表示运动强度

## 注意事项

1. 确保输入视频文件存在且可读
2. 处理长视频时可能需要较长时间
3. 输出视频使用MP4格式，确保系统支持相应编解码器
4. 建议在处理前检查磁盘空间是否充足

## 示例输出

```
视频信息: 1920x1080, 30fps, 总帧数: 3000
开始计算光流...
计算光流: 100%|██████████| 2999/2999 [05:23<00:00, 9.28it/s]
光流计算完成，结果保存至: optical_flow_video_05.mp4
成功完成光流计算！输出文件: optical_flow_video_05.mp4
```
