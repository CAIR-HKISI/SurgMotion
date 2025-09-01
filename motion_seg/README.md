# 视频光流计算工具 - 降噪版本

## 快速开始

### 1. 基本使用（推荐）
```bash
python motion_seg/video_flow.py
```
这将使用简单的降噪方法，自动生成对比视频。

### 2. 简单测试
```bash
python motion_seg/simple_test.py
```
使用原来的降噪方法进行测试。

### 3. 查看可用配置
```bash
python motion_seg/denoising_configs.py
```

## 问题解决

### 如果光流看不到或过度平滑：

1. **使用对比功能**（已启用）
   - 对比视频左侧显示原始光流
   - 对比视频右侧显示降噪后光流
   - 通过对比判断降噪是否过度

2. **调整参数**
   - 降低 `denoise_strength` 到 0.5-0.8
   - 降低 `confidence_threshold` 到 0.02-0.05

3. **使用预定义配置**
   - `surgical_video`: 手术视频推荐配置
   - `high_quality_video`: 高质量视频配置
   - `noisy_video`: 噪声较大视频配置

## 输出文件

- `*_flow.mp4`: 降噪后的光流视频
- `*_comparison.mp4`: 原始光流 vs 降噪光流对比视频

## 推荐工作流程

1. 先运行 `python motion_seg/video_flow.py` 查看效果
2. 如果效果不理想，运行 `python motion_seg/simple_test.py` 测试
3. 对比视频帮助判断降噪效果
4. 根据效果调整参数
