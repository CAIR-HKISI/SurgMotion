# 视频光流处理工具

这个工具可以将输入视频转换为光流视频。

## 功能特点

- 支持多种视频格式输入
- 使用RAFT光流模型进行光流计算
- 自动保存光流视频
- 支持命令行参数配置
- 显示处理进度

## 使用方法

### 基本用法

```bash
python pl_flow.py --input input_video.mp4 --output output_flow_video.mp4
```

### 完整参数

```bash
python pl_flow.py \
    --input /path/to/input_video.mp4 \
    --output /path/to/output_flow_video.mp4 \
    --model raft_small \
    --ckpt things
```

### 参数说明

- `--input, -i`: 输入视频路径（必需）
- `--output, -o`: 输出光流视频路径（必需）
- `--model`: 光流模型名称（默认: raft_small）
- `--ckpt`: 模型检查点路径（默认: things）

## 支持的模型

- `raft_small`: RAFT小型模型
- `raft`: RAFT标准模型
- 其他ptlflow支持的模型

## 输出格式

- 输出视频格式：MP4
- 编码器：mp4v
- 保持原始视频的帧率和分辨率

## 注意事项

1. 确保已安装所需的依赖包：
   - opencv-python
   - torch
   - ptlflow

2. 首次运行时会自动下载模型权重

3. 处理时间取决于视频长度和模型复杂度

4. 输出目录会自动创建（如果不存在）

## 示例

```bash
# 处理一个短视频
python pl_flow.py -i sample.mp4 -o flow_output.mp4

# 使用不同的模型
python pl_flow.py -i input.mp4 -o output.mp4 --model raft --ckpt sintel
```
