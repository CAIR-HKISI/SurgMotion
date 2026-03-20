#!/usr/bin/env python3
"""
检查 checkpoint 文件的完整性和架构信息
用法: python scripts/check_checkpoint.py checkpoint/latest2.pt
"""

import argparse
import sys
from pathlib import Path


def check_checkpoint(ckpt_path: str):
    import torch

    path = Path(ckpt_path)
    if not path.exists():
        print(f"错误: 文件不存在 {ckpt_path}")
        return False

    print(f"=" * 60)
    print(f"检查 checkpoint: {ckpt_path}")
    print(f"文件大小: {path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"=" * 60)

    # 1. 尝试加载
    print("\n[1] 加载 checkpoint...")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print("    ✅ 加载成功")
    except Exception as e:
        print(f"    ❌ 加载失败: {e}")
        return False

    # 2. 检查 checkpoint 结构
    print("\n[2] Checkpoint 顶层 keys:")
    if isinstance(ckpt, dict):
        for k in sorted(ckpt.keys()):
            v = ckpt[k]
            if isinstance(v, dict):
                print(f"    {k}: dict with {len(v)} keys")
            elif isinstance(v, torch.Tensor):
                print(f"    {k}: Tensor {v.shape}")
            else:
                print(f"    {k}: {type(v).__name__}")
    else:
        print(f"    类型: {type(ckpt).__name__}")

    # 3. 找到 encoder state_dict
    state_dict = None
    encoder_key = None

    # 常见的 key 模式
    possible_keys = [
        "target_encoder",
        "encoder",
        "model",
        "state_dict",
        "model_state_dict",
    ]

    for key in possible_keys:
        if isinstance(ckpt, dict) and key in ckpt:
            candidate = ckpt[key]
            if isinstance(candidate, dict) and any(
                "blocks" in k or "patch_embed" in k for k in candidate.keys()
            ):
                state_dict = candidate
                encoder_key = key
                break

    # 如果顶层就是 state_dict
    if state_dict is None and isinstance(ckpt, dict):
        if any("blocks" in k or "patch_embed" in k for k in ckpt.keys()):
            state_dict = ckpt
            encoder_key = "(top-level)"

    if state_dict is None:
        print("\n[3] ❌ 未找到有效的 encoder state_dict")
        print("    可能的 keys:", list(ckpt.keys()) if isinstance(ckpt, dict) else "N/A")
        return False

    print(f"\n[3] 找到 encoder state_dict (key='{encoder_key}')")
    print(f"    共 {len(state_dict)} 个参数")

    # 4. 分析模型架构
    print("\n[4] 分析模型架构...")

    # 找所有 block 编号 (支持 module.backbone.blocks.X 格式)
    block_nums = set()
    prefix = ""
    for k in state_dict.keys():
        if "blocks." in k:
            parts = k.split(".")
            for i, p in enumerate(parts):
                if p == "blocks" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    block_nums.add(int(parts[i + 1]))
                    if not prefix:
                        prefix = ".".join(parts[: i + 1]) + "."
                    break
    
    if prefix:
        print(f"    参数前缀: {prefix}")

    if block_nums:
        num_blocks = max(block_nums) + 1
        print(f"    Transformer blocks: {num_blocks} 层")
    else:
        num_blocks = 0
        print("    ⚠️ 未检测到 blocks")

    # 检查 hidden_dim
    hidden_dim = None
    for k, v in state_dict.items():
        if "blocks.0.norm1.weight" in k:
            hidden_dim = v.shape[0]
            break
    if hidden_dim is None:
        for k, v in state_dict.items():
            if "norm.weight" in k and "blocks" not in k:
                hidden_dim = v.shape[0]
                break

    if hidden_dim:
        print(f"    Hidden dimension: {hidden_dim}")

    # 检查 patch_embed
    patch_size = None
    tubelet_size = None
    for k, v in state_dict.items():
        if "patch_embed.proj.weight" in k:
            # Conv3d: (out_channels, in_channels, T, H, W)
            if len(v.shape) == 5:
                _, _, t, h, w = v.shape
                tubelet_size = t
                patch_size = h
                print(f"    Patch embed: Conv3d, tubelet={t}, patch={h}x{w}")
            # Conv2d: (out_channels, in_channels, H, W)
            elif len(v.shape) == 4:
                _, _, h, w = v.shape
                patch_size = h
                print(f"    Patch embed: Conv2d, patch={h}x{w}")
            break

    # 5. 推断模型类型
    print("\n[5] 推断模型类型:")

    model_type = "unknown"
    if num_blocks == 24 and hidden_dim == 1024:
        model_type = "vit_large"
        print(f"    ✅ 推断为: ViT-Large (24 blocks, dim=1024)")
    elif num_blocks == 24 and hidden_dim == 768:
        model_type = "vit_base"
        print(f"    ✅ 推断为: ViT-Base (24 blocks, dim=768)")
    elif num_blocks == 32 and hidden_dim == 1280:
        model_type = "vit_huge"
        print(f"    ✅ 推断为: ViT-Huge (32 blocks, dim=1280)")
    elif num_blocks == 40 and hidden_dim == 1408:
        model_type = "vit_giant"
        print(f"    ✅ 推断为: ViT-Giant (40 blocks, dim=1408)")
    elif num_blocks == 12 and hidden_dim == 768:
        model_type = "vit_small"
        print(f"    ✅ 推断为: ViT-Small (12 blocks, dim=768)")
    else:
        print(f"    ⚠️ 未知架构: {num_blocks} blocks, dim={hidden_dim}")

    # 6. 检查参数完整性
    print("\n[6] 参数完整性检查:")
    has_nan = False
    has_inf = False
    total_params = 0
    
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            total_params += v.numel()
            if torch.isnan(v).any():
                print(f"    ⚠️ NaN detected in: {k}")
                has_nan = True
            if torch.isinf(v).any():
                print(f"    ⚠️ Inf detected in: {k}")
                has_inf = True

    print(f"    总参数量: {total_params / 1e6:.2f}M")
    if not has_nan and not has_inf:
        print("    ✅ 无 NaN/Inf 异常值")

    # 7. 给出配置建议
    print("\n[7] 配置建议:")
    print(f"    在 YAML 配置中使用:")
    print(f"    model_name: {model_type}")
    if patch_size:
        print(f"    patch_size: {patch_size}")
    if tubelet_size:
        print(f"    tubelet_size: {tubelet_size}")
    print(f"    checkpoint_key: {encoder_key}")

    print("\n" + "=" * 60)
    print("✅ Checkpoint 检查完成，文件有效")
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(description="检查 checkpoint 文件")
    parser.add_argument(
        "checkpoint",
        type=str,
        nargs="?",
        default="checkpoint/latest2.pt",
        help="Checkpoint 文件路径",
    )
    args = parser.parse_args()

    success = check_checkpoint(args.checkpoint)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
