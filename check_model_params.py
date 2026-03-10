#!/usr/bin/env python3
"""
脚本用于查询 ckpts/ckpts_foundation 目录下模型参数的大小
"""

import os
import glob
from pathlib import Path

import torch
import numpy as np


def decode_flax_msgpack(data):
    """解码 flax/orbax 格式的 msgpack 数据，将 ExtType 转换为 numpy 数组"""
    try:
        import msgpack
    except ImportError:
        return data
    
    def decode_numpy_from_ext(ext):
        """解析 flax/orbax 格式的 numpy 数组"""
        if not isinstance(ext, msgpack.ExtType) or ext.code != 1:
            return ext
        
        # ExtType 的 data 是 msgpack 序列化的 (shape, dtype, raw_bytes)
        unpacked = msgpack.unpackb(ext.data, raw=True)
        shape = tuple(unpacked[0])
        dtype_str = unpacked[1].decode('utf-8') if isinstance(unpacked[1], bytes) else unpacked[1]
        raw_data = unpacked[2]
        
        arr = np.frombuffer(raw_data, dtype=dtype_str).reshape(shape)
        return arr
    
    def decode_recursive(d):
        """递归解码所有 ExtType"""
        if isinstance(d, msgpack.ExtType):
            return decode_numpy_from_ext(d)
        elif isinstance(d, dict):
            return {k: decode_recursive(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [decode_recursive(v) for v in d]
        return d
    
    return decode_recursive(data)


def count_parameters_recursive(data):
    """递归统计参数数量，支持 torch.Tensor, numpy.ndarray, dict, list"""
    total_params = 0
    
    if isinstance(data, torch.Tensor):
        return data.numel()
    elif isinstance(data, np.ndarray):
        return data.size
    elif isinstance(data, dict):
        for value in data.values():
            total_params += count_parameters_recursive(value)
    elif isinstance(data, (list, tuple)):
        for item in data:
            total_params += count_parameters_recursive(item)
    
    return total_params


def count_parameters(state_dict):
    """统计 state_dict 中的参数数量（兼容旧接口）"""
    return count_parameters_recursive(state_dict)


def format_params(num_params):
    """格式化参数数量显示"""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)


def get_file_size(filepath):
    """获取文件大小"""
    size_bytes = os.path.getsize(filepath)
    if size_bytes >= 1e9:
        return f"{size_bytes / 1e9:.2f} GB"
    elif size_bytes >= 1e6:
        return f"{size_bytes / 1e6:.2f} MB"
    elif size_bytes >= 1e3:
        return f"{size_bytes / 1e3:.2f} KB"
    else:
        return f"{size_bytes} B"


def load_and_count_params(filepath):
    """加载模型文件并统计参数"""
    ext = Path(filepath).suffix.lower()
    filename = os.path.basename(filepath)
    
    try:
        if ext in ['.pth', '.pt', '.torch']:
            checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
            
            # 处理不同的 checkpoint 格式
            if isinstance(checkpoint, dict):
                # 特殊处理：vitg.pt 只统计 target_encoder 部分
                if filename == 'vitg.pt' and 'target_encoder' in checkpoint:
                    return count_parameters(checkpoint['target_encoder'])
                
                # 常见的 key 名称
                possible_keys = ['state_dict', 'model', 'model_state_dict', 
                                'encoder', 'network', 'net', 'module']
                
                state_dict = None
                for key in possible_keys:
                    if key in checkpoint:
                        state_dict = checkpoint[key]
                        break
                
                if state_dict is None:
                    # 直接使用 checkpoint 作为 state_dict
                    state_dict = checkpoint
                    
                return count_parameters(state_dict)
            else:
                return count_parameters(checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else {})
                
        elif ext == '.pkl':
            # 有些 .pkl 文件实际上是用 torch.save 保存的
            try:
                checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
                if isinstance(checkpoint, dict):
                    return count_parameters(checkpoint)
                return 0
            except Exception:
                # 如果 torch.load 失败，尝试用 pickle
                import pickle
                with open(filepath, 'rb') as f:
                    checkpoint = pickle.load(f)
                if isinstance(checkpoint, dict):
                    return count_parameters(checkpoint)
                return 0
            
        elif ext == '.safetensors':
            from safetensors import safe_open
            total_params = 0
            with safe_open(filepath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    total_params += tensor.numel()
            return total_params
        
        else:
            # 没有扩展名或未知扩展名的文件，尝试多种格式
            # 先尝试 msgpack（常见于 JAX/Flax 模型）
            try:
                import msgpack
                with open(filepath, 'rb') as f:
                    data = msgpack.load(f, raw=False)
                if isinstance(data, dict):
                    # 检查是否是 flax/orbax 格式（包含 ExtType）
                    decoded = decode_flax_msgpack(data)
                    return count_parameters_recursive(decoded)
            except Exception:
                pass
            
            # 尝试 torch.load
            try:
                checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
                if isinstance(checkpoint, dict):
                    return count_parameters(checkpoint)
            except Exception:
                pass
            
            return None
            
    except Exception as e:
        print(f"  ⚠️ 加载失败: {e}")
        return None


def is_model_file(filepath):
    """判断是否可能是模型文件（用于无扩展名文件）"""
    # 排除明显不是模型的文件
    filename = os.path.basename(filepath)
    excluded = ['README', 'LICENSE', 'config', '.json', '.txt', '.md', '.py', '.yaml', '.yml']
    for exc in excluded:
        if exc.lower() in filename.lower():
            return False
    
    # 检查文件大小（模型文件通常较大，>1MB）
    try:
        size = os.path.getsize(filepath)
        return size > 1024 * 1024  # > 1MB
    except:
        return False


def scan_directory(base_dir):
    """扫描目录下的所有模型文件"""
    extensions = ['*.pth', '*.pt', '*.torch', '*.pkl', '*.safetensors']
    model_files = []
    
    # 扫描有扩展名的模型文件
    for ext in extensions:
        model_files.extend(glob.glob(os.path.join(base_dir, '**', ext), recursive=True))
    
    # 扫描没有扩展名的文件（可能是 msgpack 格式的模型）
    for root, dirs, files in os.walk(base_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            # 跳过有常见扩展名的文件
            ext = Path(filename).suffix.lower()
            if ext in ['.pth', '.pt', '.torch', '.pkl', '.safetensors', 
                       '.json', '.txt', '.md', '.py', '.yaml', '.yml', '.tar', '.zip']:
                continue
            # 检查是否可能是模型文件
            if not ext and is_model_file(filepath):
                model_files.append(filepath)
    
    return sorted(set(model_files))


def main():
    base_dir = "ckpts/ckpts_foundation"
    extra_files = ["ckpts/vitg.pt"]  # 额外需要统计的单独文件
    
    print("=" * 80)
    print(f"📁 扫描目录: {base_dir}")
    print(f"📄 额外文件: {', '.join(extra_files)}")
    print("=" * 80)
    
    model_files = scan_directory(base_dir)
    
    # 添加额外的单独文件
    for extra in extra_files:
        if os.path.isfile(extra) and extra not in model_files:
            model_files.append(extra)
    
    model_files = sorted(model_files)
    
    if not model_files:
        print("未找到模型文件!")
        return
    
    print(f"\n找到 {len(model_files)} 个模型文件\n")
    print("-" * 80)
    print(f"{'模型文件':<60} {'文件大小':<12} {'参数量':<15}")
    print("-" * 80)
    
    total_params = 0
    results = []
    
    for filepath in model_files:
        rel_path = os.path.relpath(filepath, base_dir)
        file_size = get_file_size(filepath)
        
        print(f"正在加载: {rel_path}...", end='\r')
        
        params = load_and_count_params(filepath)
        
        if params is not None:
            total_params += params
            params_str = format_params(params)
            results.append((rel_path, file_size, params, params_str))
            print(f"{rel_path:<60} {file_size:<12} {params_str:<15}")
        else:
            results.append((rel_path, file_size, 0, "N/A"))
            print(f"{rel_path:<60} {file_size:<12} {'N/A':<15}")
    
    print("-" * 80)
    print(f"\n📊 统计摘要:")
    print(f"   总模型文件数: {len(model_files)}")
    print(f"   总参数量: {format_params(total_params)} ({total_params:,})")
    print("=" * 80)
    
    # 按参数量排序显示 Top 10
    print("\n🏆 参数量 Top 10:")
    sorted_results = sorted([r for r in results if r[2] > 0], key=lambda x: x[2], reverse=True)
    for i, (name, size, params, params_str) in enumerate(sorted_results[:10], 1):
        print(f"   {i:2}. {name}: {params_str}")


if __name__ == "__main__":
    main()
