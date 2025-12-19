"""
通用工具函数：用于Foundation Model Adapters
"""

import torch
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import argparse
import sys


def load_checkpoint_generic(
    checkpoint_path: Optional[str],
    default_path: Optional[str] = None,
    strict: bool = False,
    verbose: bool = True
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    通用的checkpoint加载函数，处理各种checkpoint格式
    
    Args:
        checkpoint_path: 指定的checkpoint路径，None表示使用默认路径
        default_path: 默认checkpoint路径
        strict: 是否严格加载（传递给load_state_dict）
        verbose: 是否打印详细信息
    
    Returns:
        Tuple of (state_dict, info_message):
            - state_dict: 加载的权重字典，如果失败则为None
            - info_message: 加载信息字符串
    """
    # 确定实际使用的checkpoint路径
    if checkpoint_path is None and default_path is not None:
        checkpoint_path = default_path
        if verbose:
            print(f"📍 Using default checkpoint: {checkpoint_path}")
    
    # 检查文件是否存在
    if checkpoint_path is None:
        if verbose:
            print("No checkpoint specified, using randomly initialized weights")
        return None, "no_checkpoint"
    
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        if verbose:
            print(f"Checkpoint not found: {checkpoint_path}")
            print("   Using randomly initialized weights")
        return None, "checkpoint_not_found"
    
    # 加载checkpoint
    try:
        if verbose:
            print(f"📂 Loading checkpoint from: {checkpoint_path}")
        
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        # 处理不同的checkpoint格式
        state_dict = None
        load_key = None
        
        if isinstance(ckpt, dict):
            # 尝试常见的key
            for key in ['model', 'state_dict', 'model_state_dict', 'teacher', 'student']:
                if key in ckpt:
                    state_dict = ckpt[key]
                    load_key = key
                    break
            
            # 如果没有找到特定key，假设整个dict就是state_dict
            if state_dict is None:
                state_dict = ckpt
                load_key = "direct"
        else:
            # 非字典格式，尝试直接使用
            state_dict = ckpt
            load_key = "raw"
        
        if verbose:
            if load_key == "direct":
                print("Loaded as direct state_dict")
            else:
                print(f"Loaded from '{load_key}' key")
            
            # 显示一些统计信息
            if isinstance(state_dict, dict):
                print(f"State dict contains {len(state_dict)} keys")
                # 显示前3个key作为示例
                sample_keys = list(state_dict.keys())[:3]
                print(f"Sample keys: {sample_keys}")
        
        return state_dict, f"loaded_from_{load_key}"
    
    except Exception as e:
        if verbose:
            print(f"Error loading checkpoint: {e}")
        return None, f"error: {str(e)}"


def apply_checkpoint_to_model(
    model: torch.nn.Module,
    state_dict: Optional[Dict[str, Any]],
    strict: bool = False,
    key_prefix_to_remove: Optional[str] = None,
    verbose: bool = True
) -> Optional[Any]:
    """
    将加载的state_dict应用到模型
    
    Args:
        model: 要加载权重的模型
        state_dict: 权重字典
        strict: 是否严格匹配所有keys
        key_prefix_to_remove: 需要从key中移除的前缀（如 'module.', 'backbone.'）
        verbose: 是否打印详细信息
    
    Returns:
        load_state_dict的返回信息（missing_keys, unexpected_keys）
    """
    if state_dict is None:
        if verbose:
            print("No state_dict provided, model remains randomly initialized")
        return None
    
    # 清理key名称
    if key_prefix_to_remove:
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            if k.startswith(key_prefix_to_remove):
                new_key = k[len(key_prefix_to_remove):]
            cleaned_state_dict[new_key] = v
        state_dict = cleaned_state_dict
        if verbose:
            print(f"Removed prefix '{key_prefix_to_remove}' from keys")
    
    # 加载到模型
    try:
        msg = model.load_state_dict(state_dict, strict=strict)
        
        if verbose:
            print("Checkpoint loaded successfully")
            
            if msg.missing_keys:
                print(f"Missing keys: {len(msg.missing_keys)}")
                if len(msg.missing_keys) <= 10:
                    for key in msg.missing_keys:
                        print(f"      - {key}")
                else:
                    print(f"      (showing first 5)")
                    for key in msg.missing_keys[:5]:
                        print(f"      - {key}")
            else:
                print("No missing keys")
            
            if msg.unexpected_keys:
                print(f"Unexpected keys: {len(msg.unexpected_keys)}")
                if len(msg.unexpected_keys) <= 10:
                    for key in msg.unexpected_keys:
                        print(f"      - {key}")
                else:
                    print(f"      (showing first 5)")
                    for key in msg.unexpected_keys[:5]:
                        print(f"      - {key}")
            else:
                print("No unexpected keys")
        
        return msg
    
    except Exception as e:
        if verbose:
            print(f"Error applying checkpoint to model: {e}")
        raise e


def load_and_apply_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Optional[str],
    default_path: Optional[str] = None,
    strict: bool = False,
    key_prefix_to_remove: Optional[str] = None,
    verbose: bool = True
) -> Tuple[bool, str]:
    """
    一站式checkpoint加载和应用函数
    
    Args:
        model: 要加载权重的模型
        checkpoint_path: checkpoint路径
        default_path: 默认checkpoint路径
        strict: 是否严格加载
        key_prefix_to_remove: 需要移除的key前缀
        verbose: 是否打印详细信息
    
    Returns:
        Tuple of (success, message):
            - success: 是否成功加载
            - message: 状态信息
    """
    # 加载checkpoint
    state_dict, load_info = load_checkpoint_generic(
        checkpoint_path=checkpoint_path,
        default_path=default_path,
        strict=strict,
        verbose=verbose
    )
    
    # 应用到模型
    if state_dict is not None:
        msg = apply_checkpoint_to_model(
            model=model,
            state_dict=state_dict,
            strict=strict,
            key_prefix_to_remove=key_prefix_to_remove,
            verbose=verbose
        )
        return True, load_info
    else:
        return False, load_info

#EndoFM

def parse_args():
    """
    Parse the following arguments for a default parser for PySlowFast users.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training and testing pipeline."
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/Kinetics/SLOWFAST_4x16_R50.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See slowfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    from .defaults import get_cfg
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    return cfg