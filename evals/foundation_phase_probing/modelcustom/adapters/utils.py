"""
Common utility functions for Foundation Model Adapters.
"""

import argparse
import logging
import sys

logger = logging.getLogger(__name__)

import torch
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


def load_checkpoint_generic(
    checkpoint_path: Optional[str],
    default_path: Optional[str] = None,
    strict: bool = False,
    verbose: bool = True
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Generic checkpoint loading function that handles various checkpoint formats.
    
    Args:
        checkpoint_path: Path to checkpoint; None means use default_path
        default_path: Default checkpoint path
        strict: Whether to strictly load (passed to load_state_dict)
        verbose: Whether to print detailed info
    
    Returns:
        Tuple of (state_dict, info_message):
            - state_dict: Loaded weight dict, or None on failure
            - info_message: Loading info string
    """
    # Determine the actual checkpoint path to use
    if checkpoint_path is None and default_path is not None:
        checkpoint_path = default_path
        if verbose:
            logger.info("Using default checkpoint: %s", checkpoint_path)
    
    # Check if file exists
    if checkpoint_path is None:
        if verbose:
            logger.info("No checkpoint specified, using randomly initialized weights")
        return None, "no_checkpoint"
    
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        if verbose:
            logger.warning("Checkpoint not found: %s, using randomly initialized weights", checkpoint_path)
        return None, "checkpoint_not_found"
    
    # Load checkpoint
    try:
        if verbose:
            logger.info("Loading checkpoint from: %s", checkpoint_path)
        
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        state_dict = None
        load_key = None
        
        if isinstance(ckpt, dict):
            # Try common keys
            for key in ['model', 'state_dict', 'model_state_dict', 'teacher', 'student']:
                if key in ckpt:
                    state_dict = ckpt[key]
                    load_key = key
                    break
            
            # If no specific key found, assume the entire dict is the state_dict
            if state_dict is None:
                state_dict = ckpt
                load_key = "direct"
        else:
            # Non-dict format, try using directly
            state_dict = ckpt
            load_key = "raw"
        
        if verbose:
            if load_key == "direct":
                logger.info("Loaded as direct state_dict")
            else:
                logger.info("Loaded from '%s' key", load_key)
            if isinstance(state_dict, dict):
                logger.debug("State dict contains %s keys, sample: %s", len(state_dict), list(state_dict.keys())[:3])
        
        return state_dict, f"loaded_from_{load_key}"
    
    except Exception as e:
        if verbose:
            logger.exception("Error loading checkpoint: %s", e)
        return None, f"error: {str(e)}"


def apply_checkpoint_to_model(
    model: torch.nn.Module,
    state_dict: Optional[Dict[str, Any]],
    strict: bool = False,
    key_prefix_to_remove: Optional[str] = None,
    verbose: bool = True
) -> Optional[Any]:
    """
    Apply loaded state_dict to model.
    
    Args:
        model: Model to load weights into
        state_dict: Weight dictionary
        strict: Whether to strictly match all keys
        key_prefix_to_remove: Prefix to remove from keys (e.g. 'module.', 'backbone.')
        verbose: Whether to print detailed info
    
    Returns:
        Return info from load_state_dict (missing_keys, unexpected_keys)
    """
    if state_dict is None:
        if verbose:
            logger.info("No state_dict provided, model remains randomly initialized")
        return None
    
    # Clean key names
    if key_prefix_to_remove:
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            if k.startswith(key_prefix_to_remove):
                new_key = k[len(key_prefix_to_remove):]
            cleaned_state_dict[new_key] = v
        state_dict = cleaned_state_dict
        if verbose:
            logger.info("Removed prefix '%s' from keys", key_prefix_to_remove)
    
    # Load into model
    try:
        msg = model.load_state_dict(state_dict, strict=strict)
        
        if verbose:
            logger.info("Checkpoint loaded successfully")
            if msg.missing_keys:
                logger.debug("Missing keys (%s): %s", len(msg.missing_keys), msg.missing_keys[:10] if len(msg.missing_keys) > 10 else msg.missing_keys)
            if msg.unexpected_keys:
                logger.debug("Unexpected keys (%s): %s", len(msg.unexpected_keys), msg.unexpected_keys[:10] if len(msg.unexpected_keys) > 10 else msg.unexpected_keys)
        
        return msg
    
    except Exception as e:
        if verbose:
            logger.exception("Error applying checkpoint to model: %s", e)
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
    One-stop checkpoint loading and applying function.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Checkpoint path
        default_path: Default checkpoint path
        strict: Whether to strictly load
        key_prefix_to_remove: Key prefix to remove
        verbose: Whether to print detailed info
    
    Returns:
        Tuple of (success, message):
            - success: Whether loading succeeded
            - message: Status info
    """
    # Load checkpoint
    state_dict, load_info = load_checkpoint_generic(
        checkpoint_path=checkpoint_path,
        default_path=default_path,
        strict=strict,
        verbose=verbose
    )
    
    # Apply to model
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
    Given the arguments, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    from evals.foundation_phase_probing.modelcustom.adapters.defaults import get_cfg
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