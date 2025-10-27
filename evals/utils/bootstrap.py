"""
Bootstrap resampling utilities for computing uncertainty estimates on per-video metrics.

This module provides functions to perform bootstrap resampling on video-level metrics
to estimate standard deviations and confidence intervals.
"""

import numpy as np
from typing import Dict, List, Callable, Any, Optional
import logging

logger = logging.getLogger(__name__)


def bootstrap_per_video_metrics(
    per_video_results: List[Dict[str, Any]],
    metric_keys: List[str],
    n_bootstrap: int = 1000,
    random_seed: Optional[int] = None,
    video_id_key: str = "Video"
) -> Dict[str, Dict[str, float]]:
    """
    Perform bootstrap resampling on per-video metrics to estimate uncertainty.

    This function samples videos with replacement, calculates metrics for each bootstrap
    sample, and returns the standard deviation across bootstrap iterations as a measure
    of uncertainty.

    Args:
        per_video_results: List of dictionaries, each containing metrics for one video.
                          Each dict should have a video identifier and metric values.
        metric_keys: List of metric names to bootstrap (e.g., ['Accuracy', 'Macro_F1'])
        n_bootstrap: Number of bootstrap iterations (default: 1000)
        random_seed: Random seed for reproducibility (default: None)
        video_id_key: Key name for video identifier in per_video_results (default: "Video")

    Returns:
        Dictionary with structure:
        {
            'mean': {metric: mean_across_bootstraps, ...},
            'std': {metric: std_across_bootstraps, ...},
            'ci_lower': {metric: 2.5th_percentile, ...},
            'ci_upper': {metric: 97.5th_percentile, ...}
        }

        The 'std' values represent bootstrap standard deviation (uncertainty estimate).
        The 'ci_lower' and 'ci_upper' represent 95% confidence intervals.

    Example:
        >>> per_video = [
        ...     {"Video": "v1", "Accuracy": 85.0, "F1": 80.0},
        ...     {"Video": "v2", "Accuracy": 90.0, "F1": 88.0},
        ...     {"Video": "v3", "Accuracy": 87.0, "F1": 85.0}
        ... ]
        >>> results = bootstrap_per_video_metrics(
        ...     per_video,
        ...     metric_keys=['Accuracy', 'F1'],
        ...     n_bootstrap=1000
        ... )
        >>> print(f"Accuracy: {results['mean']['Accuracy']:.2f} ± {results['std']['Accuracy']:.2f}")
    """
    if not per_video_results:
        raise ValueError("per_video_results is empty")

    if not metric_keys:
        raise ValueError("metric_keys is empty")

    # Validate that all metric keys exist in the per-video results
    sample_result = per_video_results[0]
    missing_keys = [key for key in metric_keys if key not in sample_result]
    if missing_keys:
        raise ValueError(f"Metric keys {missing_keys} not found in per_video_results. "
                        f"Available keys: {list(sample_result.keys())}")

    n_videos = len(per_video_results)
    logger.info(f"Performing bootstrap with {n_bootstrap} iterations on {n_videos} videos")

    # Set random seed for reproducibility
    rng = np.random.RandomState(random_seed)

    # Store bootstrap results for each metric
    bootstrap_results = {metric: [] for metric in metric_keys}

    # Perform bootstrap iterations
    for i in range(n_bootstrap):
        # Sample videos with replacement
        bootstrap_indices = rng.choice(n_videos, size=n_videos, replace=True)
        bootstrap_sample = [per_video_results[idx] for idx in bootstrap_indices]

        # Calculate mean for each metric on this bootstrap sample
        for metric in metric_keys:
            metric_values = [video[metric] for video in bootstrap_sample]
            bootstrap_mean = np.mean(metric_values)
            bootstrap_results[metric].append(bootstrap_mean)

    # Calculate statistics across bootstrap iterations
    results = {
        'mean': {},
        'std': {},
        'ci_lower': {},
        'ci_upper': {}
    }

    for metric in metric_keys:
        bootstrap_means = np.array(bootstrap_results[metric])
        results['mean'][metric] = np.mean(bootstrap_means)
        results['std'][metric] = np.std(bootstrap_means)  # Bootstrap standard deviation
        results['ci_lower'][metric] = np.percentile(bootstrap_means, 2.5)  # 95% CI lower
        results['ci_upper'][metric] = np.percentile(bootstrap_means, 97.5)  # 95% CI upper

    logger.info(f"Bootstrap completed. Results computed for {len(metric_keys)} metrics.")

    return results


def bootstrap_with_custom_aggregation(
    per_video_results: List[Dict[str, Any]],
    aggregation_func: Callable[[List[Dict[str, Any]]], Dict[str, float]],
    n_bootstrap: int = 1000,
    random_seed: Optional[int] = None
) -> Dict[str, Dict[str, float]]:
    """
    Perform bootstrap resampling with a custom aggregation function.

    This is useful when you need more complex aggregation than simple mean,
    for example, when computing per-class metrics or weighted averages.

    Args:
        per_video_results: List of dictionaries, each containing metrics for one video
        aggregation_func: Function that takes a list of per-video results and returns
                         a dictionary of aggregated metrics
        n_bootstrap: Number of bootstrap iterations (default: 1000)
        random_seed: Random seed for reproducibility (default: None)

    Returns:
        Dictionary with structure:
        {
            'mean': {metric: mean_across_bootstraps, ...},
            'std': {metric: std_across_bootstraps, ...},
            'ci_lower': {metric: 2.5th_percentile, ...},
            'ci_upper': {metric: 97.5th_percentile, ...}
        }

    Example:
        >>> def custom_agg(videos):
        ...     # Custom aggregation: weighted by number of samples
        ...     total_samples = sum(v['Num_Samples'] for v in videos)
        ...     weighted_acc = sum(v['Accuracy'] * v['Num_Samples'] for v in videos) / total_samples
        ...     return {'Weighted_Accuracy': weighted_acc}
        >>> results = bootstrap_with_custom_aggregation(per_video, custom_agg, n_bootstrap=1000)
    """
    if not per_video_results:
        raise ValueError("per_video_results is empty")

    n_videos = len(per_video_results)
    logger.info(f"Performing bootstrap with custom aggregation: {n_bootstrap} iterations on {n_videos} videos")

    # Set random seed for reproducibility
    rng = np.random.RandomState(random_seed)

    # Perform one iteration to get metric keys
    sample_agg = aggregation_func(per_video_results)
    metric_keys = list(sample_agg.keys())

    # Store bootstrap results for each metric
    bootstrap_results = {metric: [] for metric in metric_keys}

    # Perform bootstrap iterations
    for i in range(n_bootstrap):
        # Sample videos with replacement
        bootstrap_indices = rng.choice(n_videos, size=n_videos, replace=True)
        bootstrap_sample = [per_video_results[idx] for idx in bootstrap_indices]

        # Apply custom aggregation function
        aggregated = aggregation_func(bootstrap_sample)

        for metric, value in aggregated.items():
            bootstrap_results[metric].append(value)

    # Calculate statistics across bootstrap iterations
    results = {
        'mean': {},
        'std': {},
        'ci_lower': {},
        'ci_upper': {}
    }

    for metric in metric_keys:
        bootstrap_means = np.array(bootstrap_results[metric])
        results['mean'][metric] = np.mean(bootstrap_means)
        results['std'][metric] = np.std(bootstrap_means)
        results['ci_lower'][metric] = np.percentile(bootstrap_means, 2.5)
        results['ci_upper'][metric] = np.percentile(bootstrap_means, 97.5)

    logger.info(f"Bootstrap with custom aggregation completed. Results for {len(metric_keys)} metrics.")

    return results


def print_bootstrap_results(
    results: Dict[str, Dict[str, float]],
    metric_keys: Optional[List[str]] = None,
    precision: int = 2
) -> None:
    """
    Pretty-print bootstrap results.

    Args:
        results: Results dictionary from bootstrap_per_video_metrics
        metric_keys: List of metrics to print (default: all metrics in results)
        precision: Number of decimal places (default: 2)
    """
    if metric_keys is None:
        metric_keys = list(results['mean'].keys())

    print("\n" + "="*70)
    print("Bootstrap Results (Uncertainty Estimates)")
    print("="*70)
    print(f"{'Metric':<25} {'Mean':<12} {'Std':<12} {'95% CI':<20}")
    print("-"*70)

    for metric in metric_keys:
        mean = results['mean'][metric]
        std = results['std'][metric]
        ci_lower = results['ci_lower'][metric]
        ci_upper = results['ci_upper'][metric]

        ci_str = f"[{ci_lower:.{precision}f}, {ci_upper:.{precision}f}]"
        print(f"{metric:<25} {mean:<12.{precision}f} {std:<12.{precision}f} {ci_str:<20}")

    print("="*70 + "\n")
