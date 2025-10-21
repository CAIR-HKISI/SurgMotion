"""
Test script for bootstrap uncertainty estimation on per-video metrics.

This script tests the bootstrap implementation with synthetic data and
can also be used with real prediction CSV files.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path to import evals modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.utils.bootstrap import bootstrap_per_video_metrics, print_bootstrap_results


def generate_synthetic_predictions(num_videos=20, frames_per_video=1000, num_classes=7):
    """
    Generate synthetic predictions similar to surgical phase recognition.

    Args:
        num_videos: Number of videos to generate
        frames_per_video: Number of frames per video
        num_classes: Number of phase classes

    Returns:
        DataFrame with columns [data_idx, vid, prediction, label]
    """
    print(f"Generating synthetic data: {num_videos} videos, {frames_per_video} frames each, {num_classes} classes")

    all_predictions = []

    for vid in range(num_videos):
        # Generate ground truth sequence (simulating surgical phases)
        # Create phase segments of varying lengths
        segments = []
        current_frame = 0

        while current_frame < frames_per_video:
            # Random phase
            phase = np.random.randint(0, num_classes)
            # Random segment length (50-200 frames)
            segment_length = np.random.randint(50, 200)
            segment_length = min(segment_length, frames_per_video - current_frame)

            segments.extend([phase] * segment_length)
            current_frame += segment_length

        gt_sequence = np.array(segments[:frames_per_video])

        # Generate predictions with some error
        # Start with ground truth
        pred_sequence = gt_sequence.copy()

        # Add random errors (10-30% error rate, varying by video)
        error_rate = np.random.uniform(0.10, 0.30)
        num_errors = int(frames_per_video * error_rate)
        error_indices = np.random.choice(frames_per_video, num_errors, replace=False)

        for idx in error_indices:
            # Replace with random wrong class
            wrong_classes = [c for c in range(num_classes) if c != gt_sequence[idx]]
            pred_sequence[idx] = np.random.choice(wrong_classes)

        # Add to predictions list
        for frame_idx in range(frames_per_video):
            all_predictions.append([
                frame_idx,  # data_idx
                vid,        # vid
                pred_sequence[frame_idx],  # prediction
                gt_sequence[frame_idx]     # label
            ])

    df = pd.DataFrame(all_predictions, columns=['data_idx', 'vid', 'prediction', 'label'])
    print(f"Generated {len(df)} total frames across {num_videos} videos")
    return df


def compute_per_video_metrics(predictions_df):
    """
    Compute per-video metrics without bootstrap (for comparison).

    Args:
        predictions_df: DataFrame with columns [data_idx, vid, prediction, label]

    Returns:
        List of per-video metric dictionaries
    """
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
    except ImportError:
        print("ERROR: scikit-learn not installed. Install with: pip install scikit-learn")
        sys.exit(1)

    per_video = []

    for vid, subdf in predictions_df.groupby('vid'):
        subdf = subdf.sort_values('data_idx')
        gt = subdf['label'].values
        pred = subdf['prediction'].values

        acc = accuracy_score(gt, pred) * 100
        macro_prec = precision_score(gt, pred, average='macro', zero_division=0) * 100
        macro_rec = recall_score(gt, pred, average='macro', zero_division=0) * 100
        macro_iou = jaccard_score(gt, pred, average='macro', zero_division=0) * 100
        macro_f1 = f1_score(gt, pred, average='macro', zero_division=0) * 100
        n_samples = len(gt)

        per_video.append({
            "Video": vid,
            "Num_Samples": n_samples,
            "Accuracy": acc,
            "Macro_Precision": macro_prec,
            "Macro_Recall": macro_rec,
            "Macro_IoU": macro_iou,
            "Macro_F1": macro_f1
        })

    return per_video


def test_bootstrap_synthetic():
    """Test bootstrap with synthetic data."""
    print("="*70)
    print("TEST 1: Bootstrap with Synthetic Data")
    print("="*70)

    # Generate synthetic predictions
    df = generate_synthetic_predictions(num_videos=20, frames_per_video=1000, num_classes=7)

    # Compute per-video metrics
    per_video = compute_per_video_metrics(df)

    print(f"\nComputed metrics for {len(per_video)} videos")
    print(f"Sample video metrics: {per_video[0]}")

    # Standard statistics (without bootstrap)
    print("\n" + "-"*70)
    print("Standard Statistics (Simple Mean ± Std)")
    print("-"*70)
    metrics = ["Accuracy", "Macro_Precision", "Macro_Recall", "Macro_IoU", "Macro_F1"]
    for metric in metrics:
        vals = [v[metric] for v in per_video]
        mean = np.mean(vals)
        std = np.std(vals)
        print(f"{metric:<25} {mean:.2f} ± {std:.2f}")

    # Bootstrap statistics
    print("\n" + "-"*70)
    print("Bootstrap Statistics (1000 iterations)")
    print("-"*70)
    bootstrap_results = bootstrap_per_video_metrics(
        per_video_results=per_video,
        metric_keys=metrics,
        n_bootstrap=1000,
        random_seed=42
    )

    print_bootstrap_results(bootstrap_results, metric_keys=metrics)

    # Compare standard std vs bootstrap std
    print("\n" + "-"*70)
    print("Comparison: Standard Std vs Bootstrap Std")
    print("-"*70)
    print(f"{'Metric':<25} {'Standard Std':<15} {'Bootstrap Std':<15} {'Difference':<15}")
    print("-"*70)
    for metric in metrics:
        vals = [v[metric] for v in per_video]
        standard_std = np.std(vals)
        bootstrap_std = bootstrap_results['std'][metric]
        diff = abs(standard_std - bootstrap_std)
        print(f"{metric:<25} {standard_std:<15.2f} {bootstrap_std:<15.2f} {diff:<15.2f}")

    print("\n" + "="*70)
    print("TEST 1 COMPLETED SUCCESSFULLY")
    print("="*70)


def test_bootstrap_with_few_videos():
    """Test bootstrap with small number of videos (edge case)."""
    print("\n\n" + "="*70)
    print("TEST 2: Bootstrap with Few Videos (5 videos)")
    print("="*70)

    # Generate synthetic predictions with only 5 videos
    df = generate_synthetic_predictions(num_videos=5, frames_per_video=500, num_classes=7)

    # Compute per-video metrics
    per_video = compute_per_video_metrics(df)

    print(f"\nComputed metrics for {len(per_video)} videos")

    # Bootstrap statistics
    metrics = ["Accuracy", "Macro_F1"]
    bootstrap_results = bootstrap_per_video_metrics(
        per_video_results=per_video,
        metric_keys=metrics,
        n_bootstrap=1000,
        random_seed=42
    )

    print_bootstrap_results(bootstrap_results, metric_keys=metrics)

    print("\nNote: With few videos, bootstrap uncertainty will be larger (wider confidence intervals)")
    print("="*70)
    print("TEST 2 COMPLETED SUCCESSFULLY")
    print("="*70)


def test_bootstrap_reproducibility():
    """Test that bootstrap results are reproducible with same seed."""
    print("\n\n" + "="*70)
    print("TEST 3: Bootstrap Reproducibility")
    print("="*70)

    # Generate synthetic predictions
    df = generate_synthetic_predictions(num_videos=10, frames_per_video=500, num_classes=7)
    per_video = compute_per_video_metrics(df)

    # Run bootstrap twice with same seed
    metrics = ["Accuracy", "Macro_F1"]

    results1 = bootstrap_per_video_metrics(
        per_video_results=per_video,
        metric_keys=metrics,
        n_bootstrap=1000,
        random_seed=42
    )

    results2 = bootstrap_per_video_metrics(
        per_video_results=per_video,
        metric_keys=metrics,
        n_bootstrap=1000,
        random_seed=42
    )

    print("\nComparing two bootstrap runs with same seed (seed=42):")
    print("-"*70)
    print(f"{'Metric':<25} {'Run 1 Mean':<15} {'Run 2 Mean':<15} {'Difference':<15}")
    print("-"*70)
    for metric in metrics:
        mean1 = results1['mean'][metric]
        mean2 = results2['mean'][metric]
        diff = abs(mean1 - mean2)
        print(f"{metric:<25} {mean1:<15.6f} {mean2:<15.6f} {diff:<15.10f}")

    # Check if results are identical
    all_identical = True
    for metric in metrics:
        if abs(results1['mean'][metric] - results2['mean'][metric]) > 1e-10:
            all_identical = False
            break

    if all_identical:
        print("\n✓ Results are IDENTICAL (reproducible)")
    else:
        print("\n✗ Results are DIFFERENT (not reproducible)")

    print("="*70)
    print("TEST 3 COMPLETED SUCCESSFULLY")
    print("="*70)


def test_bootstrap_from_csv(csv_path):
    """
    Test bootstrap with real prediction CSV file.

    Args:
        csv_path: Path to CSV file with columns [head, data_idx, vid, prediction, label, dataset_idx]
                 or [data_idx, vid, prediction, label]
    """
    print("\n\n" + "="*70)
    print(f"TEST 4: Bootstrap with Real Data from CSV")
    print(f"CSV Path: {csv_path}")
    print("="*70)

    # Load predictions
    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} predictions")
    print(f"Columns: {df.columns.tolist()}")

    # If multi-head, process each head separately
    if 'head' in df.columns:
        print(f"\nMulti-head predictions detected. Heads: {df['head'].unique()}")

        for head_id in sorted(df['head'].unique()):
            head_df = df[df['head'] == head_id]
            print(f"\n--- Processing Head {head_id} ---")

            per_video = compute_per_video_metrics(head_df)
            print(f"Videos: {len(per_video)}")

            metrics = ["Accuracy", "Macro_Precision", "Macro_Recall", "Macro_IoU", "Macro_F1"]
            bootstrap_results = bootstrap_per_video_metrics(
                per_video_results=per_video,
                metric_keys=metrics,
                n_bootstrap=1000,
                random_seed=42
            )

            print_bootstrap_results(bootstrap_results, metric_keys=metrics)
    else:
        # Single-head predictions
        per_video = compute_per_video_metrics(df)
        print(f"\nVideos: {len(per_video)}")

        metrics = ["Accuracy", "Macro_Precision", "Macro_Recall", "Macro_IoU", "Macro_F1"]
        bootstrap_results = bootstrap_per_video_metrics(
            per_video_results=per_video,
            metric_keys=metrics,
            n_bootstrap=1000,
            random_seed=42
        )

        print_bootstrap_results(bootstrap_results, metric_keys=metrics)

    print("="*70)
    print("TEST 4 COMPLETED SUCCESSFULLY")
    print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test bootstrap implementation")
    parser.add_argument("--csv", type=str, default=None,
                       help="Path to prediction CSV file for testing with real data")
    parser.add_argument("--test", type=str, default="all",
                       choices=["all", "synthetic", "few", "reproducibility", "csv"],
                       help="Which test to run")

    args = parser.parse_args()

    print("\n" + "="*70)
    print("BOOTSTRAP TESTING SUITE")
    print("="*70)

    if args.test in ["all", "synthetic"]:
        test_bootstrap_synthetic()

    if args.test in ["all", "few"]:
        test_bootstrap_with_few_videos()

    if args.test in ["all", "reproducibility"]:
        test_bootstrap_reproducibility()

    if args.test in ["csv"] or (args.test == "all" and args.csv):
        if args.csv:
            if os.path.exists(args.csv):
                test_bootstrap_from_csv(args.csv)
            else:
                print(f"\nERROR: CSV file not found: {args.csv}")
        else:
            print("\nSkipping CSV test (no --csv path provided)")

    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70)
