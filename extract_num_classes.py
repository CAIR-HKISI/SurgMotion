#!/usr/bin/env python3
"""Quick script to extract number of classes from surgical dataset CSV files."""

import pandas as pd
from pathlib import Path

# Define dataset paths (update base path if needed)
base_path = "/home/jinlin_wu/NSJepa/data/Surge_Frames"
datasets = [
    "Cholec80",
    "M2CAI16",
    "AutoLaparo",
    "BernBypass",
    "EgoSurgery",
    "OphNet2024_phase",
    "PitVis",
    "PmLR50",
    "StrasBypass70"
]

print("Number of classes per dataset:")
print("-" * 50)

num_classes_list = []
for dataset in datasets:
    csv_path = Path(base_path) / dataset / "clips_64f" / "train_dense_64f_detailed.csv"

    try:
        df = pd.read_csv(csv_path)
        num_classes = df['label'].nunique()
        num_classes_list.append(num_classes)
        print(f"{dataset:25s}: {num_classes} classes")
    except FileNotFoundError:
        print(f"{dataset:25s}: FILE NOT FOUND")
        num_classes_list.append("?")
    except Exception as e:
        print(f"{dataset:25s}: ERROR - {e}")
        num_classes_list.append("?")

print("\n" + "=" * 50)
print("YAML format for copy-paste:")
print("=" * 50)
print("num_classes:")
for dataset, num_classes in zip(datasets, num_classes_list):
    print(f"- {num_classes}   # {dataset}")
