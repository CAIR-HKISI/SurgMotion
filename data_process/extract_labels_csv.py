#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from collections import Counter

# 读取 CSV 文件
csv_file = '/home/chen_chuxi/NSJepa/data/Surge_Frames/PitVis/clips_64f/test_dense_64f_detailed.csv'

label_names = []
unique_labels = set()

with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        label_name = row['label_name']
        label_names.append(label_name)
        unique_labels.add(label_name)

# 统计每个标签的出现次数
label_counts = Counter(label_names)

# 打印结果
print(f"共找到 {len(unique_labels)} 个不重复的 label names：\n")
print("="*60)

for i, label in enumerate(sorted(unique_labels), 1):
    count = label_counts[label]
    print(f"{i:2d}. {label:<40} (出现 {count:5d} 次)")

print("\n" + "="*60)
print(f"总计: {len(label_names)} 条记录")
print(f"不重复标签数: {len(unique_labels)}")

# 保存到文件
import json

output = {
    "unique_label_names": sorted(list(unique_labels)),
    "total_count": len(unique_labels),
    "label_statistics": {
        label: label_counts[label]
        for label in sorted(unique_labels)
    },
    "total_records": len(label_names)
}

output_file = './label_names_summary.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n结果已保存到: {output_file}")