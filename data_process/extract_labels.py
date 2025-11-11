#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

# 要处理的文件列表
files = ['pumch.json', 'pwh.json', 'TSS.json']

# 创建一个集合来存储所有不重复的 timelinelabels
timeline_labels_set = set()

# 用于记录每个文件的统计信息
file_stats = {}

# 遍历所有文件
for filename in files:
    print(f"正在处理: {filename}")
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    file_labels = set()
    
    # 遍历所有任务
    for task in data:
        # 遍历每个任务的 annotations
        if 'annotations' in task:
            for annotation in task['annotations']:
                # 遍历每个 annotation 的 result
                if 'result' in annotation:
                    for result_item in annotation['result']:
                        # 检查是否有 timelinelabels
                        if 'value' in result_item and 'timelinelabels' in result_item['value']:
                            # 提取 timelinelabels 列表中的每个标签
                            for label in result_item['value']['timelinelabels']:
                                timeline_labels_set.add(label)
                                file_labels.add(label)
    
    file_stats[filename] = {
        'unique_labels': len(file_labels),
        'labels': sorted(list(file_labels))
    }
    print(f"  - 找到 {len(file_labels)} 个不重复的标签")

# 打印总体结果
print(f"\n{'='*60}")
print(f"所有文件合并后共找到 {len(timeline_labels_set)} 个不重复的 timelinelabels：\n")
for i, label in enumerate(sorted(timeline_labels_set), 1):
    print(f"{i:2d}. {label}")

# 打印每个文件的详细统计
print(f"\n{'='*60}")
print("各文件详细统计：\n")
for filename, stats in file_stats.items():
    print(f"【{filename}】 - {stats['unique_labels']} 个标签:")
    for label in stats['labels']:
        print(f"  - {label}")
    print()

# 保存到文件
output = {
    "all_unique_timeline_labels": sorted(list(timeline_labels_set)),
    "total_count": len(timeline_labels_set),
    "file_statistics": {
        filename: {
            "count": stats['unique_labels'],
            "labels": stats['labels']
        }
        for filename, stats in file_stats.items()
    }
}

output_filename = 'all_timeline_labels.json'
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"{'='*60}")
print(f"结果已保存到 {output_filename}")

