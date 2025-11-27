#!/usr/bin/env python3
"""
计算每个probing head的评估指标
包括：accuracy, precision, recall, jaccard
"""

import csv
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score
)

def calculate_head_metrics(csv_path):
    """
    读取CSV文件并计算每个head的评估指标
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        list: 包含每个head指标的字典列表
    """
    # 读取CSV文件
    print(f"正在读取文件: {csv_path}")
    
    # 按head分组存储数据
    head_data = defaultdict(lambda: {'labels': [], 'predictions': []})
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            head = int(row['head'])
            label = int(row['label'])
            prediction = int(row['prediction'])
            head_data[head]['labels'].append(label)
            head_data[head]['predictions'].append(prediction)
    
    print(f"Head数量: {len(head_data)}")
    print(f"Head列表: {sorted(head_data.keys())}")
    
    # 存储每个head的指标
    results = []
    
    for head in sorted(head_data.keys()):
        # 获取当前head的数据
        y_true = head_data[head]['labels']
        y_pred = head_data[head]['predictions']
        
        # 计算指标
        accuracy = accuracy_score(y_true, y_pred) * 100
        
        # 对于多分类问题，使用macro平均
        # macro: 每个类别单独计算后平均
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
        
        # Jaccard score (IoU)
        jaccard = jaccard_score(y_true, y_pred, average='macro', zero_division=0) * 100
        
        results.append({
            'head': head,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'jaccard': jaccard,
            'num_samples': len(y_true)
        })
        
        print(f"\nHead {head}:")
        print(f"  Accuracy: {accuracy:.4f}%")
        print(f"  Precision: {precision:.4f}%")
        print(f"  Recall: {recall:.4f}%")
        print(f"  F1 Score: {f1:.4f}%")
        print(f"  Jaccard: {jaccard:.4f}%")
        print(f"  样本数: {len(y_true)}")
    
    # 保存结果到CSV
    output_path = csv_path.replace('.csv', '_metrics.csv')
    with open(output_path, 'w', newline='') as f:
        fieldnames = ['head', 'accuracy', 'precision', 'recall', 'f1', 'jaccard', 'num_samples']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n结果已保存到: {output_path}")
    
    # 打印汇总统计
    print("\n" + "="*60)
    print("汇总统计:")
    print("="*60)
    def mean(values):
        return sum(values) / len(values) if len(values) > 0 else 0.0
    
    avg_accuracy = mean([r['accuracy'] for r in results])
    avg_precision = mean([r['precision'] for r in results])
    avg_recall = mean([r['recall'] for r in results])
    avg_f1 = mean([r['f1'] for r in results])
    avg_jaccard = mean([r['jaccard'] for r in results])
    
    print(f"平均 Accuracy: {avg_accuracy:.4f}%")
    print(f"平均 Precision: {avg_precision:.4f}%")
    print(f"平均 Recall: {avg_recall:.4f}%")
    print(f"平均 F1 Score: {avg_f1:.4f}%")
    print(f"平均 Jaccard: {avg_jaccard:.4f}%")
    
    return results

if __name__ == '__main__':
    import sys
    
    # CSV文件路径
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = 'logs10/egosurgery_predictions_epoch_0.csv'
    
    # 计算指标
    results = calculate_head_metrics(csv_path)
    
    # 显示结果表格
    print("\n" + "="*60)
    print("详细结果表格:")
    print("="*60)
    print(f"{'Head':<6} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Jaccard':<12} {'Samples':<8}")
    print("-" * 70)
    for r in results:
        print(f"{r['head']:<6} {r['accuracy']:<12.4f} {r['precision']:<12.4f} {r['recall']:<12.4f} "
              f"{r['f1']:<12.4f} {r['jaccard']:<12.4f} {r['num_samples']:<8}")

