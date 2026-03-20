#!/usr/bin/env python3
"""
从 predictions_epoch_0.csv 中提取 new_Nonbleeding (NNB) 子集并评估。

映射链路:
  predictions.data_idx  ==  test_clips_csv.Index
  test_clips_csv.case_id  ==  test_metadata.Case_ID
  test_metadata.Case_Name 以 "NNB_" 开头  →  来自 new_Nonbleeding

评估指标与 eval.py 保持一致:
  Accuracy, Macro Precision, Macro Recall, Macro F1, Macro IoU, per-class 指标
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    jaccard_score, accuracy_score, confusion_matrix,
    classification_report
)
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data/Surge_Frames/Bleeding_Dataset_70_30")
CLIPS_CSV = os.path.join(DATA_DIR, "clips_64f/test_dense_64f_detailed.csv")
META_CSV = os.path.join(DATA_DIR, "test_metadata.csv")
CASE_CACHE = os.path.join(DATA_DIR, "_case_cache.json")

PRED_DIRS = [
    "logs/cpt_bleeding/bleeding_vitl16-256px-64f_multi-head/video_classification_frozen/bleeding-probing",
    "logs/foundation/endossl_vitl_laparo_bleeding/video_classification_frozen/endossl_vitl_laparo_64f_bleeding",
    "logs/foundation/dinov3_vitl_bleeding/video_classification_frozen/dinov3_vitl_64f_bleeding",
    "logs/foundation/endomamba_small_bleeding/video_classification_frozen/endomamba_small_64f_bleeding",
    "logs/foundation/selfsupsurg_res50_Bleeding/video_classification_frozen/selfsupsurg_res50_64f_Bleeding",
    "logs/foundation/surgenet_convnextv2_bleeding/video_classification_frozen/surgenet_convnextv2_64f_bleeding",
    "logs/foundation/gsvit_vit_bleeding/video_classification_frozen/gsvit_vit_64f_bleeding",
    "logs/foundation/videomaev2_large_bleeding/video_classification_frozen/videomaev2_large_64f_bleeding",
    "logs/foundation/endofm_vitb_bleeding/video_classification_frozen/endofm_vitb_64f_bleeding",
    "logs/foundation/gastronet_vits_bleeding/video_classification_frozen/gastronet_vits_64f_bleeding",
    "logs/foundation/endovit_vitl_laparo_bleeding/video_classification_frozen/endovit_vitl_laparo_64f_bleeding",
    "logs/foundation/surgvlp_res50_bleeding/video_classification_frozen/surgvlp_res50_64f_bleeding",
]


def build_data_idx_to_case_name(clips_csv_path, meta_csv_path):
    """
    构建 data_idx -> (case_id, case_name, source_category) 的映射。
    """
    clips_df = pd.read_csv(clips_csv_path)
    meta_df = pd.read_csv(meta_csv_path)

    caseid_to_name = dict(zip(meta_df['Case_ID'], meta_df['Case_Name']))

    mapping = {}
    for _, row in clips_df.iterrows():
        idx = int(row['Index'])
        cid = int(row['case_id'])
        cname = caseid_to_name.get(cid, f"Unknown_{cid}")

        if cname.startswith("NNB_"):
            src = "new_Nonbleeding"
        elif cname.startswith("Bleeding_"):
            src = "Bleeding"
        elif cname.startswith("NonBleeding_"):
            src = "Non_bleeding"
        else:
            src = "unknown"

        true_label = int(row['label'])
        mapping[idx] = {"case_id": cid, "case_name": cname, "source": src, "true_label": true_label}

    return mapping


def evaluate_subset(pred_df, subset_name="NNB"):
    """
    与 eval.py 的 evaluate_global_action 保持一致的评估方式。
    """
    all_gt = pred_df['label'].values
    all_pred = pred_df['prediction'].values

    unique_classes = np.unique(np.concatenate([all_gt, all_pred]))

    acc = accuracy_score(all_gt, all_pred) * 100
    macro_prec = precision_score(all_gt, all_pred, average='macro', zero_division=0) * 100
    macro_rec = recall_score(all_gt, all_pred, average='macro', zero_division=0) * 100
    macro_iou = jaccard_score(all_gt, all_pred, average='macro', zero_division=0) * 100
    macro_f1 = f1_score(all_gt, all_pred, average='macro', zero_division=0) * 100

    per_class_prec = precision_score(all_gt, all_pred, labels=unique_classes, average=None, zero_division=0) * 100
    per_class_rec = recall_score(all_gt, all_pred, labels=unique_classes, average=None, zero_division=0) * 100
    per_class_f1_arr = f1_score(all_gt, all_pred, labels=unique_classes, average=None, zero_division=0) * 100
    per_class_iou = jaccard_score(all_gt, all_pred, labels=unique_classes, average=None, zero_division=0) * 100

    stats = {
        "Accuracy": acc,
        "Macro_Precision": macro_prec,
        "Macro_Recall": macro_rec,
        "Macro_F1": macro_f1,
        "Macro_IoU": macro_iou,
    }

    per_class = {}
    label_names = {0: "non_bleeding", 1: "bleeding"}
    for i, cls in enumerate(unique_classes):
        per_class[label_names.get(cls, f"Phase_{cls}")] = {
            "Precision": per_class_prec[i],
            "Recall": per_class_rec[i],
            "F1": per_class_f1_arr[i],
            "IoU": per_class_iou[i],
        }
    stats["per_class"] = per_class

    return stats


def evaluate_per_video_subset(pred_df, subset_name="NNB"):
    """
    与 eval.py 的 evaluate_per_video 保持一致的 per-video 评估方式。
    """
    pred_df = pred_df.sort_values(['vid', 'data_idx'])

    all_gt = pred_df['label'].values
    all_pred = pred_df['prediction'].values
    unique_classes = np.unique(np.concatenate([all_gt, all_pred]))

    per_video = []
    for vid, subdf in pred_df.groupby('vid'):
        subdf = subdf.sort_values('data_idx')
        gt = subdf['label'].values
        pred = subdf['prediction'].values

        v_acc = accuracy_score(gt, pred) * 100
        v_prec = precision_score(gt, pred, average='macro', zero_division=0) * 100
        v_rec = recall_score(gt, pred, average='macro', zero_division=0) * 100
        v_f1 = f1_score(gt, pred, average='macro', zero_division=0) * 100
        v_iou = jaccard_score(gt, pred, average='macro', zero_division=0) * 100

        case_name = subdf['case_name'].iloc[0] if 'case_name' in subdf.columns else str(vid)
        per_video.append({
            "vid": vid,
            "case_name": case_name,
            "n_samples": len(gt),
            "Accuracy": v_acc,
            "Macro_Precision": v_prec,
            "Macro_Recall": v_rec,
            "Macro_F1": v_f1,
            "Macro_IoU": v_iou,
        })

    video_metrics = pd.DataFrame(per_video)
    stats = {}
    for m in ["Accuracy", "Macro_Precision", "Macro_Recall", "Macro_F1", "Macro_IoU"]:
        stats[f"{m}_Mean"] = video_metrics[m].mean()
        stats[f"{m}_Std"] = video_metrics[m].std()

    return per_video, stats


def process_one_prediction_file(pred_csv_path, idx_to_info, output_dir=None):
    """处理单个 predictions_epoch_0.csv 文件。"""
    pred_df = pd.read_csv(pred_csv_path)

    pred_df['case_id_mapped'] = pred_df['data_idx'].map(
        lambda x: idx_to_info.get(x, {}).get('case_id', -1)
    )
    pred_df['case_name'] = pred_df['data_idx'].map(
        lambda x: idx_to_info.get(x, {}).get('case_name', 'unknown')
    )
    pred_df['source_category'] = pred_df['data_idx'].map(
        lambda x: idx_to_info.get(x, {}).get('source', 'unknown')
    )
    pred_df['true_label'] = pred_df['data_idx'].map(
        lambda x: idx_to_info.get(x, {}).get('true_label', -1)
    )
    pred_df['is_nnb'] = pred_df['case_name'].str.startswith('NNB_')

    label_mismatch = pred_df[(pred_df['true_label'] >= 0) & (pred_df['label'] != pred_df['true_label'])]
    if len(label_mismatch) > 0:
        n_mismatch = len(label_mismatch) // pred_df['head'].nunique()
        print(f"[WARNING] 发现 {n_mismatch} 个 data_idx 的 label 与 clips CSV 不一致，已使用 clips CSV 的真实标签")
        mask = pred_df['true_label'] >= 0
        pred_df.loc[mask, 'label'] = pred_df.loc[mask, 'true_label']

    nnb_df = pred_df[pred_df['is_nnb']].copy()

    n_total = len(pred_df)
    n_nnb = len(nnb_df)
    n_heads = pred_df['head'].nunique()

    model_name = os.path.basename(os.path.dirname(pred_csv_path))
    parent_name = os.path.basename(os.path.dirname(os.path.dirname(pred_csv_path)))

    print("=" * 70)
    print(f"模型: {parent_name}/{model_name}")
    print(f"预测文件: {pred_csv_path}")
    print(f"总样本数: {n_total}, NNB 样本数: {n_nnb} ({n_nnb/n_total*100:.1f}%)")
    print(f"Heads 数: {n_heads}")
    print("=" * 70)

    nnb_cases = nnb_df.drop_duplicates('case_name')['case_name'].tolist()
    print(f"\nNNB 案例列表 ({len(set(nnb_cases))} 个): {sorted(set(nnb_cases))}")

    source_counts = pred_df.drop_duplicates('data_idx').groupby('source_category').size()
    print(f"\n数据来源分布 (per data_idx):")
    for src, cnt in source_counts.items():
        print(f"  {src}: {cnt}")

    if n_nnb == 0:
        print("\n[WARNING] 没有找到 NNB 样本!")
        return None

    # --- Step 1: 找到全量测试集上的 best head (按 Macro F1) ---
    known_df = pred_df[pred_df['true_label'] >= 0]
    full_best_head_id = -1
    full_best_f1 = -1
    full_head_results = {}

    for head_id in sorted(known_df['head'].unique()):
        head_df = known_df[known_df['head'] == head_id]
        stats = evaluate_subset(head_df, subset_name="Full")
        full_head_results[head_id] = stats
        if stats['Macro_F1'] > full_best_f1:
            full_best_f1 = stats['Macro_F1']
            full_best_head_id = head_id

    print(f"\n全量测试集 Best Head (按 Macro F1): head_{full_best_head_id} (F1={full_best_f1:.2f}%)")

    # --- Step 2: Per-head NNB 评估 + False Positive Rate ---
    print("\n" + "=" * 70)
    print("=== NNB 子集 Per-Head 评估 ===")
    print("=== (NNB 数据全部为 non_bleeding, 关注误报率 FPR) ===")
    print("=" * 70)

    n_nnb_per_head = n_nnb // n_heads
    all_head_nnb_results = {}

    print(f"\n{'Head':>6} | {'NNB Acc':>8} | {'FPR':>7} | {'FP数':>5} | {'Full Acc':>9} | {'Full F1':>8} | {'Full IoU':>8}")
    print("-" * 80)

    for head_id in sorted(nnb_df['head'].unique()):
        head_nnb = nnb_df[nnb_df['head'] == head_id]
        n_fp = (head_nnb['prediction'] == 1).sum()
        nnb_acc = (head_nnb['prediction'] == head_nnb['label']).mean() * 100
        fpr = n_fp / len(head_nnb) * 100  # False Positive Rate on NNB

        full_stats = full_head_results.get(head_id, {})
        full_acc = full_stats.get('Accuracy', 0)
        full_f1 = full_stats.get('Macro_F1', 0)
        full_iou = full_stats.get('Macro_IoU', 0)

        marker = " <-- Full Best" if head_id == full_best_head_id else ""

        all_head_nnb_results[head_id] = {
            "nnb_accuracy": nnb_acc,
            "fpr": fpr,
            "n_false_positives": int(n_fp),
            "full_accuracy": full_acc,
            "full_f1": full_f1,
            "full_iou": full_iou,
        }

        print(f"  {head_id:>4} | {nnb_acc:>7.2f}% | {fpr:>6.2f}% | {n_fp:>5} | {full_acc:>8.2f}% | {full_f1:>7.2f}% | {full_iou:>7.2f}%{marker}")

    # --- Step 3: 用全量 Best Head 评估 NNB ---
    best_head_nnb = nnb_df[nnb_df['head'] == full_best_head_id]
    best_nnb_stats = evaluate_subset(best_head_nnb, subset_name="NNB")
    best_n_fp = (best_head_nnb['prediction'] == 1).sum()
    best_fpr = best_n_fp / len(best_head_nnb) * 100

    print(f"\n{'='*70}")
    print(f"使用全量测试集 Best Head (head_{full_best_head_id}) 评估 NNB 子集:")
    print(f"{'='*70}")
    print(f"  NNB Accuracy:     {best_nnb_stats['Accuracy']:.2f}%")
    print(f"  False Positive Rate: {best_fpr:.2f}% ({best_n_fp}/{len(best_head_nnb)} 样本被误判为 bleeding)")
    if "per_class" in best_nnb_stats:
        for cls_name, cls_metrics in best_nnb_stats["per_class"].items():
            print(f"  {cls_name}: Prec={cls_metrics['Precision']:.2f}%, Rec={cls_metrics['Recall']:.2f}%, F1={cls_metrics['F1']:.2f}%, IoU={cls_metrics['IoU']:.2f}%")

    # --- Step 4: Per-video 评估 (Full Best head) ---
    print(f"\n{'-'*70}")
    print(f"=== NNB 子集 Per-Video 评估 (Full Best Head {full_best_head_id}) ===")
    print(f"{'-'*70}")
    per_video, pv_stats = evaluate_per_video_subset(best_head_nnb)

    for v in per_video:
        n_fp_v = len(best_head_nnb[(best_head_nnb['vid'] == v['vid']) & (best_head_nnb['prediction'] == 1)])
        fpr_v = n_fp_v / v['n_samples'] * 100
        print(
            f"  {v['case_name']:>15}: "
            f"Acc={v['Accuracy']:.2f}%, F1={v['Macro_F1']:.2f}%, "
            f"FPR={fpr_v:.2f}% ({n_fp_v}/{v['n_samples']})"
        )

    print(f"\n  Per-Video 均值:")
    for m in ["Accuracy", "Macro_Precision", "Macro_Recall", "Macro_F1", "Macro_IoU"]:
        print(f"    {m}: {pv_stats[f'{m}_Mean']:.2f} ± {pv_stats[f'{m}_Std']:.2f}")

    # --- Step 5: 三方对比 (Full Best head) ---
    print(f"\n{'-'*70}")
    print(f"=== 对比: 全量 vs Non-NNB vs NNB (Full Best Head {full_best_head_id}) ===")
    print(f"{'-'*70}")

    best_all = known_df[known_df['head'] == full_best_head_id]
    best_non_nnb = best_all[~best_all['is_nnb']]

    for subset_name, subset_df in [("全量测试集", best_all), ("排除NNB", best_non_nnb), ("仅NNB", best_head_nnb)]:
        s = evaluate_subset(subset_df)
        print(
            f"  {subset_name:>10} (n={len(subset_df)}): "
            f"Acc={s['Accuracy']:.2f}%, "
            f"F1={s['Macro_F1']:.2f}%, "
            f"IoU={s['Macro_IoU']:.2f}%"
        )

    # --- 保存结果 ---
    if output_dir is None:
        output_dir = os.path.dirname(pred_csv_path)

    nnb_pred_path = os.path.join(output_dir, "predictions_epoch_0_nnb_only.csv")
    nnb_df.to_csv(nnb_pred_path, index=False)
    print(f"\nNNB 预测已保存: {nnb_pred_path}")

    with_source_path = os.path.join(output_dir, "predictions_epoch_0_with_source.csv")
    pred_df.to_csv(with_source_path, index=False)
    print(f"带来源标记的预测已保存: {with_source_path}")

    results_path = os.path.join(output_dir, "nnb_evaluation_results.txt")
    with open(results_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"NNB Subset Evaluation Results\n")
        f.write(f"Model: {parent_name}/{model_name}\n")
        f.write(f"Full-test Best Head: head_{full_best_head_id}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total test samples: {len(best_all)}, NNB samples: {n_nnb_per_head}\n")
        f.write(f"NNB False Positive Rate: {best_fpr:.2f}% ({best_n_fp}/{n_nnb_per_head})\n\n")

        f.write(f"{'Head':>6} | {'NNB Acc':>8} | {'FPR':>7} | {'FP数':>5} | {'Full Acc':>9} | {'Full F1':>8}\n")
        f.write("-" * 60 + "\n")
        for head_id, hr in sorted(all_head_nnb_results.items()):
            marker = " *" if head_id == full_best_head_id else ""
            f.write(f"  {head_id:>4} | {hr['nnb_accuracy']:>7.2f}% | {hr['fpr']:>6.2f}% | {hr['n_false_positives']:>5} | {hr['full_accuracy']:>8.2f}% | {hr['full_f1']:>7.2f}%{marker}\n")

    print(f"评估结果已保存: {results_path}")

    return {
        "model": f"{parent_name}/{model_name}",
        "full_best_head": full_best_head_id,
        "nnb_accuracy": best_nnb_stats['Accuracy'],
        "nnb_fpr": best_fpr,
        "full_f1": full_best_f1,
        "full_acc": full_head_results[full_best_head_id]['Accuracy'],
        "all_head_nnb_results": all_head_nnb_results,
        "n_nnb": n_nnb_per_head,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate NNB subset from predictions")
    parser.add_argument("--pred_csv", type=str, default=None,
                        help="Path to a single predictions_epoch_0.csv")
    parser.add_argument("--all", action="store_true",
                        help="Process all known prediction files")
    parser.add_argument("--clips_csv", type=str, default=CLIPS_CSV)
    parser.add_argument("--meta_csv", type=str, default=META_CSV)
    args = parser.parse_args()

    print("构建 data_idx -> Case_Name 映射...")
    idx_to_info = build_data_idx_to_case_name(args.clips_csv, args.meta_csv)
    print(f"映射表大小: {len(idx_to_info)}")

    if args.pred_csv:
        pred_files = [args.pred_csv]
    elif args.all:
        pred_files = []
        for d in PRED_DIRS:
            p = os.path.join(BASE_DIR, d, "predictions_epoch_0.csv")
            if os.path.exists(p):
                pred_files.append(p)
        if not pred_files:
            print("未找到任何预测文件!")
            return
    else:
        default = os.path.join(BASE_DIR, PRED_DIRS[0], "predictions_epoch_0.csv")
        if os.path.exists(default):
            pred_files = [default]
        else:
            print(f"默认预测文件不存在: {default}")
            print("请使用 --pred_csv 指定路径，或 --all 处理所有模型")
            return

    all_results = []
    for pf in pred_files:
        result = process_one_prediction_file(pf, idx_to_info)
        if result:
            all_results.append(result)
        print("\n\n")

    if len(all_results) > 1:
        print("\n" + "=" * 110)
        print("=== 所有模型对比: 全量测试集 Best Head 在 NNB 子集上的表现 ===")
        print("=" * 110)
        print(f"{'Model':<55} {'Head':>5} {'Full Acc':>9} {'Full F1':>8} {'NNB Acc':>8} {'NNB FPR':>8} {'NNB n':>6}")
        print("-" * 110)
        for r in all_results:
            print(
                f"{r['model']:<55} "
                f"{r['full_best_head']:>5} "
                f"{r['full_acc']:>8.2f}% "
                f"{r['full_f1']:>7.2f}% "
                f"{r['nnb_accuracy']:>7.2f}% "
                f"{r['nnb_fpr']:>7.2f}% "
                f"{r['n_nnb']:>6}"
            )


if __name__ == "__main__":
    main()
