#!/usr/bin/env python3
"""
验证 predictions CSV 是否与训练日志中的 val 指标一致。
用法:
    python scripts/verify_predictions.py \
        --csv  logs/cpt_bleeding/.../predictions_epoch_19.csv \
        --log  logs/foundation/bleeding_V2/bleeding_probe_attentive_clip_70_30_20260318_165350.log
"""
import re
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict


def compute_metrics_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    results = {}
    for head_id in sorted(df["head"].unique()):
        hdf = df[df["head"] == head_id]
        preds = hdf["prediction"].values
        labels = hdf["label"].values
        n = len(labels)

        acc = 100.0 * np.sum(preds == labels) / n

        classes = sorted(set(labels) | set(preds))
        per_class = {}
        for c in classes:
            tp = np.sum((preds == c) & (labels == c))
            fp = np.sum((preds == c) & (labels != c))
            fn = np.sum((preds != c) & (labels == c))
            prec = 100.0 * tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            iou = 100.0 * tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            per_class[c] = {"Prec": prec, "Rec": rec, "F1": f1, "IoU": iou}

        macro_prec = np.mean([v["Prec"] for v in per_class.values()])
        macro_rec = np.mean([v["Rec"] for v in per_class.values()])
        macro_f1 = np.mean([v["F1"] for v in per_class.values()])
        macro_iou = np.mean([v["IoU"] for v in per_class.values()])

        results[head_id] = {
            "Acc": acc, "F1": macro_f1, "IoU": macro_iou,
            "Prec": macro_prec, "Rec": macro_rec,
            "per_class": per_class, "n_samples": n,
        }
    return results


def parse_log_metrics(log_path):
    results = {}
    with open(log_path) as f:
        lines = f.readlines()

    last_epoch_lines = []
    last_epoch_start = 0
    for i, line in enumerate(lines):
        if "BEST HEAD" in line:
            last_epoch_start = i

    search_start = max(0, last_epoch_start - 200)
    block = lines[search_start:]

    for line in block:
        m = re.search(
            r"head_(\d+): Acc=([\d.]+)±[\d.]+, F1=([\d.]+)±[\d.]+, "
            r"IoU=([\d.]+)±[\d.]+, Prec=([\d.]+)±[\d.]+, Rec=([\d.]+)±[\d.]+",
            line,
        )
        if m:
            hid = int(m.group(1))
            results[hid] = {
                "Acc": float(m.group(2)),
                "F1": float(m.group(3)),
                "IoU": float(m.group(4)),
                "Prec": float(m.group(5)),
                "Rec": float(m.group(6)),
            }

        m_class = re.search(
            r"Phase_(\d+): Prec=([\d.]+)%, Rec=([\d.]+)%, F1=([\d.]+)%, IoU=([\d.]+)%",
            line,
        )

    best_m = re.search(r"BEST HEAD: head_(\d+) \(Macro_F1=([\d.]+)\)", "".join(block))
    best_head = int(best_m.group(1)) if best_m else None
    best_f1 = float(best_m.group(2)) if best_m else None

    return results, best_head, best_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()

    print("=" * 70)
    print("从 CSV 重新计算指标")
    print("=" * 70)
    csv_metrics = compute_metrics_from_csv(args.csv)

    print(f"CSV 文件: {args.csv}")
    print(f"样本数/head: {csv_metrics[0]['n_samples']}")
    print()

    print("=" * 70)
    print("从 Log 解析指标")
    print("=" * 70)
    log_metrics, best_head, best_f1 = parse_log_metrics(args.log)
    print(f"Log 文件: {args.log}")
    print(f"Log BEST HEAD: head_{best_head} (Macro_F1={best_f1})")
    print()

    csv_best_head = max(csv_metrics, key=lambda h: csv_metrics[h]["F1"])
    csv_best_f1 = csv_metrics[csv_best_head]["F1"]
    print(f"CSV BEST HEAD: head_{csv_best_head} (Macro_F1={csv_best_f1:.2f})")
    match_best = (csv_best_head == best_head)
    print(f"Best head 一致: {'YES' if match_best else 'NO'}")
    print()

    print("=" * 70)
    print(f"{'Head':>6} | {'Metric':>6} | {'CSV':>10} | {'Log':>10} | {'Diff':>10} | {'Match':>5}")
    print("-" * 70)

    all_match = True
    for hid in sorted(csv_metrics.keys()):
        csv_m = csv_metrics[hid]
        if hid not in log_metrics:
            print(f"head_{hid:>2} | Log 中未找到")
            all_match = False
            continue
        log_m = log_metrics[hid]

        for metric in ["Acc", "F1", "IoU", "Prec", "Rec"]:
            csv_val = csv_m[metric]
            log_val = log_m[metric]
            diff = abs(csv_val - log_val)
            ok = diff < 1.0  # bootstrap 会引入方差，允许 <1% 误差
            if not ok:
                all_match = False
            print(f"head_{hid:>2} | {metric:>6} | {csv_val:>9.2f}% | {log_val:>9.2f}% | {diff:>9.2f}% | {'OK' if ok else 'DIFF'}")

    print("=" * 70)
    if all_match:
        print("结论: CSV 与 Log 的指标一致 (误差 < 1%)")
    else:
        print("结论: 存在不一致项，请检查上方标记为 DIFF 的行")
        print("注意: Log 中的指标带有 bootstrap 置信区间 (±)，CSV 直接计算不含 bootstrap，")
        print("      因此 Acc 应精确匹配，但 F1/Prec/Rec/IoU 可能有微小差异。")


if __name__ == "__main__":
    main()
