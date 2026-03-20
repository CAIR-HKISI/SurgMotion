#!/usr/bin/env python3
"""
从 predictions_epoch_0.csv 提取 NNB 子集并按 eval.py 方式评估。

完整数据映射链路:
    predictions.data_idx
        → test_dense_64f_detailed.csv 的 Index 列
        → case_id
        → test_metadata.csv 的 Case_ID
        → Case_Name (NNB_* 开头 = new_Nonbleeding)
        → _case_cache.json
        → data/raw_bleeding/new_Nonbleeding/*.mp4 (原始视频)

用法:
    python eval_nnb_from_predictions.py
    python eval_nnb_from_predictions.py --pred_csv <其他预测文件路径>
    python eval_nnb_from_predictions.py --best_head 8        # 指定 head 而非自动选
    python eval_nnb_from_predictions.py --plot_head 3 8      # 为 head 3 和 8 绘制四象限图
    python eval_nnb_from_predictions.py --plot_head 0 1 2 3  # 为多个 head 绘图
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, jaccard_score, confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data/Surge_Frames/Bleeding_V2")

DEFAULTS = {
    "pred_csv": os.path.join(
        BASE,
        "logs/cpt_bleeding/bleeding_vitl16-256px-64f_multi-head/"
        "video_classification_frozen/bleeding-probing/predictions_epoch_14.csv",
    ),
    "clips_csv": os.path.join(DATA, "clips_whole/val_clips.csv"),
    "meta_csv": os.path.join(DATA, "val_metadata.csv"),
    "case_cache": os.path.join(DATA, "_case_cache.json"),
}


# ──────────────────────────────────────────────
# Step 1: 构建 data_idx → case 信息 的映射表
# ──────────────────────────────────────────────
def build_mapping(clips_csv: str, meta_csv: str) -> dict:
    """
    返回 {data_idx: {case_id, case_name, source, true_label, clip_idx,
                      start_frame, end_frame, clip_path}}
    """
    clips = pd.read_csv(clips_csv)
    meta = pd.read_csv(meta_csv)
    id2name = dict(zip(meta["Case_ID"], meta["Case_Name"]))

    mapping = {}
    for _, r in clips.iterrows():
        idx = int(r["Index"])
        cid = int(r["case_id"])
        cname = id2name.get(cid, f"Unknown_{cid}")
        if cname.startswith("NNB_"):
            src = "new_Nonbleeding"
        elif cname.startswith("Bleeding_"):
            src = "Bleeding"
        elif cname.startswith("NonBleeding_"):
            src = "Non_bleeding"
        else:
            src = "unknown"
        mapping[idx] = {
            "case_id": cid,
            "case_name": cname,
            "source": src,
            "true_label": int(r["label"]),
            "clip_idx": int(r["clip_idx"]),
            "start_frame": int(r["start_frame"]),
            "end_frame": int(r["end_frame"]),
            "clip_path": r["clip_path"],
        }
    return mapping


# ──────────────────────────────────────────────
# Step 2: 从 predictions 中提取 NNB 子集
# ──────────────────────────────────────────────
def extract_nnb(pred_df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """
    给 pred_df 添加 case_name 列, 修正 label, 返回 NNB 子集。
    """
    pred_df = pred_df.copy()
    pred_df["case_id_mapped"] = pred_df["data_idx"].map(
        lambda x: mapping.get(x, {}).get("case_id", -1)
    )
    pred_df["case_name"] = pred_df["data_idx"].map(
        lambda x: mapping.get(x, {}).get("case_name", "unknown")
    )
    pred_df["source"] = pred_df["data_idx"].map(
        lambda x: mapping.get(x, {}).get("source", "unknown")
    )
    # 用 clips CSV 的真实标签修正 (防止 dataloader hash 碰撞导致标签错位)
    true_labels = pred_df["data_idx"].map(
        lambda x: mapping.get(x, {}).get("true_label", -1)
    )
    mask = true_labels >= 0
    pred_df.loc[mask, "label"] = true_labels[mask]

    nnb = pred_df[pred_df["case_name"].str.startswith("NNB_")].copy()
    return nnb


# ──────────────────────────────────────────────
# Step 3: 按 eval.py evaluate_global_action 方式评估
# ──────────────────────────────────────────────
def evaluate_global_action(predictions_df: pd.DataFrame) -> dict:
    """与 eval.py 的 evaluate_global_action 完全一致。"""
    all_gt = predictions_df["label"].values
    all_pred = predictions_df["prediction"].values
    unique_classes = np.unique(np.concatenate([all_gt, all_pred]))

    stats = {
        "Accuracy_Mean": accuracy_score(all_gt, all_pred) * 100,
        "Macro_Precision_Mean": precision_score(all_gt, all_pred, average="macro", zero_division=0) * 100,
        "Macro_Recall_Mean": recall_score(all_gt, all_pred, average="macro", zero_division=0) * 100,
        "Macro_IoU_Mean": jaccard_score(all_gt, all_pred, average="macro", zero_division=0) * 100,
        "Macro_F1_Mean": f1_score(all_gt, all_pred, average="macro", zero_division=0) * 100,
    }

    pc_prec = precision_score(all_gt, all_pred, labels=unique_classes, average=None, zero_division=0) * 100
    pc_rec = recall_score(all_gt, all_pred, labels=unique_classes, average=None, zero_division=0) * 100
    pc_f1 = f1_score(all_gt, all_pred, labels=unique_classes, average=None, zero_division=0) * 100
    pc_iou = jaccard_score(all_gt, all_pred, labels=unique_classes, average=None, zero_division=0) * 100

    label_names = {0: "non_bleeding", 1: "bleeding"}
    per_class = {}
    for i, cls in enumerate(unique_classes):
        per_class[label_names.get(cls, f"Phase_{cls}")] = {
            "Precision": pc_prec[i], "Recall": pc_rec[i],
            "F1": pc_f1[i], "IoU": pc_iou[i],
        }
    stats["per_class"] = per_class
    return stats


# ──────────────────────────────────────────────
# Step 4: 定位回原始视频帧
# ──────────────────────────────────────────────
def locate_frames(data_idx: int, mapping: dict, case_cache: dict, meta_df: pd.DataFrame):
    """
    给定一个 data_idx, 返回:
      - clip 覆盖的帧路径列表 (从 txt 文件读取)
      - 对应的 Case_Name 和原始 mp4 列表
    """
    info = mapping.get(data_idx)
    if not info:
        return None

    # 读 clip txt 获取帧路径
    frame_paths = []
    if os.path.exists(info["clip_path"]):
        with open(info["clip_path"]) as f:
            frame_paths = [l.strip() for l in f]

    # 从 metadata 获取该 case 的帧详情
    case_meta = meta_df[meta_df["Case_ID"] == info["case_id"]].sort_values("Frame_Path").reset_index(drop=True)
    label_frame = case_meta.iloc[info["end_frame"] - 1] if info["end_frame"] <= len(case_meta) else None

    # 从 case_cache 获取原始 mp4
    raw_clips = []
    cache_entry = case_cache.get(info["case_name"], {})
    if cache_entry:
        raw_clips = [c["filepath"] for c in cache_entry.get("clips", [])]

    return {
        "case_name": info["case_name"],
        "case_id": info["case_id"],
        "clip_idx": info["clip_idx"],
        "start_frame": info["start_frame"],
        "end_frame": info["end_frame"],
        "frame_paths": frame_paths,
        "label_frame": label_frame["Frame_Path"] if label_frame is not None else None,
        "raw_mp4s": raw_clips,
    }


# ──────────────────────────────────────────────
# Step 5: TP/FP/TN/FN 四象限图
# ──────────────────────────────────────────────
def plot_confusion_quadrant(
    pred_df: pd.DataFrame,
    mapping: dict,
    head_id: int,
    out_path: str,
    title_suffix: str = "",
):
    """
    绘制指定 head 在全量测试集上的 TP/FP/TN/FN 四象限图,
    并用颜色标注 NNB / Bleeding / NonBleeding 三种数据来源。
    同时在右侧附一张 NNB-only 的四象限子图。
    """
    valid = pred_df[
        (pred_df["head"] == head_id) & pred_df["data_idx"].isin(mapping)
    ].copy()
    valid["true_label"] = valid["data_idx"].map(lambda x: mapping[x]["true_label"])
    valid["case_name"] = valid["data_idx"].map(lambda x: mapping[x]["case_name"])
    valid["source"] = valid["data_idx"].map(lambda x: mapping[x]["source"])

    gt = valid["true_label"].values
    pr = valid["prediction"].values

    tp_mask = (gt == 1) & (pr == 1)
    fp_mask = (gt == 0) & (pr == 1)
    tn_mask = (gt == 0) & (pr == 0)
    fn_mask = (gt == 1) & (pr == 0)

    quadrants = {
        "TP": {"mask": tp_mask, "pos": (1, 1), "color": "#2ecc71"},
        "FP": {"mask": fp_mask, "pos": (0, 1), "color": "#e74c3c"},
        "TN": {"mask": tn_mask, "pos": (0, 0), "color": "#3498db"},
        "FN": {"mask": fn_mask, "pos": (1, 0), "color": "#f39c12"},
    }

    src_colors = {
        "Bleeding": "#e74c3c",
        "Non_bleeding": "#3498db",
        "new_Nonbleeding": "#9b59b6",
    }
    src_labels = {
        "Bleeding": "Bleeding",
        "Non_bleeding": "NonBleeding",
        "new_Nonbleeding": "NNB (new_Nonbleeding)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [3, 2]})

    # ── 左图: 全量测试集四象限 ──
    ax = axes[0]
    ax.set_xlim(-0.05, 2.05)
    ax.set_ylim(-0.05, 2.05)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    for name, q in quadrants.items():
        x0, y0 = q["pos"]
        rect = mpatches.FancyBboxPatch(
            (x0 + 0.02, y0 + 0.02), 0.96, 0.96,
            boxstyle="round,pad=0.03",
            facecolor=q["color"], alpha=0.12, edgecolor=q["color"], linewidth=2,
        )
        ax.add_patch(rect)

        count = q["mask"].sum()
        total = len(gt)
        ax.text(
            x0 + 0.5, y0 + 0.78, name,
            ha="center", va="center", fontsize=28, fontweight="bold", color=q["color"],
        )
        ax.text(
            x0 + 0.5, y0 + 0.55, f"{count}",
            ha="center", va="center", fontsize=36, fontweight="bold", color=q["color"],
        )
        ax.text(
            x0 + 0.5, y0 + 0.38, f"({count/total*100:.1f}%)",
            ha="center", va="center", fontsize=14, color=q["color"], alpha=0.8,
        )

        sub = valid[q["mask"]]
        src_counts = sub["source"].value_counts()
        y_off = 0.22
        for src_name in ["Bleeding", "Non_bleeding", "new_Nonbleeding"]:
            sc = src_counts.get(src_name, 0)
            if sc > 0:
                ax.text(
                    x0 + 0.5, y0 + y_off,
                    f"  {src_labels.get(src_name, src_name)}: {sc}",
                    ha="center", va="center", fontsize=10,
                    color=src_colors.get(src_name, "gray"),
                )
                y_off -= 0.10

    ax.plot([1, 1], [0, 2], "k-", linewidth=1, alpha=0.3)
    ax.plot([0, 2], [1, 1], "k-", linewidth=1, alpha=0.3)

    ax.text(0.5, 2.02, "GT: Non-bleeding", ha="center", va="bottom", fontsize=12, color="gray")
    ax.text(1.5, 2.02, "GT: Bleeding", ha="center", va="bottom", fontsize=12, color="gray")
    ax.text(-0.03, 0.5, "Pred:\nNon-bleeding", ha="right", va="center", fontsize=11, color="gray", rotation=0)
    ax.text(-0.03, 1.5, "Pred:\nBleeding", ha="right", va="center", fontsize=11, color="gray", rotation=0)

    acc = accuracy_score(gt, pr) * 100
    mf1 = f1_score(gt, pr, average="macro", zero_division=0) * 100
    ax.set_title(
        f"Full Test Set - Head {head_id}  (Acc={acc:.1f}%, F1={mf1:.1f}%){title_suffix}",
        fontsize=14, fontweight="bold", pad=18,
    )

    legend_patches = [
        mpatches.Patch(color=c, label=src_labels[s], alpha=0.7)
        for s, c in src_colors.items()
    ]
    ax.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=10,
              bbox_to_anchor=(0.5, -0.08), frameon=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── 右图: NNB-only 四象限 ──
    ax2 = axes[1]
    nnb = valid[valid["source"] == "new_Nonbleeding"]
    if len(nnb) == 0:
        ax2.text(0.5, 0.5, "No NNB data", ha="center", va="center", fontsize=16)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
    else:
        nnb_gt = nnb["true_label"].values
        nnb_pr = nnb["prediction"].values
        nnb_tp = ((nnb_gt == 1) & (nnb_pr == 1)).sum()
        nnb_fp = ((nnb_gt == 0) & (nnb_pr == 1)).sum()
        nnb_tn = ((nnb_gt == 0) & (nnb_pr == 0)).sum()
        nnb_fn = ((nnb_gt == 1) & (nnb_pr == 0)).sum()
        nnb_total = len(nnb)

        nnb_data = [
            ("TP", nnb_tp, 1, 1, "#2ecc71"),
            ("FP", nnb_fp, 0, 1, "#e74c3c"),
            ("TN", nnb_tn, 0, 0, "#3498db"),
            ("FN", nnb_fn, 1, 0, "#f39c12"),
        ]

        ax2.set_xlim(-0.05, 2.05)
        ax2.set_ylim(-0.05, 2.05)
        ax2.set_aspect("equal")
        ax2.set_xticks([])
        ax2.set_yticks([])

        for name, count, x0, y0, color in nnb_data:
            alpha_fill = 0.25 if count > 0 else 0.05
            rect = mpatches.FancyBboxPatch(
                (x0 + 0.02, y0 + 0.02), 0.96, 0.96,
                boxstyle="round,pad=0.03",
                facecolor=color, alpha=alpha_fill, edgecolor=color,
                linewidth=2 if count > 0 else 0.5,
            )
            ax2.add_patch(rect)
            text_alpha = 1.0 if count > 0 else 0.25
            ax2.text(x0 + 0.5, y0 + 0.7, name, ha="center", va="center",
                     fontsize=22, fontweight="bold", color=color, alpha=text_alpha)
            ax2.text(x0 + 0.5, y0 + 0.45, f"{count}", ha="center", va="center",
                     fontsize=30, fontweight="bold", color=color, alpha=text_alpha)
            if nnb_total > 0:
                ax2.text(x0 + 0.5, y0 + 0.22, f"({count/nnb_total*100:.1f}%)", ha="center",
                         va="center", fontsize=12, color=color, alpha=text_alpha * 0.7)

        ax2.plot([1, 1], [0, 2], "k-", linewidth=1, alpha=0.3)
        ax2.plot([0, 2], [1, 1], "k-", linewidth=1, alpha=0.3)
        ax2.text(0.5, 2.02, "GT: 0", ha="center", va="bottom", fontsize=11, color="gray")
        ax2.text(1.5, 2.02, "GT: 1", ha="center", va="bottom", fontsize=11, color="gray")
        ax2.text(-0.03, 0.5, "Pred: 0", ha="right", va="center", fontsize=11, color="gray")
        ax2.text(-0.03, 1.5, "Pred: 1", ha="right", va="center", fontsize=11, color="gray")

        nnb_acc = accuracy_score(nnb_gt, nnb_pr) * 100 if nnb_total > 0 else 0
        ax2.set_title(
            f"NNB Subset - Head {head_id}  (n={nnb_total}, Acc={nnb_acc:.1f}%)",
            fontsize=13, fontweight="bold", pad=18,
        )
        ax2.text(
            1.0, -0.04,
            "NNB: all labels=0, so TP/FN quadrants are always empty",
            ha="center", va="top", fontsize=9, color="gray", style="italic",
        )
    for spine in ax2.spines.values():
        spine.set_visible(False)
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  四象限图已保存: {out_path}")


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", default=DEFAULTS["pred_csv"])
    parser.add_argument("--clips_csv", default=DEFAULTS["clips_csv"])
    parser.add_argument("--meta_csv", default=DEFAULTS["meta_csv"])
    parser.add_argument("--case_cache", default=DEFAULTS["case_cache"])
    parser.add_argument("--best_head", type=int, default=None,
                        help="指定 head 编号; 不指定则按全量测试集 Macro F1 自动选")
    parser.add_argument("--plot_head", type=int, nargs="*", default=None,
                        help="为指定的 head 绘制四象限图; 不指定则只绘制 best head")
    args = parser.parse_args()

    # ── 1. 构建映射 ──
    print("=" * 70)
    print("Step 1: 构建 data_idx → Case_Name 映射")
    print("=" * 70)
    mapping = build_mapping(args.clips_csv, args.meta_csv)
    print(f"  映射表大小: {len(mapping)} 条 (Index 0~{max(mapping.keys())})")

    # ── 2. 读取预测, 提取 NNB ──
    print("\n" + "=" * 70)
    print("Step 2: 读取预测文件 & 提取 NNB 子集")
    print("=" * 70)
    pred_df = pd.read_csv(args.pred_csv)
    nnb_df = extract_nnb(pred_df, mapping)

    n_heads = pred_df["head"].nunique()
    n_nnb_per_head = len(nnb_df) // n_heads
    print(f"  预测总行数: {len(pred_df)} ({len(pred_df)//n_heads} clips × {n_heads} heads)")
    print(f"  NNB 子集:   {len(nnb_df)} ({n_nnb_per_head} clips × {n_heads} heads)")
    print(f"  NNB 案例:")
    for cname in sorted(nnb_df["case_name"].unique()):
        n = len(nnb_df[(nnb_df["case_name"] == cname) & (nnb_df["head"] == nnb_df["head"].min())])
        cid = nnb_df[nnb_df["case_name"] == cname]["case_id_mapped"].iloc[0]
        print(f"    {cname:<12} (Case_ID={cid:>3}): {n} clips")

    # ── 3. 选择 best head ──
    if args.best_head is not None:
        best_head = args.best_head
        print(f"\n  使用指定 head: {best_head}")
    else:
        # 在可映射的全量数据上选 best head
        valid = pred_df[pred_df["data_idx"].isin(mapping)].copy()
        true_labels = valid["data_idx"].map(lambda x: mapping[x]["true_label"])
        valid.loc[:, "label"] = true_labels.values
        best_f1, best_head = -1, 0
        for hid in sorted(valid["head"].unique()):
            hdf = valid[valid["head"] == hid]
            mf1 = f1_score(hdf["label"], hdf["prediction"], average="macro", zero_division=0)
            if mf1 > best_f1:
                best_f1, best_head = mf1, hid
        print(f"\n  全量测试集 Best Head: head_{best_head} (Macro F1={best_f1*100:.2f}%)")

    # ── 4. 评估 NNB (所有 head + best head 详情) ──
    print("\n" + "=" * 70)
    print("Step 3: NNB 子集评估 (evaluate_global_action 方式)")
    print("=" * 70)

    print(f"\n{'Head':>6} | {'Acc':>8} | {'Prec':>8} | {'Rec':>8} | {'F1':>8} | {'IoU':>8} | {'FPR':>7}")
    print("-" * 72)
    for hid in sorted(nnb_df["head"].unique()):
        hdf = nnb_df[nnb_df["head"] == hid]
        s = evaluate_global_action(hdf)
        n_fp = (hdf["prediction"] == 1).sum()
        fpr = n_fp / len(hdf) * 100
        marker = " ← Best" if hid == best_head else ""
        print(
            f"  {hid:>4} | {s['Accuracy_Mean']:>7.2f}% | {s['Macro_Precision_Mean']:>7.2f}% | "
            f"{s['Macro_Recall_Mean']:>7.2f}% | {s['Macro_F1_Mean']:>7.2f}% | "
            f"{s['Macro_IoU_Mean']:>7.2f}% | {fpr:>6.2f}%{marker}"
        )

    # Best head 详细输出
    best_nnb = nnb_df[nnb_df["head"] == best_head]
    best_stats = evaluate_global_action(best_nnb)
    n_fp = (best_nnb["prediction"] == 1).sum()

    print(f"\n{'='*70}")
    print(f"Best Head {best_head} 在 NNB 子集上的结果:")
    print(f"{'='*70}")
    print(f"  Accuracy:        {best_stats['Accuracy_Mean']:.2f}%")
    print(f"  Macro Precision: {best_stats['Macro_Precision_Mean']:.2f}%")
    print(f"  Macro Recall:    {best_stats['Macro_Recall_Mean']:.2f}%")
    print(f"  Macro F1:        {best_stats['Macro_F1_Mean']:.2f}%")
    print(f"  Macro IoU:       {best_stats['Macro_IoU_Mean']:.2f}%")
    print(f"  False Positive:  {n_fp}/{len(best_nnb)} ({n_fp/len(best_nnb)*100:.2f}%)")
    if "per_class" in best_stats:
        for cls_name, m in best_stats["per_class"].items():
            print(f"    {cls_name}: Prec={m['Precision']:.2f}%, Rec={m['Recall']:.2f}%, "
                  f"F1={m['F1']:.2f}%, IoU={m['IoU']:.2f}%")

    # Per-case 细分
    print(f"\n  Per-Case 细分 (head {best_head}):")
    for cname in sorted(best_nnb["case_name"].unique()):
        csub = best_nnb[best_nnb["case_name"] == cname]
        cacc = accuracy_score(csub["label"], csub["prediction"]) * 100
        cfp = (csub["prediction"] == 1).sum()
        print(f"    {cname:<12}: Acc={cacc:.2f}%, FP={cfp}/{len(csub)}")

    out_dir = os.path.dirname(args.pred_csv)

    # ── 5. 绘制四象限图 ──
    plot_heads = args.plot_head if args.plot_head is not None else [best_head]
    for hid in plot_heads:
        fig_path = os.path.join(out_dir, f"confusion_quadrant_head_{hid}.png")
        plot_confusion_quadrant(pred_df, mapping, hid, fig_path)

    # ── 6. 保存 NNB 预测 ──
    nnb_path = os.path.join(out_dir, "predictions_epoch_0_NNB.csv")
    nnb_df.to_csv(nnb_path, index=False)
    print(f"\n  NNB 预测已保存: {nnb_path}")

    # ── 7. 帧定位示例 ──
    print(f"\n{'='*70}")
    print("Step 5: 帧定位示例 (NNB 中第一个 FP 样本)")
    print(f"{'='*70}")

    meta_df = pd.read_csv(args.meta_csv)
    with open(args.case_cache) as f:
        cache = json.load(f)

    fp_samples = best_nnb[best_nnb["prediction"] == 1]
    if len(fp_samples) > 0:
        sample_idx = fp_samples["data_idx"].iloc[0]
        loc = locate_frames(sample_idx, mapping, cache, meta_df)
        if loc:
            print(f"  data_idx = {sample_idx}")
            print(f"  Case:       {loc['case_name']} (Case_ID={loc['case_id']})")
            print(f"  Clip:       #{loc['clip_idx']}, 帧范围 [{loc['start_frame']}, {loc['end_frame']})")
            print(f"  标签帧:     {loc['label_frame']}")
            print(f"  clip txt:   {mapping[sample_idx]['clip_path']}")
            if loc["frame_paths"]:
                unique = list(dict.fromkeys(loc["frame_paths"]))
                print(f"  实际帧数:   {len(unique)} (txt 共 {len(loc['frame_paths'])} 行, 含 padding)")
                print(f"  首帧:       {unique[0]}")
                print(f"  末帧:       {unique[-1]}")
            if loc["raw_mp4s"]:
                print(f"  原始 mp4:   共 {len(loc['raw_mp4s'])} 段, 如:")
                for p in loc["raw_mp4s"][:3]:
                    print(f"              {p}")
    else:
        print("  Best head 在 NNB 上无 FP 样本 (全部正确预测为 non_bleeding)")
        sample_idx = best_nnb["data_idx"].iloc[0]
        loc = locate_frames(sample_idx, mapping, cache, meta_df)
        if loc:
            print(f"\n  示例正确样本 data_idx={sample_idx}:")
            print(f"  Case:       {loc['case_name']} (Case_ID={loc['case_id']})")
            print(f"  标签帧:     {loc['label_frame']}")
            if loc["raw_mp4s"]:
                print(f"  原始 mp4:   共 {len(loc['raw_mp4s'])} 段")


if __name__ == "__main__":
    main()
