import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score, accuracy_score

def evaluate_per_video(csv_file, phases=None):
    df = pd.read_csv(csv_file)
    if phases is None:
        all_labels = np.concatenate([df['label'].values, df['prediction'].values])
        classes = np.unique(all_labels)
        phases = [str(c) for c in classes]
    else:
        classes = list(range(len(phases)))

    per_video = []
    for vid, subdf in df.groupby('index'):
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

    metrics = ["Accuracy", "Macro_Precision", "Macro_Recall", "Macro_IoU", "Macro_F1"]
    stats = {}
    for m in metrics:
        vals = [v[m] for v in per_video]
        stats[f"{m}_Mean"] = np.mean(vals)
        stats[f"{m}_Std"] = np.std(vals)

    stats["Num_Samples_Mean"] = np.mean([v["Num_Samples"] for v in per_video])
    stats["Num_Samples_Std"] = np.std([v["Num_Samples"] for v in per_video])

    return per_video, stats, phases

def evaluate_per_phase(csv_file, phases=None):
    df = pd.read_csv(csv_file)
    if phases is None:
        all_labels = np.concatenate([df['label'].values, df['prediction'].values])
        classes = np.unique(all_labels)
        phases = [str(c) for c in classes]
    else:
        classes = list(range(len(phases)))

    gt = df['label'].values
    pred = df['prediction'].values

    prec = precision_score(gt, pred, average=None, zero_division=0) * 100
    rec = recall_score(gt, pred, average=None, zero_division=0) * 100
    iou = jaccard_score(gt, pred, average=None, zero_division=0) * 100
    f1 = f1_score(gt, pred, average=None, zero_division=0) * 100

    n_samples_per_phase = [(gt == c).sum() for c in classes]

    macro_prec = np.mean(prec)
    macro_rec = np.mean(rec)
    macro_iou = np.mean(iou)
    macro_f1 = np.mean(f1)

    per_phase = []
    for i, phase in enumerate(phases):
        per_phase.append({
            "Phase": phase,
            "Num_Samples": n_samples_per_phase[i],
            "Precision": prec[i],
            "Recall": rec[i],
            "IoU": iou[i],
            "F1": f1[i]
        })

    macro_mean = {
        "Macro_Precision": macro_prec,
        "Macro_Recall": macro_rec,
        "Macro_IoU": macro_iou,
        "Macro_F1": macro_f1,
        "Num_Samples_Mean": np.mean(n_samples_per_phase),
        "Num_Samples_Std": np.std(n_samples_per_phase)
    }

    return per_phase, macro_mean, phases

def save_video_metrics_csv(per_video, stats, csv_file):
    out_dir = os.path.dirname(os.path.abspath(csv_file))
    base = os.path.splitext(os.path.basename(csv_file))[0]
    out_path = os.path.join(out_dir, f"{base}_video_metrics.csv")
    df = pd.DataFrame(per_video)
    df.to_csv(out_path, index=False)
    print(f"[保存成功] 每视频统计结果已保存到: {out_path}")

    stats_path = os.path.join(out_dir, f"{base}_video_metrics_stats.csv")
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(stats_path, index=False)
    print(f"[保存成功] 每视频均值和方差已保存到: {stats_path}")

def save_phase_metrics_csv(per_phase, macro_mean, csv_file):
    out_dir = os.path.dirname(os.path.abspath(csv_file))
    base = os.path.splitext(os.path.basename(csv_file))[0]
    out_path = os.path.join(out_dir, f"{base}_phase_metrics.csv")
    df = pd.DataFrame(per_phase)
    df.to_csv(out_path, index=False)
    print(f"[保存成功] 每phase统计结果已保存到: {out_path}")

    macro_path = os.path.join(out_dir, f"{base}_phase_metrics_macro.csv")
    macro_df = pd.DataFrame([macro_mean])
    macro_df.to_csv(macro_path, index=False)
    print(f"[保存成功] 所有phase的macro均值已保存到: {macro_path}")

def print_video_metrics(per_video, stats):
    print("\n==== 按视频统计 ====")
    print(f"{'Video':10s} {'#Samples':>8s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'IoU':>7s} {'F1':>7s}")
    for v in per_video:
        print(f"{str(v['Video']):10s} {v['Num_Samples']:8d} {v['Accuracy']:7.2f} {v['Macro_Precision']:7.2f} {v['Macro_Recall']:7.2f} {v['Macro_IoU']:7.2f} {v['Macro_F1']:7.2f}")
    print("\n---- 所有视频均值±方差 ----")
    for k in ["Accuracy", "Macro_Precision", "Macro_Recall", "Macro_IoU", "Macro_F1"]:
        mean, std = stats[f"{k}_Mean"], stats[f"{k}_Std"]
        print(f"{k:16s}: {mean:.2f} ± {std:.2f}")
    print(f"Num_Samples     : {stats['Num_Samples_Mean']:.2f} ± {stats['Num_Samples_Std']:.2f}")

def print_phase_metrics(per_phase, macro_mean):
    print("\n==== 按phase统计（所有视频合并） ====")
    print(f"{'Phase':20s} {'#Samples':>8s} {'Prec':>7s} {'Rec':>7s} {'IoU':>7s} {'F1':>7s}")
    for p in per_phase:
        print(f"{p['Phase']:20s} {p['Num_Samples']:8d} {p['Precision']:7.2f} {p['Recall']:7.2f} {p['IoU']:7.2f} {p['F1']:7.2f}")
    print("\n---- 所有phase的macro均值 ----")
    for k in ["Macro_Precision", "Macro_Recall", "Macro_IoU", "Macro_F1"]:
        print(f"{k:16s}: {macro_mean[k]:.2f}")
    print(f"Num_Samples     : {macro_mean['Num_Samples_Mean']:.2f} ± {macro_mean['Num_Samples_Std']:.2f}")

# ======= 用法示例 =======
if __name__ == "__main__":
    # csv_path = "your_prediction.csv"  # 修改为你的文件名
    # csv_path = "/scratch/esg8sdce/wjl/NSJepa/logs/probing_cholec80/vitl16_16x2x3_64f_orgin_probing_1epochs/video_classification_frozen/ssv2-vitl16-16x2x3-64f/all_predictions_epoch_1.csv"  # 修改为你的文件名
    # csv_path = "/scratch/esg8sdce/wjl/NSJepa/logs/probing_cholec80/vith_origin_attentive_64f_probing_1epochs/video_classification_frozen/ssv2-vih16-16x2x3-16f/all_predictions_epoch_1.csv"
    # csv_path="/scratch/esg8sdce/wjl/NSJepa/logs/cpt_cholec80/cpt_vitl16-256px-64f_lr1e-4_epoch-20/video_classification_frozen/ssv2-vitl16-16x2x3-64f/all_predictions_epoch_1.csv"
    # csv_path="/scratch/esg8sdce/wjl/NSJepa/logs/cpt_cholec80/cpt_vith16-256px-64f_lr1e-4_epoch-20/video_classification_frozen/ssv2-vih16-16x2x3-16f/all_predictions_epoch_1.csv"
    # csv_path="logs/cpt_cholec80/cpt_vitl16-256px-64f_lr1e-4_epoch-20/epoch4/video_classification_frozen/ssv2-vitl16-16x2x3-64f/all_predictions_epoch_1.csv"
    csv_path = "logs/cpt_cholec80/cpt_vitl16-256px-64f_lr1e-4_epoch-20/epoch10/video_classification_frozen/ssv2-vitl16-16x2x3-64f/all_predictions_epoch_6.csv"
    # csv_path = "logs/cpt_cholec80/cpt_vith16-256px-64f_lr1e-4_epoch-20/video_classification_frozen/ssv2-vih16-16x2x3-16f/all_predictions_epoch_5.csv"
    
    per_video, stats, phases = evaluate_per_video(csv_path)
    print_video_metrics(per_video, stats)
    save_video_metrics_csv(per_video, stats, csv_path)

    per_phase, macro_mean, phases = evaluate_per_phase(csv_path)
    print_phase_metrics(per_phase, macro_mean)
    save_phase_metrics_csv(per_phase, macro_mean, csv_path)
    
    



