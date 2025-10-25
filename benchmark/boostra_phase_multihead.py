import argparse
import numpy as np
import pandas as pd
import itertools
from collections import defaultdict
from colorama import Fore
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def bootstrap_metrics(gt, pred, vids, n_phase=7, n_resamples=1000, seed=42):
    """
    gt, pred: numpy arrays of shape (N,)
    vids: numpy array of video ids, shape (N,)
    return: dict with bootstrap distributions for metrics
    """
    rng = np.random.default_rng(seed)
    preds_resample = defaultdict(list)
    gts_resample = defaultdict(list)

    for resample_idx in range(n_resamples):
        preds_boot, gts_boot = [], []
        for v in np.unique(vids):
            mask = (vids == v)
            idxs_v = rng.choice(np.where(mask)[0], size=mask.sum(), replace=True)
            preds_boot.append(pred[idxs_v])
            gts_boot.append(gt[idxs_v])
        preds_resample[resample_idx] = preds_boot
        gts_resample[resample_idx] = gts_boot

    image_level_accs = []
    video_level_accs = []
    jaccards = []
    precisions = []
    recalls = []

    for key in preds_resample.keys():
        logits_list = preds_resample[key]
        labels_list = gts_resample[key]

        # image-level accuracy
        preds_all = list(itertools.chain(*logits_list))
        gts_all = list(itertools.chain(*labels_list))
        img_acc = np.mean(np.array(preds_all) == np.array(gts_all))
        image_level_accs.append(img_acc)

        # video-level accuracy (每个视频算 acc，再取均值)
        vid_accs = []
        for p, g in zip(logits_list, labels_list):
            vid_accs.append(np.mean(p == g))
        video_level_accs.append(np.mean(vid_accs))

        # phase-level metrics
        res_all, prec_all, rec_all = [], [], []
        for p, g in zip(logits_list, labels_list):
            res_, prec_, rec_ = [], [], []
            for iPhase in range(n_phase):
                tp = np.sum((p == iPhase) & (g == iPhase))
                if tp == 0:
                    res_.append(np.nan)
                    prec_.append(np.nan)
                    rec_.append(np.nan)
                    continue
                iPUnion = np.sum((p == iPhase) | (g == iPhase))
                jaccard = tp / iPUnion * 100
                res_.append(jaccard)
                sumPred = np.sum(p == iPhase)
                sumGT = np.sum(g == iPhase)
                prec_.append((tp * 100) / sumPred if sumPred > 0 else np.nan)
                rec_.append((tp * 100) / sumGT if sumGT > 0 else np.nan)
            res_all.append(res_); prec_all.append(prec_); rec_all.append(rec_)
        jaccards.append(np.nanmean(res_all))
        precisions.append(np.nanmean(prec_all))
        recalls.append(np.nanmean(rec_all))

    return dict(
        image_level_accs=image_level_accs,
        video_level_accs=video_level_accs,
        jaccards=jaccards,
        precisions=precisions,
        recalls=recalls,
    )

def summarize_with_ci(values):
    mean = np.mean(values)
    lower = mean - 1.96 * np.std(values)
    upper = mean + 1.96 * np.std(values)
    return mean, lower, upper

def evaluate_from_csv(df, n_phase=7, n_resamples=1000):
    results = {}
    for head, g in df.groupby("head"):
        gt = g["label"].to_numpy(dtype=int)
        pred = g["prediction"].to_numpy(dtype=int)
        vids = g["vid"].to_numpy(dtype=int)

        metrics = bootstrap_metrics(gt, pred, vids, n_phase=n_phase, n_resamples=n_resamples)
        metrics["original_acc"] = np.mean(pred == gt)  # 原始 image-level acc
        results[head] = metrics
    return results

def print_results(results):
    color1 = Fore.RED
    color2 = Fore.GREEN
    color3 = Fore.BLUE
    for head, metrics in results.items():
        print(Fore.YELLOW + "="*50 + Fore.RESET)
        print(f"Head {head} evaluation with 95% CI")

        img_mean, img_low, img_up = summarize_with_ci(metrics["image_level_accs"])
        vid_mean, vid_low, vid_up = summarize_with_ci(metrics["video_level_accs"])
        jac_mean, jac_low, jac_up = summarize_with_ci(metrics["jaccards"])
        prec_mean, prec_low, prec_up = summarize_with_ci(metrics["precisions"])
        rec_mean, rec_low, rec_up = summarize_with_ci(metrics["recalls"])

        print(f"Image-level acc 95% CI: {color1}{img_mean*100:.2f}{Fore.RESET}, "
              f"{color2}{img_low*100:.2f}{Fore.RESET}, {color3}{img_up*100:.2f}{Fore.RESET}")
        print(f"Video-level acc 95% CI: {color1}{vid_mean*100:.2f}{Fore.RESET}, "
              f"{color2}{vid_low*100:.2f}{Fore.RESET}, {color3}{vid_up*100:.2f}{Fore.RESET}")
        print(f"Jaccard 95% CI:         {color1}{jac_mean:.2f}{Fore.RESET}, "
              f"{color2}{jac_low:.2f}{Fore.RESET}, {color3}{jac_up:.2f}{Fore.RESET}")
        print(f"Precision 95% CI:       {color1}{prec_mean:.2f}{Fore.RESET}, "
              f"{color2}{prec_low:.2f}{Fore.RESET}, {color3}{prec_up:.2f}{Fore.RESET}")
        print(f"Recall 95% CI:          {color1}{rec_mean:.2f}{Fore.RESET}, "
              f"{color2}{rec_low:.2f}{Fore.RESET}, {color3}{rec_up:.2f}{Fore.RESET}")

        print(f"Original image-level accuracy (no bootstrap): {metrics['original_acc']*100:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--n_resamples", type=int, default=1000)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    results = evaluate_from_csv(df, n_phase=7, n_resamples=args.n_resamples)
    print_results(results)

