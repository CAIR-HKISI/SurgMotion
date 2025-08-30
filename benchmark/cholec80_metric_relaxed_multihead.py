import argparse
import math
import numpy as np
import pandas as pd
from collections import defaultdict

N_PHASES = 7

def bwconncomp_bool(arr_bool):
    """Find connected components (1D) for True regions. Return list of index arrays (0-based)."""
    idxs = np.where(arr_bool)[0]
    if idxs.size == 0:
        return []
    comps = []
    start = idxs[0]
    prev = idxs[0]
    for i in idxs[1:]:
        if i == prev + 1:
            prev = i
        else:
            comps.append(np.arange(start, prev + 1))
            start = i
            prev = i
    comps.append(np.arange(start, prev + 1))
    return comps

def evaluate_video(gtLabelID, predLabelID, fps):
    """
    Python implementation of the MATLAB Evaluate function with relaxed boundary.
    Inputs:
      - gtLabelID: 1D numpy array of ints in [1..7]
      - predLabelID: 1D numpy array of ints in [1..7]
      - fps: frames per second (int/float)
    Returns:
      jaccard: (7,) array in percentage
      prec:    (7,) array in percentage
      rec:     (7,) array in percentage
      acc:     scalar percentage
      f1:      (7,) array in percentage (same scale as prec/rec)
    """
    assert gtLabelID.shape == predLabelID.shape
    T = len(gtLabelID)
    oriT = int(round(10 * fps))  # 10 seconds relaxed boundary
    diff = (predLabelID - gtLabelID).astype(np.int32)
    updatedDiff = np.copy(diff)

    # Initialize to non-zero sentinel to later overwrite only in GT segments
    updatedDiff[:] = diff

    # Relaxed boundary per GT connected component by phase
    for iPhase in range(1, N_PHASES + 1):
        gt_mask = (gtLabelID == iPhase)
        comps = bwconncomp_bool(gt_mask)
        for comp in comps:
            startIdx = comp[0]
            endIdx = comp[-1]
            # slice inclusive
            curDiff = updatedDiff[startIdx:endIdx + 1].copy()

            t = min(oriT, len(curDiff))
            if t < 1:
                continue

            # Apply relaxed rules
            if iPhase in (4, 5):
                # start window: curDiff == -1 -> 0
                win_start = curDiff[:t]
                win_start[win_start == -1] = 0
                curDiff[:t] = win_start
                # end window: curDiff == 1 or 2 -> 0
                win_end = curDiff[-t:]
                mask = (win_end == 1) | (win_end == 2)
                win_end[mask] = 0
                curDiff[-t:] = win_end

            elif iPhase in (6, 7):
                # start window: curDiff == -1 or -2 -> 0
                win_start = curDiff[:t]
                mask = (win_start == -1) | (win_start == -2)
                win_start[mask] = 0
                curDiff[:t] = win_start
                # end window: curDiff == 1 or 2 -> 0
                win_end = curDiff[-t:]
                mask = (win_end == 1) | (win_end == 2)
                win_end[mask] = 0
                curDiff[-t:] = win_end
            else:
                # general
                win_start = curDiff[:t]
                win_start[win_start == -1] = 0
                curDiff[:t] = win_start

                win_end = curDiff[-t:]
                win_end[win_end == 1] = 0
                curDiff[-t:] = win_end

            # write back
            updatedDiff[startIdx:endIdx + 1] = curDiff

    # Metrics per phase
    jaccard = np.full(N_PHASES, np.nan, dtype=float)
    prec = np.full(N_PHASES, np.nan, dtype=float)
    rec = np.full(N_PHASES, np.nan, dtype=float)
    f1 = np.full(N_PHASES, np.nan, dtype=float)

    for iPhase in range(1, N_PHASES + 1):
        gt_mask = (gtLabelID == iPhase)
        pred_mask = (predLabelID == iPhase)

        gt_comps = bwconncomp_bool(gt_mask)
        if len(gt_comps) == 0:
            # No this phase in GT
            continue

        # Union indices (G ∪ P)
        idx_gt = np.where(gt_mask)[0]
        idx_pred = np.where(pred_mask)[0]
        if idx_pred.size == 0:
            union_idx = idx_gt
        elif idx_gt.size == 0:
            union_idx = idx_pred
        else:
            union_idx = np.union1d(idx_gt, idx_pred)

        # True positives under relaxed boundary: updatedDiff == 0 within union
        tp = int(np.sum(updatedDiff[union_idx] == 0))
        denom_union = len(union_idx)
        j = (tp / denom_union) * 100 if denom_union > 0 else np.nan
        jaccard[iPhase - 1] = j

        sum_pred = int(np.sum(pred_mask))
        sum_gt = int(np.sum(gt_mask))
        p = (tp * 100 / sum_pred) if sum_pred > 0 else np.nan
        r = (tp * 100 / sum_gt) if sum_gt > 0 else np.nan
        prec[iPhase - 1] = p
        rec[iPhase - 1] = r

        # F1 with percentages directly, same as MATLAB
        if p is not None and r is not None and not (np.isnan(p) or np.isnan(r)) and (p + r) > 0:
            f1[iPhase - 1] = 2 * (p * r) / (p + r)
        else:
            f1[iPhase - 1] = np.nan

    # Accuracy over whole video (relaxed)
    acc = (np.sum(updatedDiff == 0) / T) * 100 if T > 0 else np.nan

    # Cap >100 to 100, as in MATLAB
    for arr in (jaccard, prec, rec, f1):
        over = arr > 100
        arr[over] = 100.0

    return jaccard, prec, rec, acc, f1


def aggregate_across_videos(metrics_per_vid):
    """
    metrics_per_vid: dict vid -> dict with keys: jaccard, prec, rec, acc, f1
                     where jaccard/prec/rec/f1 are arrays of shape (7,), acc is scalar
    Returns summary dict with per-phase means/stds and global means/stds.
    """
    vids = sorted(metrics_per_vid.keys())
    n = len(vids)
    jmat = np.vstack([metrics_per_vid[v]["jaccard"] for v in vids])  # (n,7)
    pmat = np.vstack([metrics_per_vid[v]["prec"] for v in vids])     # (n,7)
    rmat = np.vstack([metrics_per_vid[v]["rec"] for v in vids])      # (n,7)
    fmat = np.vstack([metrics_per_vid[v]["f1"] for v in vids])       # (n,7)
    accs = np.array([metrics_per_vid[v]["acc"] for v in vids])       # (n,)

    # per-phase mean over videos (nanmean across axis 0)
    meanJaccPerPhase = np.nanmean(jmat, axis=0)  # (7,)
    meanPrecPerPhase = np.nanmean(pmat, axis=0)
    meanRecPerPhase = np.nanmean(rmat, axis=0)
    meanF1PerPhase = np.nanmean(fmat, axis=0)

    # per-video mean over phases
    meanJaccPerVideo = np.nanmean(jmat, axis=1)  # (n,)
    meanF1PerVideo = np.nanmean(fmat, axis=1)

    meanJacc = float(np.nanmean(meanJaccPerPhase))
    stdJacc = float(np.nanstd(meanJaccPerPhase))

    meanF1 = float(np.nanmean(meanF1PerPhase))
    stdF1 = float(np.nanstd(meanF1PerVideo))  # 注意：跟给定 MATLAB 脚本一致

    meanPrec = float(np.nanmean(meanPrecPerPhase))
    stdPrec = float(np.nanstd(meanPrecPerPhase))

    meanRec = float(np.nanmean(meanRecPerPhase))
    stdRec = float(np.nanstd(meanRecPerPhase))

    meanAcc = float(np.nanmean(accs))
    stdAcc = float(np.nanstd(accs))

    # per-phase std across videos
    stdJaccPerPhase = np.nanstd(jmat, axis=0)
    stdPrecPerPhase = np.nanstd(pmat, axis=0)
    stdRecPerPhase = np.nanstd(rmat, axis=0)
    stdF1PerPhase = np.nanstd(fmat, axis=0)

    return {
        "meanJaccPerPhase": meanJaccPerPhase,
        "stdJaccPerPhase": stdJaccPerPhase,
        "meanPrecPerPhase": meanPrecPerPhase,
        "stdPrecPerPhase": stdPrecPerPhase,
        "meanRecPerPhase": meanRecPerPhase,
        "stdRecPerPhase": stdRecPerPhase,
        "meanF1PerPhase": meanF1PerPhase,
        "stdF1PerPhase": stdF1PerPhase,
        "meanJacc": meanJacc,
        "stdJacc": stdJacc,
        "meanF1": meanF1,
        "stdF1": stdF1,
        "meanAcc": meanAcc,
        "stdAcc": stdAcc,
        "meanPrec": meanPrec,
        "stdPrec": stdPrec,
        "meanRec": meanRec,
        "stdRec": stdRec,
        "meanJaccPerVideo": meanJaccPerVideo,
    }


def evaluate_from_csv(df, fps=1):
    """
    df columns: head, data_idx, vid, prediction, label
    - head: 分类器序号
    - Sort by vid then data_idx ascending.
    - Group by head (分类器) and vid (视频) 评估每个分类器在每个视频上的表现
    """
    # Ensure correct dtypes
    needed_cols = ["head", "data_idx", "vid", "prediction", "label"]
    for c in needed_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    # Sort to ensure frame order within each video
    df = df.sort_values(["head", "vid", "data_idx"], ascending=[True, True, True]).reset_index(drop=True)

    # 存储每个分类器的评估结果
    all_metrics = {}
    
    # 按分类器(head)分组
    for head, head_group in df.groupby("head", sort=True):
        metrics_per_vid = {}
        
        # 对当前分类器，按视频(vid)分组评估
        for vid, vid_group in head_group.groupby("vid", sort=False):
            gt = vid_group["label"].to_numpy(dtype=int)
            pred = vid_group["prediction"].to_numpy(dtype=int)

            # 转换标签从0..6到1..7
            gt = gt + 1
            pred = pred + 1

            jaccard, prec, rec, acc, f1 = evaluate_video(gt, pred, fps)

            metrics_per_vid[vid] = {
                "jaccard": jaccard,
                "prec": prec,
                "rec": rec,
                "acc": acc,
                "f1": f1,
            }
        
        # 汇总当前分类器的所有视频结果
        summary = aggregate_across_videos(metrics_per_vid)
        all_metrics[head] = {
            "per_video": metrics_per_vid,
            "summary": summary
        }

    return all_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="predictions.csv", 
                      help="Path to CSV with columns: head,data_idx,vid,prediction,label")
    parser.add_argument("--fps", type=float, default=1.0, 
                      help="Frames per second used for relaxed boundary (default 1)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    all_metrics = evaluate_from_csv(df, fps=args.fps)

    phases = [
        "Preparation",
        "CalotTriangleDissection",
        "ClippingCutting",
        "GallbladderDissection",
        "GallbladderPackaging",
        "CleaningCoagulation",
        "GallbladderRetraction",
    ]

    # 打印每个分类器的评估结果
    for head in sorted(all_metrics.keys()):
        print(f"\n================================================")
        print(f"Classifier {head} Evaluation Results")
        print(f"================================================")
        
        summary = all_metrics[head]["summary"]
        
        print(f"{'Phase':>25}|{'Jacc':>6}|{'Prec':>6}|{'Rec':>6}|")
        print("================================================")
        for i in range(N_PHASES):
            mj = summary["meanJaccPerPhase"][i]
            mp = summary["meanPrecPerPhase"][i]
            mr = summary["meanRecPerPhase"][i]
            print(f"{phases[i]:>25}|{mj:6.2f}|{mp:6.2f}|{mr:6.2f}|")
            print("---------------------------------------------")
        print("================================================")
        print(f"Mean jaccard:   {summary['meanJacc']:5.2f}+/-{summary['stdJacc']:5.2f}")
        print(f"Mean f1-score:  {summary['meanF1']:5.2f}+/-{summary['stdF1']:5.2f}")
        print(f"Mean accuracy:  {summary['meanAcc']:5.2f}+/-{summary['stdAcc']:5.2f}")
        print(f"Mean precision: {summary['meanPrec']:5.2f}+/-{summary['stdPrec']:5.2f}")
        print(f"Mean recall:    {summary['meanRec']:5.2f}+/-{summary['stdRec']:5.2f}")

if __name__ == "__main__":
    main()
