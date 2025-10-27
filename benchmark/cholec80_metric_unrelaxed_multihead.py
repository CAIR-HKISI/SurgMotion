import argparse
import numpy as np
import pandas as pd

N_PHASES = 7

def bwconncomp_bool(arr_bool):
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

def evaluate_video_unrelaxed(gtLabelID, predLabelID):
    assert gtLabelID.shape == predLabelID.shape
    T = len(gtLabelID)

    jaccard = np.full(N_PHASES, np.nan, dtype=float)
    prec = np.full(N_PHASES, np.nan, dtype=float)
    rec = np.full(N_PHASES, np.nan, dtype=float)
    f1 = np.full(N_PHASES, np.nan, dtype=float)

    correct_mask = (predLabelID == gtLabelID)

    for iPhase in range(1, N_PHASES + 1):
        gt_mask = (gtLabelID == iPhase)
        pred_mask = (predLabelID == iPhase)

        if not np.any(gt_mask):
            continue

        if not np.any(pred_mask):
            union_idx = np.where(gt_mask)[0]
        else:
            union_idx = np.union1d(np.where(gt_mask)[0], np.where(pred_mask)[0])

        tp = int(np.sum(correct_mask[union_idx] & (gtLabelID[union_idx] == iPhase)))

        denom_union = len(union_idx)
        j = (tp / denom_union) * 100 if denom_union > 0 else np.nan
        jaccard[iPhase - 1] = j

        sum_pred = int(np.sum(pred_mask))
        sum_gt = int(np.sum(gt_mask))

        p = (tp * 100 / sum_pred) if sum_pred > 0 else np.nan
        r = (tp * 100 / sum_gt) if sum_gt > 0 else np.nan
        prec[iPhase - 1] = p
        rec[iPhase - 1] = r

        if p is not None and r is not None and not (np.isnan(p) or np.isnan(r)) and (p + r) > 0:
            f1[iPhase - 1] = 2 * (p * r) / (p + r)
        else:
            f1[iPhase - 1] = np.nan

    acc = (np.sum(correct_mask) / T) * 100 if T > 0 else np.nan

    for arr in (jaccard, prec, rec, f1):
        over = arr > 100
        arr[over] = 100.0

    return jaccard, prec, rec, acc, f1

def aggregate_across_videos(metrics_per_vid):
    vids = sorted(metrics_per_vid.keys())
    n = len(vids)
    jmat = np.vstack([metrics_per_vid[v]["jaccard"] for v in vids])  
    pmat = np.vstack([metrics_per_vid[v]["prec"] for v in vids])     
    rmat = np.vstack([metrics_per_vid[v]["rec"] for v in vids])      
    fmat = np.vstack([metrics_per_vid[v]["f1"] for v in vids])       
    accs = np.array([metrics_per_vid[v]["acc"] for v in vids])       

    meanJaccPerPhase = np.nanmean(jmat, axis=0)  
    meanPrecPerPhase = np.nanmean(pmat, axis=0)
    meanRecPerPhase = np.nanmean(rmat, axis=0)
    meanF1PerPhase = np.nanmean(fmat, axis=0)

    meanJaccPerVideo = np.nanmean(jmat, axis=1)  
    meanF1PerVideo = np.nanmean(fmat, axis=1)

    meanJacc = float(np.nanmean(meanJaccPerPhase))
    stdJacc = float(np.nanstd(meanJaccPerPhase))

    meanF1 = float(np.nanmean(meanF1PerPhase))
    stdF1 = float(np.nanstd(meanF1PerVideo))

    meanPrec = float(np.nanmean(meanPrecPerPhase))
    stdPrec = float(np.nanstd(meanPrecPerPhase))

    meanRec = float(np.nanmean(meanRecPerPhase))
    stdRec = float(np.nanstd(meanRecPerPhase))

    meanAcc = float(np.nanmean(accs))
    stdAcc = float(np.nanstd(accs))

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
    needed_cols = ["head", "data_idx", "vid", "prediction", "label"]
    for c in needed_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    df = df.sort_values(["head","vid","data_idx"], ascending=[True,True,True]).reset_index(drop=True)

    metrics_per_head = {}
    for head, g_head in df.groupby("head", sort=False):
        metrics_per_vid = {}
        for vid, g in g_head.groupby("vid", sort=False):
            gt = g["label"].to_numpy(dtype=int)
            pred = g["prediction"].to_numpy(dtype=int)
            gt = gt + 1
            pred = pred + 1
            jaccard, prec, rec, acc, f1 = evaluate_video_unrelaxed(gt, pred)
            metrics_per_vid[vid] = {"jaccard": jaccard,"prec": prec,"rec": rec,"acc": acc,"f1": f1}
        summary = aggregate_across_videos(metrics_per_vid)
        metrics_per_head[head] = (metrics_per_vid, summary)
    return metrics_per_head

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV with columns: head,data_idx,vid,prediction,label")
    parser.add_argument("--fps", type=float, default=1.0)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    results = evaluate_from_csv(df, fps=args.fps)

    phases = ["Preparation","CalotTriangleDissection","ClippingCutting","GallbladderDissection","GallbladderPackaging","CleaningCoagulation","GallbladderRetraction"]

    for head, (metrics_per_vid, summary) in results.items():
        print("\n\n")
        print("################################################")
        print(f"Classifier HEAD={head}")
        print("================================================")
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

