import os
import numpy as np
from sklearn.metrics import *


def read_file(gt_pt_p):
    with open(gt_pt_p, 'r+') as f:
        gt_arr = [i.split(' ') for i in f.readlines()]
    gt_arr = [i[:3] for i in gt_arr]
    return np.array(gt_arr, dtype=int)

def get_scores(result_per_vid):
    # mean video-wise metrics
    # acc_scores = []
    # for result in result_per_vid:
    #     acc = accuracy_score(result[:, 2],result[:, 1])
    #     acc_scores.append(acc)
    acc_scores = np.array([accuracy_score(result[:, 2],result[:, 1]) for result in result_per_vid])
    print(acc_scores)
    acc_vid = np.nanmean(acc_scores)
    acc_vid_std = np.nanstd(acc_scores)
    
    ba_vid = np.nanmean([balanced_accuracy_score(result[:, 2],result[:, 1]) for result in result_per_vid])

    # frame-wise metrics
    
    all_results = np.concatenate(result_per_vid)

    all_predictions = all_results[:, 1]
    all_targets = all_results[:, 2]

    acc_frames = accuracy_score(all_targets,all_predictions)
    p = precision_score(all_targets,all_predictions,average='macro')
    p_std = np.nanstd([
        [precision_score(all_targets == i, all_predictions == i) for i in range(np.max(all_targets)+1)]
    ])
    
    r = recall_score(all_targets,all_predictions,average='macro')
    r_std = np.nanstd([
        [recall_score(all_targets == i, all_predictions == i) for i in range(np.max(all_targets)+1)]
    ])
    
    j = jaccard_score(all_targets,all_predictions,average='macro')
    j_std = np.nanstd([
        [jaccard_score(all_targets == i, all_predictions == i) for i in range(np.max(all_targets)+1)]
    ])
    
    f1 = f1_score(all_targets,all_predictions,average='macro')
    f1_std = np.nanstd([
        [f1_score(all_targets == i, all_predictions == i) for i in range(np.max(all_targets)+1)]
    ])
    
    return acc_frames, p, r, j, f1, ba_vid, acc_vid, (acc_vid_std, p_std, r_std, j_std, f1_std)

def evaluate_allmetric(resultp, savep, video_len):
    result_total = []
    for f in sorted(os.listdir(resultp)):
        if f.startswith('video_all'):
            result_all = read_file(os.path.join(resultp, f)) 
            start_idx = 0  

            for length in video_len:
                end_idx = start_idx + length
                result_total.append(result_all[start_idx:end_idx])
                start_idx = end_idx  
            
        elif f.startswith('video'):
            result_total.append(read_file(os.path.join(resultp, f)))
    
    acc_frame, p, r, j, f1, ba_video, acc_video, std_ = get_scores(result_total)
    log_message = (
        f'acc frame {acc_frame*100:1.2f}\n'
        f'prec      {p*100:1.2f} ± {std_[1]*100:1.2f}, rec       {r*100:1.2f} ± {std_[2]*100:1.2f}, jacc      {j*100:1.2f} ± {std_[3]*100:1.2f}, f1        {f1*100:1.2f} ± {std_[4]*100:1.2f}\n'
        f'acc video {acc_video*100:1.2f} ± {std_[0]*100:1.2f}, ba  video {ba_video*100:1.2f}'
    )

    print(log_message)
    with open(savep, "w+") as f:
        f.write(log_message)

if __name__ == "__main__":
    resultp = "/home/tqy/endomamba/videomamba/downstream/SurgicalPhase/Surgformer/result/EndoMamba_numframe32_trainlen32"
    savep = resultp + "/metric.txt"
    video_len = [3652, 3611, 4677, 3545, 3412, 4831, 4325]
    evaluate_allmetric(resultp, savep, video_len)
