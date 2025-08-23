import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def evaluate_from_csv(csv_file_path, phases=None):
    """
    从CSV文件读取数据并使用MATLAB风格评估
    
    Args:
        csv_file_path: CSV文件路径
        phases: 阶段名称列表，如果为None则使用默认的手术阶段
    
    Returns:
        dict: 包含所有评估指标的字典
    """
    
    if phases is None:
        phases = ['Preparation', 'CalotTriangleDissection', 'ClippingCutting', 
                  'GallbladderDissection', 'GallbladderPackaging', 'CleaningCoagulation', 
                  'GallbladderRetraction']
    
    # 读取CSV文件
    print(f"Reading data from {csv_file_path}...")
    df = pd.read_csv(csv_file_path)
    
    # 检查必要的列是否存在
    required_cols = ['classifier_id', 'index', 'vid', 'prediction', 'label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"Data shape: {df.shape}")
    print(f"Unique videos: {sorted(df['vid'].unique())}")
    print(f"Unique classifiers: {sorted(df['classifier_id'].unique())}")
    
    # 获取唯一的分类器和视频
    unique_classifiers = df['classifier_id'].unique()
    unique_videos = df['vid'].unique()
    
    results_by_classifier = {}
    
    for classifier_id in unique_classifiers:
        print(f"\nEvaluating classifier: {classifier_id}")
        classifier_df = df[df['classifier_id'] == classifier_id]
        
        # 存储每个视频的结果
        video_results = []
        
        for vid in unique_videos:
            video_df = classifier_df[classifier_df['vid'] == vid].copy()
            
            if len(video_df) == 0:
                print(f"  Warning: No data for video {vid}")
                continue
            
            # 按index排序确保顺序正确
            video_df = video_df.sort_values('index')
            
            # 获取真实标签和预测标签
            gt_labels = video_df['label'].values
            pred_labels = video_df['prediction'].values
            
            # 转换为数值（如果是字符串）
            try:
                gt_numeric = np.array([int(label) for label in gt_labels])
                pred_numeric = np.array([int(label) for label in pred_labels])
            except ValueError as e:
                print(f"  Error converting labels to numeric for video {vid}: {e}")
                continue
            
            # 检查标签范围
            max_label = len(phases) - 1
            if np.max(gt_numeric) > max_label or np.max(pred_numeric) > max_label:
                print(f"  Warning: Labels exceed expected range (0-{max_label}) for video {vid}")
            
            # 为每个视频计算指标
            video_result = evaluate_single_video(gt_numeric, pred_numeric, len(phases))
            video_result['vid'] = vid
            video_result['num_frames'] = len(gt_numeric)
            
            video_results.append(video_result)
            print(f"  Processed video {vid}: {len(gt_numeric)} frames")
        
        if not video_results:
            print(f"  No valid videos processed for classifier {classifier_id}")
            continue
        
        # 聚合所有视频的结果（MATLAB风格）
        classifier_result = aggregate_video_results(video_results, phases)
        classifier_result['classifier_id'] = classifier_id
        classifier_result['num_videos'] = len(video_results)
        
        results_by_classifier[classifier_id] = classifier_result
        
        # 打印该分类器的结果
        print_results(classifier_result, phases, classifier_id)
    
    return results_by_classifier

def evaluate_single_video(gt_numeric, pred_numeric, num_phases):
    """
    评估单个视频的指标
    
    Args:
        gt_numeric: 真实标签数组
        pred_numeric: 预测标签数组
        num_phases: 阶段数量
    
    Returns:
        dict: 包含该视频所有指标的字典
    """
    
    # 整体准确率
    accuracy = accuracy_score(gt_numeric, pred_numeric) * 100
    
    # 为每个阶段计算二元分类指标
    jaccard_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for phase in range(num_phases):
        # 创建二元标签 (当前阶段 vs 其他)
        gt_binary = (gt_numeric == phase).astype(int)
        pred_binary = (pred_numeric == phase).astype(int)
        
        # 只有当真实标签中存在该阶段时才计算
        if np.sum(gt_binary) > 0:
            jaccard_scores.append(jaccard_score(gt_binary, pred_binary, zero_division=0) * 100)
            precision_scores.append(precision_score(gt_binary, pred_binary, zero_division=0) * 100)
            recall_scores.append(recall_score(gt_binary, pred_binary, zero_division=0) * 100)
            f1_scores.append(f1_score(gt_binary, pred_binary, zero_division=0) * 100)
        else:
            # 该阶段在真实标签中不存在
            jaccard_scores.append(np.nan)
            precision_scores.append(np.nan)
            recall_scores.append(np.nan)
            f1_scores.append(np.nan)
    
    return {
        'accuracy': accuracy,
        'jaccard': np.array(jaccard_scores),
        'precision': np.array(precision_scores),
        'recall': np.array(recall_scores),
        'f1': np.array(f1_scores)
    }

def aggregate_video_results(video_results, phases):
    """
    聚合多个视频的结果（MATLAB风格）
    
    Args:
        video_results: 视频结果列表
        phases: 阶段名称列表
    
    Returns:
        dict: 聚合后的结果
    """
    
    num_phases = len(phases)
    
    # 收集所有视频的指标
    all_jaccard = np.array([result['jaccard'] for result in video_results]).T  # (num_phases, num_videos)
    all_precision = np.array([result['precision'] for result in video_results]).T
    all_recall = np.array([result['recall'] for result in video_results]).T
    all_f1 = np.array([result['f1'] for result in video_results]).T
    all_accuracy = np.array([result['accuracy'] for result in video_results])
    
    # 限制值范围 (与MATLAB代码一致)
    all_jaccard = np.clip(all_jaccard, 0, 100)
    all_precision = np.clip(all_precision, 0, 100)
    all_recall = np.clip(all_recall, 0, 100)
    all_f1 = np.clip(all_f1, 0, 100)
    
    # 计算每个阶段的平均值（跨视频）
    mean_jacc_per_phase = np.nanmean(all_jaccard, axis=1)
    mean_prec_per_phase = np.nanmean(all_precision, axis=1)
    mean_rec_per_phase = np.nanmean(all_recall, axis=1)
    mean_f1_per_phase = np.nanmean(all_f1, axis=1)
    
    # 计算每个阶段的标准差
    std_jacc_per_phase = np.nanstd(all_jaccard, axis=1)
    std_prec_per_phase = np.nanstd(all_precision, axis=1)
    std_rec_per_phase = np.nanstd(all_recall, axis=1)
    std_f1_per_phase = np.nanstd(all_f1, axis=1)
    
    # 计算整体平均值和标准差（与MATLAB完全一致）
    mean_jacc = np.mean(mean_jacc_per_phase)
    std_jacc = np.std(mean_jacc_per_phase)
    
    mean_prec = np.nanmean(mean_prec_per_phase)
    std_prec = np.nanstd(mean_prec_per_phase)
    
    mean_rec = np.mean(mean_rec_per_phase)
    std_rec = np.std(mean_rec_per_phase)
    
    mean_f1 = np.mean(mean_f1_per_phase)
    std_f1 = np.nanstd(np.nanmean(all_f1, axis=0))  # 与MATLAB一致
    
    mean_acc = np.nanmean(all_accuracy)
    std_acc = np.nanstd(all_accuracy)
    
    return {
        'mean_jaccard': mean_jacc,
        'mean_precision': mean_prec,
        'mean_recall': mean_rec,
        'mean_f1': mean_f1,
        'mean_accuracy': mean_acc,
        'std_jaccard': std_jacc,
        'std_precision': std_prec,
        'std_recall': std_rec,
        'std_f1': std_f1,
        'std_accuracy': std_acc,
        'per_phase_results': {
            'jaccard': mean_jacc_per_phase,
            'precision': mean_prec_per_phase,
            'recall': mean_rec_per_phase,
            'f1': mean_f1_per_phase,
            'jaccard_std': std_jacc_per_phase,
            'precision_std': std_prec_per_phase,
            'recall_std': std_rec_per_phase,
            'f1_std': std_f1_per_phase
        },
        'raw_results': {
            'all_jaccard': all_jaccard,
            'all_precision': all_precision,
            'all_recall': all_recall,
            'all_f1': all_f1,
            'all_accuracy': all_accuracy
        }
    }

def print_results(results, phases, classifier_id=None):
    """
    打印评估结果
    
    Args:
        results: 评估结果字典
        phases: 阶段名称列表
        classifier_id: 分类器ID（可选）
    """
    
    if classifier_id:
        print(f'\n========== Results for Classifier: {classifier_id} ==========')
    else:
        print('\n========== Evaluation Results ==========')
    
    print('================================================')
    print(f"{'Phase':<25}|{'Jacc':>6}|{'Prec':>6}|{'Rec':>6}|{'F1':>6}|")
    print('================================================')
    
    per_phase = results['per_phase_results']
    for i, phase in enumerate(phases):
        print(f"{phase:<25}|{per_phase['jaccard'][i]:>6.2f}|"
              f"{per_phase['precision'][i]:>6.2f}|{per_phase['recall'][i]:>6.2f}|"
              f"{per_phase['f1'][i]:>6.2f}|")
        print('---------------------------------------------')
    
    print('================================================')
    print(f'Mean jaccard:  {results["mean_jaccard"]:5.2f} ± {results["std_jaccard"]:5.2f}')
    print(f'Mean precision:{results["mean_precision"]:5.2f} ± {results["std_precision"]:5.2f}')
    print(f'Mean recall:   {results["mean_recall"]:5.2f} ± {results["std_recall"]:5.2f}')
    print(f'Mean f1-score: {results["mean_f1"]:5.2f} ± {results["std_f1"]:5.2f}')
    print(f'Mean accuracy: {results["mean_accuracy"]:5.2f} ± {results["std_accuracy"]:5.2f}')
    print(f'Number of videos: {results["num_videos"]}')

def save_results_to_csv(results_by_classifier, output_file, phases):
    """
    将结果保存到CSV文件
    
    Args:
        results_by_classifier: 按分类器分组的结果
        output_file: 输出文件路径
        phases: 阶段名称列表
    """
    
    # 准备数据
    rows = []
    
    for classifier_id, results in results_by_classifier.items():
        # 整体指标
        row = {
            'classifier_id': classifier_id,
            'metric_type': 'overall',
            'phase': 'all',
            'mean_jaccard': results['mean_jaccard'],
            'std_jaccard': results['std_jaccard'],
            'mean_precision': results['mean_precision'],
            'std_precision': results['std_precision'],
            'mean_recall': results['mean_recall'],
            'std_recall': results['std_recall'],
            'mean_f1': results['mean_f1'],
            'std_f1': results['std_f1'],
            'mean_accuracy': results['mean_accuracy'],
            'std_accuracy': results['std_accuracy'],
            'num_videos': results['num_videos']
        }
        rows.append(row)
        
        # 每个阶段的指标
        per_phase = results['per_phase_results']
        for i, phase in enumerate(phases):
            row = {
                'classifier_id': classifier_id,
                'metric_type': 'per_phase',
                'phase': phase,
                'mean_jaccard': per_phase['jaccard'][i],
                'std_jaccard': per_phase['jaccard_std'][i],
                'mean_precision': per_phase['precision'][i],
                'std_precision': per_phase['precision_std'][i],
                'mean_recall': per_phase['recall'][i],
                'std_recall': per_phase['recall_std'][i],
                'mean_f1': per_phase['f1'][i],
                'std_f1': per_phase['f1_std'][i],
                'mean_accuracy': np.nan,  # 准确率只有整体的
                'std_accuracy': np.nan,
                'num_videos': results['num_videos']
            }
            rows.append(row)
    
    # 保存到CSV
    df_results = pd.DataFrame(rows)
    df_results.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

# 使用示例
if __name__ == "__main__":
    # 定义阶段（根据你的数据调整）
    phases = ['Preparation', 'CalotTriangleDissection', 'ClippingCutting', 
              'GallbladderDissection', 'GallbladderPackaging', 'CleaningCoagulation', 
              'GallbladderRetraction']
    
    # 评估
    # csv_file = "/data/wjl/vjepa2/logs5/autolaparo_vith16_16x2x3_64f_orgin_probing/video_classification_frozen/autolaparo-vith16-16x2x3-64f/all_predictions_epoch_1.csv"  # 替换为你的CSV文件路径
#     csv_file = "/data/wjl/vjepa2/logs5/autolaparo_vith16_16x2x3_64f_orgin_probing_wd-0.8/video_classification_frozen/autolaparo-vith16-16x2x3-64f/all_predictions_epoch_1.csv"
    # csv_file = "/data/wjl/vjepa2/logs5/autolaparo_vith16_16x2x3_64f_orgin_probing_wd-0.8/video_classification_frozen/autolaparo-vith16-16x2x3-64f/all_predictions_epoch_1.csv"
    # csv_file = "/data/wjl/vjepa2/logs5/autolaparo_vith16_16x2x3_64f_orgin_probing_4epoch/video_classification_frozen/autolaparo-vith16-16x2x3-64f/all_predictions_epoch_4.csv"
    # csv_file = "/data/wjl/vjepa2/logs5/autolaparo_vitl16_16x2x3_64f_pitvis-cpt_probing_4epoch/video_classification_frozen/autolaparo-vitl16-16x2x3-64f/all_predictions_epoch_4.csv"
    csv_file="/data/wjl/vjepa2/logs5/autolaparo_vith16_16x2x3_64f_orgin_probing_10epoch/video_classification_frozen/autolaparo-vith16-16x2x3-64f/all_predictions_epoch_6.csv"
    try:
        results = evaluate_from_csv(csv_file, phases)
        
        # 保存详细结果
        save_results_to_csv(results, "evaluation_results.csv", phases)
        
        # 如果只有一个分类器，也可以这样获取结果
        if len(results) == 1:
            single_result = list(results.values())[0]
            print("\n========== Summary ==========")
            print(f"Best F1 Score: {single_result['mean_f1']:.2f} ± {single_result['std_f1']:.2f}")
            print(f"Best Accuracy: {single_result['mean_accuracy']:.2f} ± {single_result['std_accuracy']:.2f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()