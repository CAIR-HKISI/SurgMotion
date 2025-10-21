import os
import logging
import math
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from sklearn.metrics import precision_score, recall_score, jaccard_score, f1_score, accuracy_score
import wandb

from evals.surgical_video_classification_frozen.models import init_module
from evals.video_classification_frozen.utils import make_transforms
from evals.utils.bootstrap import bootstrap_per_video_metrics, print_bootstrap_results
from src.datasets.data_manager import init_data
from src.models.attentive_pooler import AttentiveClassifier
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import AllReduce, init_distributed
from src.utils.logging import AverageMeter

import torch.distributed as dist

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


# ----------------------
# Edit Distance Calculation
# ----------------------
def compress_segments(sequence):
    """
    Compress consecutive repeated labels into segments.
    Example: [0, 0, 0, 1, 1, 1, 2] -> [0, 1, 2]
             [0, 1, 0, 1, 0] -> [0, 1, 0, 1, 0]
    """
    if len(sequence) == 0:
        return []

    segments = [sequence[0]]
    for i in range(1, len(sequence)):
        if sequence[i] != sequence[i-1]:
            segments.append(sequence[i])

    return segments


def levenshtein_distance(seq1, seq2):
    """Calculate Levenshtein (edit) distance between two sequences."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n]


def segmental_edit_distance(seq1, seq2):
    """
    Calculate edit distance on compressed segments (not frame-level).
    Returns normalized edit score (0-100 scale).
    """
    # Compress sequences to segments
    segments1 = compress_segments(seq1)
    segments2 = compress_segments(seq2)

    # Calculate edit distance on segments
    edit_dist = levenshtein_distance(segments1, segments2)
    max_len = max(len(segments1), len(segments2))

    if max_len == 0:
        return 0.0

    return (edit_dist / max_len) * 100


# ----------------------
# 视频级评估函数
# ----------------------
def evaluate_per_video(predictions_df, phases=None, use_bootstrap=False, n_bootstrap=1000, random_seed=None):
    """
    Evaluate per-video metrics with optional bootstrap uncertainty estimation.

    Args:
        predictions_df: DataFrame with columns [data_idx, vid, prediction, label]
        phases: List of phase names (optional)
        use_bootstrap: If True, perform bootstrap resampling for uncertainty estimation
        n_bootstrap: Number of bootstrap iterations (default: 1000)
        random_seed: Random seed for reproducibility (default: None)

    Returns:
        per_video: List of per-video metrics
        stats: Aggregated statistics (with bootstrap uncertainty if use_bootstrap=True)
        phases: List of phase names
    """
    if phases is None:
        all_labels = np.concatenate([predictions_df['label'].values, predictions_df['prediction'].values])
        classes = np.unique(all_labels)
        phases = [str(c) for c in classes]

    # Sort predictions by video and temporal index to ensure correct ordering
    predictions_df = predictions_df.sort_values(['vid', 'data_idx'])

    # Compute overall (across all videos) per-class metrics
    all_gt = predictions_df['label'].values
    all_pred = predictions_df['prediction'].values

    # Get unique classes present in the data
    unique_classes = np.unique(np.concatenate([all_gt, all_pred]))

    # Per-class metrics (using None for labels to get per-class results)
    per_class_precision = precision_score(all_gt, all_pred, labels=unique_classes, average=None, zero_division=0) * 100
    per_class_recall = recall_score(all_gt, all_pred, labels=unique_classes, average=None, zero_division=0) * 100
    per_class_f1 = f1_score(all_gt, all_pred, labels=unique_classes, average=None, zero_division=0) * 100
    per_class_iou = jaccard_score(all_gt, all_pred, labels=unique_classes, average=None, zero_division=0) * 100

    # Per-video metrics
    per_video = []
    for vid, subdf in predictions_df.groupby('vid'):
        # Ensure temporal ordering within each video
        subdf = subdf.sort_values('data_idx')
        gt = subdf['label'].values
        pred = subdf['prediction'].values

        acc = accuracy_score(gt, pred) * 100
        macro_prec = precision_score(gt, pred, average='macro', zero_division=0) * 100
        macro_rec = recall_score(gt, pred, average='macro', zero_division=0) * 100
        macro_iou = jaccard_score(gt, pred, average='macro', zero_division=0) * 100
        macro_f1 = f1_score(gt, pred, average='macro', zero_division=0) * 100
        n_samples = len(gt)

        # Calculate segmental edit distance (temporal segmentation metric)
        edit_dist = segmental_edit_distance(gt.tolist(), pred.tolist())

        per_video.append({
            "Video": vid,
            "Num_Samples": n_samples,
            "Accuracy": acc,
            "Macro_Precision": macro_prec,
            "Macro_Recall": macro_rec,
            "Macro_IoU": macro_iou,
            "Macro_F1": macro_f1,
            "Edit_Distance": edit_dist
        })

    # Aggregate stats across videos
    metrics = ["Accuracy", "Macro_Precision", "Macro_Recall", "Macro_IoU", "Macro_F1", "Edit_Distance"]
    stats = {}

    if use_bootstrap:
        # Perform bootstrap resampling for uncertainty estimation
        logger.info(f"Performing bootstrap with {n_bootstrap} iterations...")
        bootstrap_results = bootstrap_per_video_metrics(
            per_video_results=per_video,
            metric_keys=metrics,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed
        )

        # Store bootstrap results
        for m in metrics:
            stats[f"{m}_Mean"] = bootstrap_results['mean'][m]
            stats[f"{m}_Std"] = bootstrap_results['std'][m]
            stats[f"{m}_CI_Lower"] = bootstrap_results['ci_lower'][m]
            stats[f"{m}_CI_Upper"] = bootstrap_results['ci_upper'][m]

        # Print bootstrap results
        print_bootstrap_results(bootstrap_results, metric_keys=metrics)
    else:
        # Standard aggregation (simple mean and std)
        for m in metrics:
            vals = [v[m] for v in per_video]
            stats[f"{m}_Mean"] = np.mean(vals)
            stats[f"{m}_Std"] = np.std(vals)

    # Add per-class metrics to stats
    per_class_metrics = {}
    for i, cls in enumerate(unique_classes):
        per_class_metrics[f"Phase_{cls}"] = {
            "Precision": per_class_precision[i],
            "Recall": per_class_recall[i],
            "F1": per_class_f1[i],
            "IoU": per_class_iou[i]
        }
    stats["per_class"] = per_class_metrics

    return per_video, stats, phases


# ----------------------
# 主入口
# ----------------------
def main(args_eval, resume_preempt=False):

    val_only = args_eval.get("val_only", False)
    pretrain_folder = args_eval.get("folder", None)
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    eval_tag = args_eval.get("tag", None)
    num_workers = args_eval.get("num_workers", 12)

    # Quick run / debug mode configuration
    quick_run = args_eval.get("quick_run", False)
    quick_run_num_videos = args_eval.get("quick_run_num_videos", 2)

    # Bootstrap configuration (default: enabled)
    use_bootstrap = args_eval.get("use_bootstrap", True)
    n_bootstrap = args_eval.get("n_bootstrap", 1000)
    bootstrap_seed = args_eval.get("bootstrap_seed", None)

    # wandb configuration
    wandb_config = args_eval.get("wandb", {})
    wandb_project = wandb_config.get("project", "nsjepa-surgical-probing")
    wandb_entity = wandb_config.get("entity", None)
    wandb_name = wandb_config.get("name", None)
    wandb_tags = wandb_config.get("tags", [])
    wandb_group = wandb_config.get("group", None)
    wandb_notes = wandb_config.get("notes", None)

    args_pretrain = args_eval.get("model_kwargs")
    checkpoint = args_pretrain.get("checkpoint")
    module_name = args_pretrain.get("module_name")
    args_model = args_pretrain.get("pretrain_kwargs")
    args_wrapper = args_pretrain.get("wrapper_kwargs")

    args_exp = args_eval.get("experiment")
    args_classifier = args_exp.get("classifier")
    num_probe_blocks = args_classifier.get("num_probe_blocks", 1)
    num_heads = args_classifier.get("num_heads", 16)

    args_data = args_exp.get("data")
    dataset_type = args_data.get("dataset_type", "VideoDataset")

    # Support both single and multiple datasets
    train_data_path = args_data.get("dataset_train")
    if isinstance(train_data_path, str):
        train_data_path = [train_data_path]

    val_data_path = args_data.get("dataset_val")
    if isinstance(val_data_path, str):
        val_data_path = [val_data_path]

    # Support datasets_weights for sampling from multiple datasets
    datasets_weights = args_data.get("datasets_weights", None)

    # Support per-dataset num_classes or single num_classes for all datasets
    num_classes = args_data.get("num_classes")
    if isinstance(num_classes, int):
        num_classes_list = [num_classes] * len(train_data_path)
    else:
        num_classes_list = num_classes

    # Support head-to-dataset mapping (which dataset each head trains on)
    head_to_dataset_map = args_data.get("head_to_dataset_map", None)

    resolution = args_data.get("resolution", 224)
    num_segments = args_data.get("num_segments", 1)
    frames_per_clip = args_data.get("frames_per_clip", 16)
    frame_step = args_data.get("frame_step", 1)
    duration = args_data.get("clip_duration", None)
    num_views_per_segment = args_data.get("num_views_per_segment", 1)
    normalization = args_data.get("normalization", None)

    args_opt = args_exp.get("optimization")
    batch_size = args_opt.get("batch_size")
    num_epochs = args_opt.get("num_epochs")
    use_bfloat16 = args_opt.get("use_bfloat16")
    opt_kwargs = args_opt.get("multihead_kwargs")  # list，每个分类头一个 kwargs

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    world_size, rank = init_distributed()

    # Quick run mode: create subset CSV files (only on rank 0)
    if quick_run and rank == 0:
        logger.info(f"Quick run mode enabled: using {quick_run_num_videos} video(s)")

        # Create subset for training data
        train_data_path_subset = []
        for path in train_data_path:
            if path.endswith('.csv'):
                subset_path = create_quick_run_subset(path, num_videos=quick_run_num_videos)
                train_data_path_subset.append(subset_path)
            else:
                logger.warning(f"Quick run mode only supports CSV files, skipping: {path}")
                train_data_path_subset.append(path)

        # Create subset for validation data
        val_data_path_subset = []
        for path in val_data_path:
            if path.endswith('.csv'):
                subset_path = create_quick_run_subset(path, num_videos=quick_run_num_videos)
                val_data_path_subset.append(subset_path)
            else:
                logger.warning(f"Quick run mode only supports CSV files, skipping: {path}")
                val_data_path_subset.append(path)

        # Use subset paths
        train_data_path = train_data_path_subset
        val_data_path = val_data_path_subset

    # Sync across ranks if using distributed training
    if dist.is_initialized():
        dist.barrier()

    # Initialize wandb (only on rank 0)
    if rank == 0:
        # Prepare config dict for wandb
        wandb_run_config = {
            "eval_name": args_eval.get("eval_name"),
            "tag": eval_tag,
            "num_workers": num_workers,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "use_bfloat16": use_bfloat16,
            "num_probe_blocks": num_probe_blocks,
            "num_heads": num_heads,
            "frames_per_clip": frames_per_clip,
            "resolution": resolution,
            "num_classes": num_classes_list,
            "use_bootstrap": use_bootstrap,
            "n_bootstrap": n_bootstrap if use_bootstrap else None,
            "quick_run": quick_run,
            "quick_run_num_videos": quick_run_num_videos if quick_run else None,
        }

        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name,
            config=wandb_run_config,
            tags=wandb_tags,
            group=wandb_group,
            notes=wandb_notes,
            resume="allow"
        )
        logger.info(f"wandb initialized: {wandb.run.url}")

    # checkpoint 路径
    folder = os.path.join(pretrain_folder, "video_classification_frozen/")
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    os.makedirs(folder, exist_ok=True)
    latest_path = os.path.join(folder, "latest.pt")

    # 构建 encoder
    encoder = init_module(
        module_name=module_name,
        frames_per_clip=frames_per_clip,
        resolution=resolution,
        checkpoint=checkpoint,
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device=device,
    )

    # 构建多个分类头
    # If head_to_dataset_map is provided, use per-dataset num_classes
    if head_to_dataset_map is not None:
        classifiers = [
            AttentiveClassifier(
                embed_dim=encoder.embed_dim,
                num_heads=num_heads,
                depth=num_probe_blocks,
                num_classes=num_classes_list[head_to_dataset_map[idx]],
                use_activation_checkpointing=True,
            ).to(device)
            for idx in range(len(opt_kwargs))
        ]
    else:
        # Default: all heads use the first dataset's num_classes
        classifiers = [
            AttentiveClassifier(
                embed_dim=encoder.embed_dim,
                num_heads=num_heads,
                depth=num_probe_blocks,
                num_classes=num_classes_list[0],
                use_activation_checkpointing=True,
            ).to(device)
            for _ in opt_kwargs
        ]
    classifiers = [DistributedDataParallel(c, static_graph=True) for c in classifiers]

    train_loader, train_sampler = make_dataloader(
        dataset_type=dataset_type,
        root_path=train_data_path,
        img_size=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        eval_duration=duration,
        num_segments=num_segments,
        num_views_per_segment=1,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=True,
        num_workers=num_workers,
        normalization=normalization,
        datasets_weights=datasets_weights,
    )
    val_loader, _ = make_dataloader(
        dataset_type=dataset_type,
        root_path=val_data_path,
        img_size=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_segments=num_segments,
        eval_duration=duration,
        num_views_per_segment=num_views_per_segment,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=False,
        num_workers=num_workers,
        normalization=normalization,
    )
    ipe = len(train_loader)

    # 多头优化器
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        classifiers=classifiers,
        opt_kwargs=opt_kwargs,
        iterations_per_epoch=ipe,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16,
    )

    # 断点恢复
    start_epoch = 0
    if resume_checkpoint and os.path.exists(latest_path):
        encoder, classifiers, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=latest_path,
            encoder=encoder,
            classifiers=classifiers,
            opt=optimizer,
            scaler=scaler,
            val_only=val_only,
        )
        for _ in range(start_epoch * ipe):
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

    def save_checkpoint(epoch):
        save_dict = {
            "encoder": encoder.state_dict(),
            "classifiers": [c.module.state_dict() for c in classifiers],
            "opt": [o.state_dict() for o in optimizer],
            "scaler": [None if s is None else s.state_dict() for s in scaler],
            "epoch": epoch,
        }
        if rank == 0:
            torch.save(save_dict, latest_path)

    # ----------------
    # 训练循环
    # ----------------
    for epoch in range(start_epoch, num_epochs):
        train_sampler.set_epoch(epoch)

        if not val_only:
            train_metrics = run_one_epoch(
                device=device,
                training=True,
                encoder=encoder,
                classifiers=classifiers,
                scaler=scaler,
                optimizer=optimizer,
                scheduler=scheduler,
                wd_scheduler=wd_scheduler,
                data_loader=train_loader,
                use_bfloat16=use_bfloat16,
                num_classes=num_classes,
                epoch=epoch,
                head_to_dataset_map=head_to_dataset_map,
                rank=rank,
            )
        else:
            train_metrics = None

        val_metrics = run_one_epoch(
            device=device,
            training=False,
            encoder=encoder,
            classifiers=classifiers,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16,
            num_classes=num_classes,
            epoch=epoch,
            save_predictions=True,
            folder=folder,
            head_to_dataset_map=head_to_dataset_map,
            use_bootstrap=use_bootstrap,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
            rank=rank,
            train_loader_len=len(train_loader),
        )

        logger.info(f"Epoch {epoch+1}: train={train_metrics} val={val_metrics}")

        if val_only:
            if dist.is_initialized():
                dist.destroy_process_group()
            return

        save_checkpoint(epoch + 1)

    # Finish wandb run (only on rank 0)
    if rank == 0:
        wandb.finish()

    if dist.is_initialized():
        dist.destroy_process_group()


# ----------------------
# 单个 epoch 训练/验证
# ----------------------
def run_one_epoch(
    device, training, encoder, classifiers, scaler, optimizer,
    scheduler, wd_scheduler, data_loader, use_bfloat16, num_classes,
    epoch=0, folder=None, save_predictions=False, fps=1.0, log_interval=20,
    head_to_dataset_map=None, use_bootstrap=False, n_bootstrap=1000, bootstrap_seed=None,
    rank=0, train_loader_len=None
):
    for c in classifiers:
        c.train(mode=training)

    criterion = torch.nn.CrossEntropyLoss()
    acc_meters = [AverageMeter() for _ in classifiers]
    loss_meters = [AverageMeter() for _ in classifiers]

    if not training:
        all_predictions = []

    for itr, data in enumerate(data_loader):
        if training:
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):
            clips = [[dij.to(device) for dij in di] for di in data[0]]
            clip_indices = [d.to(device) for d in data[2]]
            labels = data[1][0].to(device)
            batch_size = len(labels)

            # Extract dataset_idx if available (for multi-dataset training)
            if len(data[1]) > 3:
                dataset_indices = data[1][3].to(device)
            else:
                dataset_indices = None

            with torch.no_grad():
                features = encoder(clips, clip_indices)

            # 每个分类器独立输出
            outputs = [[c(f) for f in features] for c in classifiers]

            # 每个分类器独立 loss with optional masking for multi-dataset training
            losses = []
            has_samples = []  # Track which heads have actual samples in this batch
            if head_to_dataset_map is not None and dataset_indices is not None:
                # Multi-dataset training: mask loss by dataset
                for head_idx, coutputs in enumerate(outputs):
                    assigned_dataset = head_to_dataset_map[head_idx]
                    head_losses = []
                    head_has_samples = False
                    for o in coutputs:
                        mask = (dataset_indices == assigned_dataset)
                        if mask.sum() > 0:
                            loss = criterion(o[mask], labels[mask])
                            head_has_samples = True
                        else:
                            # No samples from this dataset in batch, create dummy loss
                            loss = torch.tensor(0.0, device=device, requires_grad=True)
                        head_losses.append(loss)
                    losses.append(head_losses)
                    has_samples.append(head_has_samples)
            else:
                # Single dataset or no masking: standard loss calculation
                losses = [[criterion(o, labels) for o in coutputs] for coutputs in outputs]
                has_samples = [True] * len(losses)

        if training:
            if use_bfloat16:
                for s, li, o, has_sample in zip(scaler, losses, optimizer, has_samples):
                    if has_sample:
                        for lij in li:
                            s.scale(lij).backward(retain_graph=True)
                        s.step(o)
                        s.update()
                        o.zero_grad()
                    else:
                        # Skip optimizer step if no samples for this head in batch
                        o.zero_grad()
            else:
                for li, o, has_sample in zip(losses, optimizer, has_samples):
                    if has_sample:
                        for lij in li:
                            lij.backward(retain_graph=True)
                        o.step()
                        o.zero_grad()
                    else:
                        # Skip optimizer step if no samples for this head in batch
                        o.zero_grad()

        with torch.no_grad():
            for idx, coutputs in enumerate(outputs):
                avg_output = torch.stack([F.softmax(o, dim=1) for o in coutputs]).mean(0)
                preds = avg_output.argmax(dim=1)

                # Calculate metrics: if masking, only for assigned dataset samples
                if head_to_dataset_map is not None and dataset_indices is not None:
                    assigned_dataset = head_to_dataset_map[idx]
                    mask = (dataset_indices == assigned_dataset)
                    if mask.sum() > 0:
                        masked_preds = preds[mask]
                        masked_labels = labels[mask]
                        acc = 100.0 * masked_preds.eq(masked_labels).sum() / mask.sum()
                        acc = float(AllReduce.apply(acc))
                        acc_meters[idx].update(acc, mask.sum().item())
                    # else: no samples from this dataset in batch, skip update
                else:
                    # No masking: standard accuracy calculation
                    acc = 100.0 * preds.eq(labels).sum() / batch_size
                    acc = float(AllReduce.apply(acc))
                    acc_meters[idx].update(acc)

                loss_val = torch.stack([lij.detach() for lij in losses[idx]]).mean().item()
                loss_meters[idx].update(loss_val, batch_size)

                if not training:
                    vid_ids, data_idxs = data[1][1], data[1][2]
                    # Save predictions with dataset_idx for per-dataset analysis
                    if dataset_indices is not None:
                        for pred, label, vid, did, ds_idx in zip(
                            preds.cpu(), labels.cpu(), vid_ids.cpu(), data_idxs.cpu(), dataset_indices.cpu()
                        ):
                            all_predictions.append([idx, did.item(), vid.item(), pred.item(), label.item(), ds_idx.item()])
                    else:
                        for pred, label, vid, did in zip(
                            preds.cpu(), labels.cpu(), vid_ids.cpu(), data_idxs.cpu()
                        ):
                            all_predictions.append([idx, did.item(), vid.item(), pred.item(), label.item(), 0])

        if itr % log_interval == 0:
            if training:
                logger.info(
                    f"[Train][Epoch {epoch}][Iter {itr}/{len(data_loader)}] "
                    + " ".join([f"Head{h}: Acc={am.avg:.2f}%, Loss={lm.avg:.4f}"
                                for h, (am, lm) in enumerate(zip(acc_meters, loss_meters))])
                )

                # Log training metrics to wandb at log_interval (only on rank 0)
                if rank == 0:
                    global_step = epoch * len(data_loader) + itr
                    wandb_metrics = {}
                    for h, (am, lm) in enumerate(zip(acc_meters, loss_meters)):
                        wandb_metrics[f"train/head_{h}/Acc"] = am.avg
                        wandb_metrics[f"train/head_{h}/Loss"] = lm.avg
                    wandb.log(wandb_metrics, step=global_step)
            else:
                logger.info(
                    f"[Val][Epoch {epoch}][Iter {itr}/{len(data_loader)}] "
                    + " ".join([f"Head{h}: Acc={am.avg:.2f}%" for h, am in enumerate(acc_meters)])
                )

    metrics = {f"head_{i}": {"Acc": acc_meters[i].avg, "Loss": loss_meters[i].avg} for i in range(len(classifiers))}

    if not training and len(all_predictions) > 0:
        df = pd.DataFrame(all_predictions, columns=["head","data_idx","vid","prediction","label","dataset_idx"])
        results = {}

        # Calculate global step for validation logging
        # Use the step at the END of the epoch (after all training steps)
        global_step = (epoch + 1) * train_loader_len

        # Evaluate per head
        for head_id, g in df.groupby("head"):
            per_video, stats, phases = evaluate_per_video(
                g,
                use_bootstrap=use_bootstrap,
                n_bootstrap=n_bootstrap,
                random_seed=bootstrap_seed
            )
            results[f"head_{head_id}"] = stats

            # Log to wandb (only on rank 0)
            if rank == 0:
                wandb_metrics = {
                    f"val/head_{head_id}/Accuracy": stats['Accuracy_Mean'],
                    f"val/head_{head_id}/Macro_F1": stats['Macro_F1_Mean'],
                    f"val/head_{head_id}/Macro_IoU": stats['Macro_IoU_Mean'],
                    f"val/head_{head_id}/Macro_Precision": stats['Macro_Precision_Mean'],
                    f"val/head_{head_id}/Macro_Recall": stats['Macro_Recall_Mean'],
                    f"val/head_{head_id}/Edit_Distance": stats['Edit_Distance_Mean'],
                }

                # Add uncertainty metrics if bootstrap was used
                if use_bootstrap:
                    wandb_metrics.update({
                        f"val/head_{head_id}/Accuracy_Std": stats['Accuracy_Std'],
                        f"val/head_{head_id}/Macro_F1_Std": stats['Macro_F1_Std'],
                        f"val/head_{head_id}/Macro_IoU_Std": stats['Macro_IoU_Std'],
                        f"val/head_{head_id}/Accuracy_CI_Width": stats['Accuracy_CI_Upper'] - stats['Accuracy_CI_Lower'],
                        f"val/head_{head_id}/Macro_F1_CI_Width": stats['Macro_F1_CI_Upper'] - stats['Macro_F1_CI_Lower'],
                    })

                wandb.log(wandb_metrics, step=global_step)

        logger.info("=== Evaluation per head ===")
        for k, v in results.items():
            logger.info(
                f"{k}: "
                f"Acc={v['Accuracy_Mean']:.2f}±{v['Accuracy_Std']:.2f}, "
                f"F1={v['Macro_F1_Mean']:.2f}±{v['Macro_F1_Std']:.2f}, "
                f"IoU={v['Macro_IoU_Mean']:.2f}±{v['Macro_IoU_Std']:.2f}, "
                f"Prec={v['Macro_Precision_Mean']:.2f}±{v['Macro_Precision_Std']:.2f}, "
                f"Rec={v['Macro_Recall_Mean']:.2f}±{v['Macro_Recall_Std']:.2f}, "
                f"Edit={v['Edit_Distance_Mean']:.2f}±{v['Edit_Distance_Std']:.2f}"
            )

            # Log per-class metrics
            if "per_class" in v:
                logger.info(f"  Per-class metrics for {k}:")
                for phase_name, phase_metrics in v["per_class"].items():
                    logger.info(
                        f"    {phase_name}: "
                        f"Prec={phase_metrics['Precision']:.2f}%, "
                        f"Rec={phase_metrics['Recall']:.2f}%, "
                        f"F1={phase_metrics['F1']:.2f}%, "
                        f"IoU={phase_metrics['IoU']:.2f}%"
                    )

        # Per-dataset evaluation if we have multi-dataset setup
        if head_to_dataset_map is not None:
            logger.info("\n=== Evaluation per dataset ===")
            dataset_results = {}
            for ds_idx in df['dataset_idx'].unique():
                ds_df = df[df['dataset_idx'] == ds_idx]
                # Find heads assigned to this dataset
                assigned_heads = [h for h, d in enumerate(head_to_dataset_map) if d == ds_idx]

                logger.info(f"\nDataset {ds_idx} (Heads: {assigned_heads}):")
                for head_id in assigned_heads:
                    head_df = ds_df[ds_df['head'] == head_id]
                    if len(head_df) > 0:
                        per_video, stats, phases = evaluate_per_video(
                            head_df,
                            use_bootstrap=use_bootstrap,
                            n_bootstrap=n_bootstrap,
                            random_seed=bootstrap_seed
                        )
                        dataset_results[f"dataset_{ds_idx}_head_{head_id}"] = stats

                        # Log to wandb (only on rank 0)
                        if rank == 0:
                            wandb_metrics = {
                                f"val/dataset_{ds_idx}/head_{head_id}/Accuracy": stats['Accuracy_Mean'],
                                f"val/dataset_{ds_idx}/head_{head_id}/Macro_F1": stats['Macro_F1_Mean'],
                                f"val/dataset_{ds_idx}/head_{head_id}/Macro_IoU": stats['Macro_IoU_Mean'],
                                f"val/dataset_{ds_idx}/head_{head_id}/Edit_Distance": stats['Edit_Distance_Mean'],
                            }
                            if use_bootstrap:
                                wandb_metrics.update({
                                    f"val/dataset_{ds_idx}/head_{head_id}/Accuracy_Std": stats['Accuracy_Std'],
                                    f"val/dataset_{ds_idx}/head_{head_id}/Macro_F1_Std": stats['Macro_F1_Std'],
                                })
                            wandb.log(wandb_metrics, step=global_step)

                        logger.info(
                            f"  Head {head_id}: "
                            f"Acc={stats['Accuracy_Mean']:.2f}±{stats['Accuracy_Std']:.2f}, "
                            f"F1={stats['Macro_F1_Mean']:.2f}±{stats['Macro_F1_Std']:.2f}, "
                            f"IoU={stats['Macro_IoU_Mean']:.2f}±{stats['Macro_IoU_Std']:.2f}, "
                            f"Edit={stats['Edit_Distance_Mean']:.2f}±{stats['Edit_Distance_Std']:.2f}"
                        )

                        # Log per-class metrics for this dataset
                        if "per_class" in stats:
                            logger.info(f"    Per-class metrics (Dataset {ds_idx}, Head {head_id}):")
                            for phase_name, phase_metrics in stats["per_class"].items():
                                logger.info(
                                    f"      {phase_name}: "
                                    f"Prec={phase_metrics['Precision']:.2f}%, "
                                    f"Rec={phase_metrics['Recall']:.2f}%, "
                                    f"F1={phase_metrics['F1']:.2f}%, "
                                    f"IoU={phase_metrics['IoU']:.2f}%"
                                )
            metrics.update(dataset_results)

        if save_predictions and folder is not None:
            pred_file = os.path.join(folder, f"predictions_epoch_{epoch}.csv")
            df.to_csv(pred_file, index=False)
            logger.info(f"Saved predictions to {pred_file}")

    return metrics


# ----------------------
# checkpoint 加载
# ----------------------
def load_checkpoint(device, r_path, encoder, classifiers, opt, scaler, val_only=False):
    checkpoint = robust_checkpoint_loader(r_path, map_location="cpu")

    encoder.load_state_dict(checkpoint["encoder"])
    for c, state in zip(classifiers, checkpoint["classifiers"]):
        c.module.load_state_dict(state)

    if val_only:
        return encoder, classifiers, opt, scaler, 0

    epoch = checkpoint["epoch"]
    for o, state in zip(opt, checkpoint["opt"]):
        o.load_state_dict(state)
    for s, state in zip(scaler, checkpoint["scaler"]):
        if s is not None and state is not None:
            s.load_state_dict(state)
    return encoder, classifiers, opt, scaler, epoch


# ----------------------
# Quick run / debug utilities
# ----------------------
def create_quick_run_subset(csv_path, num_videos=1, output_dir=None):
    """
    Create a subset CSV file with only N videos for quick debugging runs.

    Args:
        csv_path: Path to original CSV file
        num_videos: Number of videos to include in subset
        output_dir: Directory to save subset CSV (default: same as original)

    Returns:
        Path to subset CSV file
    """
    import pandas as pd
    import os

    df = pd.read_csv(csv_path)

    # Get unique video IDs (case_id column)
    unique_videos = df['case_id'].unique()

    if len(unique_videos) < num_videos:
        logger.warning(
            f"Requested {num_videos} videos but only {len(unique_videos)} available. "
            f"Using all {len(unique_videos)} videos."
        )
        num_videos = len(unique_videos)

    # Select first N videos
    selected_videos = unique_videos[:num_videos]
    subset_df = df[df['case_id'].isin(selected_videos)]

    # Create output path
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)

    basename = os.path.basename(csv_path)
    name_without_ext = os.path.splitext(basename)[0]
    subset_path = os.path.join(f"{name_without_ext}_quick_{num_videos}vid.csv")

    # Save subset
    subset_df.to_csv(subset_path, index=False)
    logger.info(
        f"Created quick run subset: {subset_path} "
        f"({len(subset_df)} samples from {num_videos} video(s))"
    )

    return subset_path


# ----------------------
# dataloader
# ----------------------
DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

def make_dataloader(
    root_path,
    batch_size,
    world_size,
    rank,
    dataset_type="VideoDataset",
    img_size=224,
    frames_per_clip=16,
    frame_step=4,
    num_segments=8,
    eval_duration=None,
    num_views_per_segment=1,
    allow_segment_overlap=True,
    training=False,
    num_workers=12,
    subset_file=None,
    normalization=None,
    datasets_weights=None,
):
    if normalization is None:
        normalization = DEFAULT_NORMALIZATION

    transform = make_transforms(
        training=training,
        num_views_per_clip=num_views_per_segment,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4 / 3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=img_size,
        normalize=normalization,
    )

    data_loader, data_sampler = init_data(
        data=dataset_type,
        root_path=root_path,
        transform=transform,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        clip_len=frames_per_clip,
        frame_sample_rate=frame_step,
        duration=eval_duration,
        num_clips=num_segments,
        allow_clip_overlap=allow_segment_overlap,
        num_workers=num_workers,
        drop_last=False,
        subset_file=subset_file,
        datasets_weights=datasets_weights,
    )
    return data_loader, data_sampler


# ----------------------
# optimizer + scheduler
# ----------------------
def init_opt(classifiers, opt_kwargs, iterations_per_epoch, num_epochs, use_bfloat16=False):
    optimizers, schedulers, wd_schedulers, scalers = [], [], [], []
    for c, kwargs in zip(classifiers, opt_kwargs):
        param_groups = [{
            "params": c.parameters(),
            "mc_warmup_steps": int(kwargs.get("warmup") * iterations_per_epoch),
            "mc_start_lr": kwargs.get("start_lr"),
            "mc_ref_lr": kwargs.get("lr"),
            "mc_final_lr": kwargs.get("final_lr"),
            "mc_ref_wd": kwargs.get("weight_decay"),
            "mc_final_wd": kwargs.get("final_weight_decay"),
        }]
        optim = torch.optim.AdamW(param_groups)
        schedulers.append(WarmupCosineLRSchedule(optim, T_max=int(num_epochs * iterations_per_epoch)))
        wd_schedulers.append(CosineWDSchedule(optim, T_max=int(num_epochs * iterations_per_epoch)))
        optimizers.append(optim)
        scalers.append(torch.cuda.amp.GradScaler() if use_bfloat16 else None)
    return optimizers, scalers, schedulers, wd_schedulers


class WarmupCosineLRSchedule:
    def __init__(self, optimizer, T_max, last_epoch=-1):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0

    def step(self):
        self._step += 1
        for group in self.optimizer.param_groups:
            ref_lr = group.get("mc_ref_lr")
            final_lr = group.get("mc_final_lr")
            start_lr = group.get("mc_start_lr")
            warmup_steps = group.get("mc_warmup_steps")
            T_max = self.T_max - warmup_steps
            if self._step < warmup_steps:
                progress = self._step / max(1, warmup_steps)
                new_lr = start_lr + progress * (ref_lr - start_lr)
            else:
                progress = (self._step - warmup_steps) / max(1, T_max)
                new_lr = max(
                    final_lr, final_lr + (ref_lr - final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
                )
            group["lr"] = new_lr


class CosineWDSchedule:
    def __init__(self, optimizer, T_max):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        for group in self.optimizer.param_groups:
            ref_wd = group.get("mc_ref_wd")
            final_wd = group.get("mc_final_wd")
            new_wd = final_wd + (ref_wd - final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))
            if final_wd <= ref_wd:
                new_wd = max(final_wd, new_wd)
            else:
                new_wd = min(final_wd, new_wd)
            group["weight_decay"] = new_wd