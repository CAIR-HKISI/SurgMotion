# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import logging
import math
import pprint
import csv
from collections import defaultdict

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from sklearn.metrics import precision_recall_fscore_support

from evals.surgical_video_classification_frozen.models import init_module
from evals.video_classification_frozen.utils import make_transforms
from src.datasets.data_manager import init_data
from src.models.attentive_pooler import AttentiveClassifier
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import AllReduce, init_distributed
from src.utils.logging import AverageMeter, CSVLogger

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def main(args_eval, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- VAL ONLY
    val_only = args_eval.get("val_only", False)
    if val_only:
        logger.info("VAL ONLY")

    # -- EXPERIMENT
    pretrain_folder = args_eval.get("folder", None)
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    eval_tag = args_eval.get("tag", None)
    num_workers = args_eval.get("num_workers", 12)

    # -- PRETRAIN
    args_pretrain = args_eval.get("model_kwargs")
    checkpoint = args_pretrain.get("checkpoint")
    module_name = args_pretrain.get("module_name")
    args_model = args_pretrain.get("pretrain_kwargs")
    args_wrapper = args_pretrain.get("wrapper_kwargs")

    args_exp = args_eval.get("experiment")

    # -- CLASSIFIER
    args_classifier = args_exp.get("classifier")
    num_probe_blocks = args_classifier.get("num_probe_blocks", 1)
    num_heads = args_classifier.get("num_heads", 16)

    # -- DATA
    args_data = args_exp.get("data")
    dataset_type = args_data.get("dataset_type", "VideoDataset")
    num_classes = args_data.get("num_classes")
    train_data_path = [args_data.get("dataset_train")]
    val_data_path = [args_data.get("dataset_val")]
    resolution = args_data.get("resolution", 224)
    num_segments = args_data.get("num_segments", 1)
    frames_per_clip = args_data.get("frames_per_clip", 16)
    frame_step = args_data.get("frame_step", 1)
    duration = args_data.get("clip_duration", None)
    num_views_per_segment = args_data.get("num_views_per_segment", 1)
    normalization = args_data.get("normalization", None)

    # -- OPTIMIZATION
    args_opt = args_exp.get("optimization")
    batch_size = args_opt.get("batch_size")
    num_epochs = args_opt.get("num_epochs")
    use_bfloat16 = args_opt.get("use_bfloat16")
    opt_kwargs = [
        dict(
            ref_wd=kwargs.get("weight_decay"),
            final_wd=kwargs.get("final_weight_decay"),
            start_lr=kwargs.get("start_lr"),
            ref_lr=kwargs.get("lr"),
            final_lr=kwargs.get("final_lr"),
            warmup=kwargs.get("warmup"),
        )
        for kwargs in args_opt.get("multihead_kwargs")
    ]
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # -- log/checkpointing paths
    folder = os.path.join(pretrain_folder, "video_classification_frozen/")
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")

    # -- make csv_logger for each classifier
    if rank == 0:
        csv_loggers = []
        for c_idx in range(len(opt_kwargs)):
            log_file_c = os.path.join(folder, f"log_classifier_{c_idx}_r{rank}.csv")
            csv_logger = CSVLogger(
                log_file_c, 
                ("%d", "epoch"), 
                ("%.5f", "acc"), 
                ("%.5f", "precision"), 
                ("%.5f", "recall"), 
                ("%.5f", "f1_score"),
                ("%.5f", "video_precision_mean"),
                ("%.5f", "video_precision_var"),
                ("%.5f", "video_recall_mean"),
                ("%.5f", "video_recall_var"),
                ("%.5f", "video_f1_mean"),
                ("%.5f", "video_f1_var")
            )
            csv_loggers.append(csv_logger)

    # Initialize model

    # -- init models
    encoder = init_module(
        module_name=module_name,
        frames_per_clip=frames_per_clip,
        resolution=resolution,
        checkpoint=checkpoint,
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device=device,
    )
    
    # -- init classifier
    classifiers = [
        AttentiveClassifier(
            embed_dim=encoder.embed_dim,
            num_heads=num_heads,
            depth=num_probe_blocks,
            num_classes=num_classes,
            use_activation_checkpointing=True,
        ).to(device)
        for _ in opt_kwargs
    ]
    classifiers = [DistributedDataParallel(c, static_graph=True) for c in classifiers]
    print(classifiers[0])

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
        normalization=normalization
    )
    ipe = len(train_loader)
    logger.info(f"Dataloader created... iterations per epoch: {ipe}")

    # -- optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        classifiers=classifiers,
        opt_kwargs=opt_kwargs,
        iterations_per_epoch=ipe,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16,
    )

    # -- load training checkpoint
    start_epoch = 0
    if resume_checkpoint and os.path.exists(latest_path):
        classifiers, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=latest_path,
            classifiers=classifiers,
            opt=optimizer,
            scaler=scaler,
            val_only=val_only,
        )
        for _ in range(start_epoch * ipe):
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

    def save_checkpoint(epoch):
        all_classifier_dicts = [c.state_dict() for c in classifiers]
        all_opt_dicts = [o.state_dict() for o in optimizer]

        save_dict = {
            "classifiers": all_classifier_dicts,
            "opt": all_opt_dicts,
            "scaler": None if scaler is None else [s.state_dict() for s in scaler],
            "epoch": epoch,
            "batch_size": batch_size,
            "world_size": world_size,
        }
        if rank == 0:
            torch.save(save_dict, latest_path)

    # TRAIN LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info("="*50)
        logger.info(f"Starting Epoch {epoch + 1}/{num_epochs}")
        logger.info("="*50)
        train_sampler.set_epoch(epoch)
        
        if val_only:
            train_metrics = {
                "classifier_metrics": [{"acc": -1.0, "precision": -1.0, "recall": -1.0, 
                                       "f1_score": -1.0, "video_precision_mean": -1.0,
                                       "video_precision_var": -1.0, "video_recall_mean": -1.0,
                                       "video_recall_var": -1.0, "video_f1_mean": -1.0,
                                       "video_f1_var": -1.0} 
                                      for _ in classifiers]
            }
        else:
            logger.info(f"Training phase - Epoch {epoch + 1}/{num_epochs}")
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
                epoch=epoch + 1,
                num_epochs=num_epochs,
                folder=folder,
                save_predictions=False,
            )

        logger.info(f"\nValidation phase - Epoch {epoch + 1}/{num_epochs}")
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
            epoch=epoch + 1,
            num_epochs=num_epochs,
            folder=folder,
            save_predictions=True,  # Save predictions during validation
        )

        # Log results for each classifier
        logger.info("\n" + "="*50)
        logger.info(f"Epoch {epoch + 1}/{num_epochs} Summary")
        logger.info("="*50)
        
        for c_idx, (train_m, val_m) in enumerate(zip(train_metrics["classifier_metrics"], 
                                                     val_metrics["classifier_metrics"])):
            logger.info(f"\nClassifier {c_idx}:")
            logger.info(
                "TRAIN - acc=%.3f%% prec=%.3f rec=%.3f f1=%.3f"
                % (train_m["acc"], train_m["precision"], train_m["recall"], 
                   train_m["f1_score"])
            )
            logger.info(
                "VAL   - acc=%.3f%% prec=%.3f rec=%.3f f1=%.3f"
                % (val_m["acc"], val_m["precision"], val_m["recall"], 
                   val_m["f1_score"])
            )
            logger.info(
                "VAL Video Metrics - prec_mean=%.3f prec_var=%.3f rec_mean=%.3f rec_var=%.3f f1_mean=%.3f f1_var=%.3f"
                % (val_m["video_precision_mean"], val_m["video_precision_var"], 
                   val_m["video_recall_mean"], val_m["video_recall_var"],
                   val_m["video_f1_mean"], val_m["video_f1_var"])
            )
            
            # Log to CSV
            if rank == 0:
                csv_loggers[c_idx].log(
                    epoch + 1,
                    val_m["acc"], 
                    val_m["precision"], 
                    val_m["recall"], 
                    val_m["f1_score"],
                    val_m["video_precision_mean"],
                    val_m["video_precision_var"],
                    val_m["video_recall_mean"],
                    val_m["video_recall_var"],
                    val_m["video_f1_mean"],
                    val_m["video_f1_var"]
                )
        logger.info("="*50 + "\n")

        if val_only:
            return

        save_checkpoint(epoch + 1)


def run_one_epoch(
    device,
    training,
    encoder,
    classifiers,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
    num_classes,
    epoch,
    num_epochs,
    folder,
    save_predictions=False,
):

    for c in classifiers:
        c.train(mode=training)

    criterion = torch.nn.CrossEntropyLoss()
    
    # Meters for each classifier
    acc_meters = [AverageMeter() for _ in classifiers]
    
    # Store predictions and labels for each classifier, grouped by video ID
    if not training:
        # 每个分类器都有一个字典，键是视频ID，值是(预测列表, 标签列表)
        video_predictions = [defaultdict(lambda: {'preds': [], 'labels': []}) for _ in classifiers]
        
        # 存储详细预测结果 [index, vid, prediction, label]
        detailed_predictions = [[] for _ in classifiers]
        global_index = 0  # 用于跟踪全局索引
    
    # 获取总的iteration数量
    total_iters = len(data_loader)
    
    for itr, data in enumerate(data_loader):
        if training:
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):
            # Load data and put on GPU
            ## data:[clips, [labels, vids], clip_indices]
            clips = [
                [dij.to(device, non_blocking=True) for dij in di]
                for di in data[0]
            ]
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
            labels = data[1][0].to(device)
            batch_size = len(labels)
            
            vid_ids = data[1][1]  # 获取视频ID列表

            # Forward and prediction
            with torch.no_grad():
                outputs = encoder(clips, clip_indices)
                if not training:
                    outputs = [[c(o) for o in outputs] for c in classifiers]
            if training:
                outputs = [[c(o) for o in outputs] for c in classifiers]

        # Compute loss
        losses = [[criterion(o, labels) for o in coutputs] for coutputs in outputs]
        
        with torch.no_grad():
            # Process each classifier separately
            for c_idx, coutputs in enumerate(outputs):
                # 由于 num_segments=1，coutputs 只有一个元素
                output = coutputs[0]  # 直接取第一个（也是唯一的）输出
                
                # Predictions
                preds = output.max(dim=1).indices
                acc = 100.0 * preds.eq(labels).sum() / batch_size
                acc = float(AllReduce.apply(acc))
                acc_meters[c_idx].update(acc)
                
                # 按视频ID存储预测和标签
                if not training:
                    # 保存详细预测结果 [index, vid, prediction, label]
                    for pred, label, vid in zip(preds.cpu().numpy(), labels.cpu().numpy(), vid_ids.numpy()):
                        detailed_predictions[c_idx].append([
                            global_index,  # 全局索引
                            vid,           # 视频ID
                            pred,          # 预测结果
                            label          # 真实标签
                        ])
                        global_index += 1  # 增加全局索引
                        
                        # 同时更新视频级别的预测集合
                        video_predictions[c_idx][vid]['preds'].append(pred)
                        video_predictions[c_idx][vid]['labels'].append(label)

        if training:
            if use_bfloat16:
                # 简化：直接处理单个loss
                for s, losses_per_classifier in zip(scaler, losses):
                    s.scale(losses_per_classifier[0]).backward()
                [s.step(o) for s, o in zip(scaler, optimizer)]
                [s.update() for s in scaler]
            else:
                [losses_per_classifier[0].backward() for losses_per_classifier in losses]
                [o.step() for o in optimizer]
            [o.zero_grad() for o in optimizer]

        if itr % 10 == 0:
            _acc = [am.avg for am in acc_meters]
            
            # 修改日志格式，显示每个分类器的准确率
            acc_str = " ".join([f"C{i}:{a:.1f}%" for i, a in enumerate(_acc)])
            logger.info(
                "[Epoch %d/%d, Iter %5d/%5d] accuracy: %s [mem: %.2e]"
                % (
                    epoch,
                    num_epochs,
                    itr + 1,
                    total_iters,
                    acc_str,
                    torch.cuda.max_memory_allocated() / 1024.0**2,
                )
            )

    # Calculate metrics for each classifier at the end of epoch
    classifier_metrics = []
    all_predictions = []  # 存储所有分类器的预测结果，用于合并保存
    
    for c_idx in range(len(classifiers)):
        metrics = {
            "acc": acc_meters[c_idx].avg,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "video_precision_mean": 0.0,
            "video_precision_var": 0.0,
            "video_recall_mean": 0.0,
            "video_recall_var": 0.0,
            "video_f1_mean": 0.0,
            "video_f1_var": 0.0
        }
        
        # Calculate precision, recall, f1 only at epoch end and only for validation
        if not training and len(video_predictions[c_idx]) > 0:
            world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            
            # 收集所有进程的视频预测结果
            if world_size > 1:
                gathered_videos = [None for _ in range(world_size)]
                torch.distributed.all_gather_object(gathered_videos, video_predictions[c_idx])
                
                # 合并所有进程的视频预测结果
                merged_videos = defaultdict(lambda: {'preds': [], 'labels': []})
                for proc_videos in gathered_videos:
                    for vid, data in proc_videos.items():
                        merged_videos[vid]['preds'].extend(data['preds'])
                        merged_videos[vid]['labels'].extend(data['labels'])
                video_predictions[c_idx] = merged_videos
                
                # 收集所有进程的详细预测结果
                gathered_predictions = [None for _ in range(world_size)]
                torch.distributed.all_gather_object(gathered_predictions, detailed_predictions[c_idx])
                
                # 合并所有进程的详细预测结果
                merged_predictions = []
                for proc_preds in gathered_predictions:
                    merged_predictions.extend(proc_preds)
                detailed_predictions[c_idx] = merged_predictions
            
            # 将当前分类器的预测结果添加到总列表，增加classifier_id列
            for pred_data in detailed_predictions[c_idx]:
                all_predictions.append([c_idx] + pred_data)
            
            # 计算整个数据集的总体指标
            all_preds = []
            all_labels = []
            for vid_data in video_predictions[c_idx].values():
                all_preds.extend(vid_data['preds'])
                all_labels.extend(vid_data['labels'])
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, 
                all_preds, 
                average='macro',
                zero_division=0
            )
            
            metrics["precision"] = precision * 100.0
            metrics["recall"] = recall * 100.0
            metrics["f1_score"] = f1 * 100.0
            
            # 计算每个视频的指标
            video_metrics = []
            for vid, data in video_predictions[c_idx].items():
                # 对于单个视频，使用micro平均
                vid_precision, vid_recall, vid_f1, _ = precision_recall_fscore_support(
                    data['labels'], 
                    data['preds'], 
                    average='macro',
                    zero_division=0
                )
                video_metrics.append({
                    'video_id': vid,
                    'precision': vid_precision * 100.0,
                    'recall': vid_recall * 100.0,
                    'f1_score': vid_f1 * 100.0,
                    'num_clips': len(data['preds'])
                })
                logger.info(f"####### Vid {vid}: precision {vid_precision}, recall {vid_recall}, f1 score {vid_f1}.")
            
            # 计算所有视频指标的平均值和方差
            if video_metrics:
                precisions = [m['precision'] for m in video_metrics]
                recalls = [m['recall'] for m in video_metrics]
                f1_scores = [m['f1_score'] for m in video_metrics]
                
                metrics["video_precision_mean"] = np.mean(precisions)
                metrics["video_precision_var"] = np.std(precisions)
                metrics["video_recall_mean"] = np.mean(recalls)
                metrics["video_recall_var"] = np.std(recalls)
                metrics["video_f1_mean"] = np.mean(f1_scores)
                metrics["video_f1_var"] = np.std(f1_scores)
            
            # 保存每个视频的指标
            if save_predictions and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
                video_metrics_file = os.path.join(folder, f"video_metrics_classifier_{c_idx}_epoch_{epoch}.csv")
                with open(video_metrics_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['video_id', 'precision', 'recall', 'f1_score', 'num_clips'])
                    for m in video_metrics:
                        writer.writerow([m['video_id'], m['precision'], m['recall'], m['f1_score'], m['num_clips']])
                logger.info(f"Saved video metrics for classifier {c_idx} to {video_metrics_file}")
                
                # 保存总体指标
                summary_file = os.path.join(folder, f"summary_metrics_classifier_{c_idx}_epoch_{epoch}.csv")
                with open(summary_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['metric', 'value'])
                    writer.writerow(['overall_accuracy', metrics['acc']])
                    writer.writerow(['overall_precision', metrics['precision']])
                    writer.writerow(['overall_recall', metrics['recall']])
                    writer.writerow(['overall_f1_score', metrics['f1_score']])
                    writer.writerow(['video_precision_mean', metrics['video_precision_mean']])
                    writer.writerow(['video_precision_var', metrics['video_precision_var']])
                    writer.writerow(['video_recall_mean', metrics['video_recall_mean']])
                    writer.writerow(['video_recall_var', metrics['video_recall_var']])
                    writer.writerow(['video_f1_mean', metrics['video_f1_mean']])
                    writer.writerow(['video_f1_var', metrics['video_f1_var']])
                logger.info(f"Saved summary metrics for classifier {c_idx} to {summary_file}")
        
        classifier_metrics.append(metrics)
    
    # 保存所有分类器的预测结果到一个文件
    if not training and save_predictions and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
        predictions_file = os.path.join(folder, f"all_predictions_epoch_{epoch}.csv")
        with open(predictions_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # 添加classifier_id列以区分不同分类器的预测
            writer.writerow(['classifier_id', 'index', 'vid', 'prediction', 'label'])
            for pred_data in all_predictions:
                writer.writerow(pred_data)
        logger.info(f"Saved all predictions to {predictions_file}")
    
    if not training:
        # Log individual classifier performance at epoch end
        logger.info(f"\n[Epoch {epoch}/{num_epochs}] End of Epoch - Classifier Performance:")
        for c_idx, metrics in enumerate(classifier_metrics):
            logger.info(
                f"Classifier {c_idx}: Accuracy: {metrics['acc']:.3f}%, "
                f"Precision: {metrics['precision']:.3f}%, "
                f"Recall: {metrics['recall']:.3f}%, "
                f"F1-Score: {metrics['f1_score']:.3f}%"
            )
            logger.info(
                f"Video Metrics: Precision (mean±var): {metrics['video_precision_mean']:.3f}±{metrics['video_precision_var']:.3f}%, "
                f"Recall (mean±var): {metrics['video_recall_mean']:.3f}±{metrics['video_recall_var']:.3f}%, "
                f"F1-Score (mean±var): {metrics['video_f1_mean']:.3f}±{metrics['video_f1_var']:.3f}%"
            )
    
    return {"classifier_metrics": classifier_metrics}


def load_checkpoint(device, r_path, classifiers, opt, scaler, val_only=False):
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    logger.info(f"read-path: {r_path}")

    # -- loading encoder
    pretrained_dict = checkpoint["classifiers"]
    msg = [c.load_state_dict(pd) for c, pd in zip(classifiers, pretrained_dict)]

    if val_only:
        logger.info(f"loaded pretrained classifier from epoch with msg: {msg}")
        return classifiers, opt, scaler, 0

    epoch = checkpoint["epoch"]
    logger.info(f"loaded pretrained classifier from epoch {epoch} with msg: {msg}")

    # -- loading optimizer
    [o.load_state_dict(pd) for o, pd in zip(opt, checkpoint["opt"])]

    if scaler is not None:
        [s.load_state_dict(pd) for s, pd in zip(scaler, checkpoint["scaler"])]

    logger.info(f"loaded optimizers from epoch {epoch}")

    return classifiers, opt, scaler, epoch


def load_pretrained(encoder, pretrained, checkpoint_key="target_encoder"):
    logger.info(f"Loading pretrained model from {pretrained}")
    checkpoint = robust_checkpoint_loader(pretrained, map_location="cpu")
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint["encoder"]

    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f"key '{k}' could not be found in loaded state dict")
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f"{pretrained_dict[k].shape} | {v.shape}")
            logger.info(f"key '{k}' is of different shape in model and loaded state dict")
            exit(1)
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    print(encoder)
    logger.info(f"loaded pretrained model with msg: {msg}")
    logger.info(f"loaded pretrained encoder from epoch: {checkpoint['epoch']}\n path: {pretrained}")
    del checkpoint
    return encoder


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
):
    if normalization is None:
        normalization = DEFAULT_NORMALIZATION

    # Make Video Transforms
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
    )
    return data_loader, data_sampler


def init_opt(classifiers, iterations_per_epoch, opt_kwargs, num_epochs, use_bfloat16=False):
    optimizers, schedulers, wd_schedulers, scalers = [], [], [], []
    for c, kwargs in zip(classifiers, opt_kwargs):
        param_groups = [
            {
                "params": (p for n, p in c.named_parameters()),
                "mc_warmup_steps": int(kwargs.get("warmup") * iterations_per_epoch),
                "mc_start_lr": kwargs.get("start_lr"),
                "mc_ref_lr": kwargs.get("ref_lr"),
                "mc_final_lr": kwargs.get("final_lr"),
                "mc_ref_wd": kwargs.get("ref_wd"),
                "mc_final_wd": kwargs.get("final_wd"),
            }
        ]
        logger.info("Using AdamW")
        optimizers += [torch.optim.AdamW(param_groups)]
        schedulers += [WarmupCosineLRSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch))]
        wd_schedulers += [CosineWDSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch))]
        scalers += [torch.cuda.amp.GradScaler() if use_bfloat16 else None]
    return optimizers, scalers, schedulers, wd_schedulers


class WarmupCosineLRSchedule(object):

    def __init__(self, optimizer, T_max, last_epoch=-1):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        for group in self.optimizer.param_groups:
            ref_lr = group.get("mc_ref_lr")
            final_lr = group.get("mc_final_lr")
            start_lr = group.get("mc_start_lr")
            warmup_steps = group.get("mc_warmup_steps")
            T_max = self.T_max - warmup_steps
            if self._step < warmup_steps:
                progress = float(self._step) / float(max(1, warmup_steps))
                new_lr = start_lr + progress * (ref_lr - start_lr)
            else:
                # -- progress after warmup
                progress = float(self._step - warmup_steps) / float(max(1, T_max))
                new_lr = max(
                    final_lr,
                    final_lr + (ref_lr - final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)),
                )
            group["lr"] = new_lr


class CosineWDSchedule(object):

    def __init__(self, optimizer, T_max):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

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
