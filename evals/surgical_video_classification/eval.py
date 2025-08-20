import os
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
    val_only = args_eval.get("val_only", False)
    pretrain_folder = args_eval.get("folder", None)
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    eval_tag = args_eval.get("tag", None)
    num_workers = args_eval.get("num_workers", 12)

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

    args_opt = args_exp.get("optimization")
    batch_size = args_opt.get("batch_size")
    num_epochs = args_opt.get("num_epochs")
    use_bfloat16 = args_opt.get("use_bfloat16")
    opt_kwargs = args_opt.get("multihead_kwargs")[0]  # 只用第一个

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

    folder = os.path.join(pretrain_folder, "video_classification_frozen/")
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")

    # -- CSVLogger
    if rank == 0:
        csv_logger = CSVLogger(
            log_file, 
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

    # -- Build model
    encoder = init_module(
        module_name=module_name,
        frames_per_clip=frames_per_clip,
        resolution=resolution,
        checkpoint=checkpoint,
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device=device,
    )
    classifier = AttentiveClassifier(
        embed_dim=encoder.embed_dim,
        num_heads=num_heads,
        depth=num_probe_blocks,
        num_classes=num_classes,
        use_activation_checkpointing=True,
    ).to(device)

    for p in encoder.parameters():
        p.requires_grad = True

    # DDP wrap
    encoder = DistributedDataParallel(encoder, static_graph=True)
    classifier = DistributedDataParallel(classifier, static_graph=True)

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

    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        classifier=classifier,
        opt_kwargs=opt_kwargs,
        iterations_per_epoch=ipe,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16,
    )

    # -- load training checkpoint
    start_epoch = 0
    if resume_checkpoint and os.path.exists(latest_path):
        encoder, classifier, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=latest_path,
            encoder=encoder,
            classifier=classifier,
            opt=optimizer,
            scaler=scaler,
            val_only=val_only,
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()

    def save_checkpoint(epoch):
        save_dict = {
            "encoder": encoder.module.state_dict(),
            "classifier": classifier.module.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
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
                "acc": -1.0, "precision": -1.0, "recall": -1.0, "f1_score": -1.0,
                "video_precision_mean": -1.0, "video_precision_var": -1.0,
                "video_recall_mean": -1.0, "video_recall_var": -1.0,
                "video_f1_mean": -1.0, "video_f1_var": -1.0
            }
        else:
            logger.info(f"Training phase - Epoch {epoch + 1}/{num_epochs}")
            train_metrics = run_one_epoch(
                device=device,
                training=True,
                encoder=encoder,
                classifier=classifier,
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
            classifier=classifier,
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

        logger.info("\n" + "="*50)
        logger.info(f"Epoch {epoch + 1}/{num_epochs} Summary")
        logger.info("="*50)
        logger.info(
            "TRAIN - acc=%.3f%% prec=%.3f rec=%.3f f1=%.3f"
            % (train_metrics["acc"], train_metrics["precision"], train_metrics["recall"], train_metrics["f1_score"])
        )
        logger.info(
            "VAL   - acc=%.3f%% prec=%.3f rec=%.3f f1=%.3f"
            % (val_metrics["acc"], val_metrics["precision"], val_metrics["recall"], val_metrics["f1_score"])
        )
        logger.info(
            "VAL Video Metrics - prec_mean=%.3f prec_var=%.3f rec_mean=%.3f rec_var=%.3f f1_mean=%.3f f1_var=%.3f"
            % (val_metrics["video_precision_mean"], val_metrics["video_precision_var"], 
               val_metrics["video_recall_mean"], val_metrics["video_recall_var"],
               val_metrics["video_f1_mean"], val_metrics["video_f1_var"])
        )
        if rank == 0:
            csv_logger.log(
                epoch + 1,
                val_metrics["acc"], 
                val_metrics["precision"], 
                val_metrics["recall"], 
                val_metrics["f1_score"],
                val_metrics["video_precision_mean"],
                val_metrics["video_precision_var"],
                val_metrics["video_recall_mean"],
                val_metrics["video_recall_var"],
                val_metrics["video_f1_mean"],
                val_metrics["video_f1_var"]
            )
        logger.info("="*50 + "\n")

        if val_only:
            return

        save_checkpoint(epoch + 1)

def run_one_epoch(
    device,
    training,
    encoder,
    classifier,
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
    encoder.train(mode=training)
    classifier.train(mode=training)

    criterion = torch.nn.CrossEntropyLoss()
    acc_meter = AverageMeter()

    if not training:
        video_predictions = defaultdict(lambda: {'preds': [], 'labels': []})
        detailed_predictions = []
        global_index = 0

    total_iters = len(data_loader)
    for itr, data in enumerate(data_loader):
        if training:
            scheduler.step()
            wd_scheduler.step()
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):
            clips = [
                [dij.to(device, non_blocking=True) for dij in di]
                for di in data[0]
            ]
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
            labels = data[1][0].to(device)
            batch_size = len(labels)
            vid_ids = data[1][1]

            features = encoder(clips, clip_indices)
            outputs = [classifier(f) for f in features]
            output = outputs[0]  # 只取一个

            loss = criterion(output, labels)

        if training:
            if use_bfloat16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            preds = output.max(dim=1).indices
            acc = 100.0 * preds.eq(labels).sum() / batch_size
            acc = float(AllReduce.apply(acc))
            acc_meter.update(acc)

            if not training:
                for pred, label, vid in zip(preds.cpu().numpy(), labels.cpu().numpy(), vid_ids.numpy()):
                    detailed_predictions.append([
                        global_index, vid, pred, label
                    ])
                    global_index += 1
                    video_predictions[vid]['preds'].append(pred)
                    video_predictions[vid]['labels'].append(label)

        if itr % 10 == 0:
            logger.info(
                "[Epoch %d/%d, Iter %5d/%5d] accuracy: %.1f%% [mem: %.2e]"
                % (
                    epoch, num_epochs, itr + 1, total_iters,
                    acc_meter.avg, torch.cuda.max_memory_allocated() / 1024.0**2,
                )
            )

    metrics = {
        "acc": acc_meter.avg,
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

    if not training and len(video_predictions) > 0:
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        if world_size > 1:
            gathered_videos = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(gathered_videos, video_predictions)
            merged_videos = defaultdict(lambda: {'preds': [], 'labels': []})
            for proc_videos in gathered_videos:
                for vid, data in proc_videos.items():
                    merged_videos[vid]['preds'].extend(data['preds'])
                    merged_videos[vid]['labels'].extend(data['labels'])
            video_predictions = merged_videos

            gathered_predictions = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(gathered_predictions, detailed_predictions)
            merged_predictions = []
            for proc_preds in gathered_predictions:
                merged_predictions.extend(proc_preds)
            detailed_predictions = merged_predictions

        all_preds = []
        all_labels = []
        for vid_data in video_predictions.values():
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

        video_metrics = []
        for vid, data in video_predictions.items():
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

        if save_predictions and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            video_metrics_file = os.path.join(folder, f"video_metrics_epoch_{epoch}.csv")
            with open(video_metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['video_id', 'precision', 'recall', 'f1_score', 'num_clips'])
                for m in video_metrics:
                    writer.writerow([m['video_id'], m['precision'], m['recall'], m['f1_score'], m['num_clips']])
            logger.info(f"Saved video metrics to {video_metrics_file}")

            summary_file = os.path.join(folder, f"summary_metrics_epoch_{epoch}.csv")
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
            logger.info(f"Saved summary metrics to {summary_file}")

            predictions_file = os.path.join(folder, f"all_predictions_epoch_{epoch}.csv")
            with open(predictions_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['index', 'vid', 'prediction', 'label'])
                for pred_data in detailed_predictions:
                    writer.writerow(pred_data)
            logger.info(f"Saved all predictions to {predictions_file}")

    return metrics

def load_checkpoint(device, r_path, encoder, classifier, opt, scaler, val_only=False):
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    logger.info(f"read-path: {r_path}")

    encoder.module.load_state_dict(checkpoint["encoder"])
    classifier.module.load_state_dict(checkpoint["classifier"])
    if val_only:
        logger.info(f"loaded pretrained from epoch with val_only")
        return encoder, classifier, opt, scaler, 0

    epoch = checkpoint["epoch"]
    logger.info(f"loaded pretrained from epoch {epoch}")

    opt.load_state_dict(checkpoint["opt"])
    if scaler is not None and checkpoint["scaler"] is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    logger.info(f"loaded optimizers from epoch {epoch}")
    return encoder, classifier, opt, scaler, epoch

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

def init_opt(encoder, classifier, opt_kwargs, iterations_per_epoch, num_epochs, use_bfloat16=False):
    # 设置不同学习率和weight_decay
    param_groups = [
        {
            "params": encoder.parameters(),
            "lr": opt_kwargs.get("encoder_lr", 1e-5),   # 可通过config传入
            "weight_decay": opt_kwargs.get("encoder_wd", 0.05),
        },
        {
            "params": classifier.parameters(),
            "lr": opt_kwargs.get("classifier_lr", 1e-3),
            "weight_decay": opt_kwargs.get("classifier_wd", 0.01),
            "mc_warmup_steps": int(opt_kwargs.get("warmup") * iterations_per_epoch),
            "mc_start_lr": opt_kwargs.get("start_lr"),
            "mc_ref_lr": opt_kwargs.get("lr"),
            "mc_final_lr": opt_kwargs.get("final_lr"),
            "mc_ref_wd": opt_kwargs.get("weight_decay"),
            "mc_final_wd": opt_kwargs.get("final_weight_decay"),
        }
    ]
    logger.info("Using AdamW")
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineLRSchedule(optimizer, T_max=int(num_epochs * iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(optimizer, T_max=int(num_epochs * iterations_per_epoch))
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler

class WarmupCosineLRSchedule(object):
    def __init__(self, optimizer, T_max, last_epoch=-1):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0
    def step(self):
        self._step += 1
        for i, group in enumerate(self.optimizer.param_groups):
            # 只对 classifier param_group 调度
            if "mc_ref_lr" not in group or "mc_final_lr" not in group or "mc_start_lr" not in group:
                continue
            ref_lr = group.get("mc_ref_lr")
            final_lr = group.get("mc_final_lr")
            start_lr = group.get("mc_start_lr")
            warmup_steps = group.get("mc_warmup_steps", 0)
            T_max = self.T_max - warmup_steps
            if self._step < warmup_steps:
                progress = float(self._step) / float(max(1, warmup_steps))
                new_lr = start_lr + progress * (ref_lr - start_lr)
            else:
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
            # 只对 classifier param_group 调度
            if "mc_ref_wd" not in group or "mc_final_wd" not in group:
                continue
            ref_wd = group.get("mc_ref_wd")
            final_wd = group.get("mc_final_wd")
            new_wd = final_wd + (ref_wd - final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))
            if final_wd <= ref_wd:
                new_wd = max(final_wd, new_wd)
            else:
                new_wd = min(final_wd, new_wd)
            group["weight_decay"] = new_wd