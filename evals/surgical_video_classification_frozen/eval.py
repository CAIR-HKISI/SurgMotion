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
    frame_step = args_data.get("frame_step", 4)
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

    # -- make csv_logger without frame_acc
    if rank == 0:
        csv_logger = CSVLogger(
            log_file, 
            ("%d", "epoch"), 
            # Best metrics
            ("%.5f", "acc_best"), 
            ("%.5f", "precision_best"), ("%.5f", "recall_best"), ("%.5f", "f1_best"),
            # Mean metrics
            ("%.5f", "acc_mean"), 
            ("%.5f", "precision_mean"), ("%.5f", "recall_mean"), ("%.5f", "f1_mean"),
            # Worst metrics
            ("%.5f", "acc_worst"), 
            ("%.5f", "precision_worst"), ("%.5f", "recall_worst"), ("%.5f", "f1_worst")
        )

    # Initialize model

    # -- init models
    # if "mae" in module_name:
    #     encoder = init_videomae_module(
    #         module_name=module_name,
    #         frames_per_clip=frames_per_clip,
    #         resolution=resolution,
    #         checkpoint=checkpoint,
    #         model_kwargs=args_model,
    #         wrapper_kwargs=args_wrapper,
    #         device=device,
    #     )
    # else:
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
        normalization=normalization,
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
                "all_metrics": [{"acc": -1.0, "precision": -1.0, "recall": -1.0, "f1_score": -1.0}]
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
        )

        # Extract aggregated metrics
        train_agg = aggregate_metrics(train_metrics["all_metrics"])
        val_agg = aggregate_metrics(val_metrics["all_metrics"])
        
        logger.info("\n" + "="*50)
        logger.info(f"Epoch {epoch + 1}/{num_epochs} Summary")
        logger.info("="*50)
        
        logger.info(
            "TRAIN - Best: acc=%.3f%% prec=%.3f rec=%.3f f1=%.3f | "
            "Mean: acc=%.3f%% prec=%.3f rec=%.3f f1=%.3f | "
            "Worst: acc=%.3f%% prec=%.3f rec=%.3f f1=%.3f"
            % (
                train_agg["best"]["acc"], 
                train_agg["best"]["precision"], train_agg["best"]["recall"], train_agg["best"]["f1_score"],
                train_agg["mean"]["acc"], 
                train_agg["mean"]["precision"], train_agg["mean"]["recall"], train_agg["mean"]["f1_score"],
                train_agg["worst"]["acc"], 
                train_agg["worst"]["precision"], train_agg["worst"]["recall"], train_agg["worst"]["f1_score"],
            )
        )

        logger.info(
            "VAL   - Best: acc=%.3f%% prec=%.3f rec=%.3f f1=%.3f | "
            "Mean: acc=%.3f%% prec=%.3f rec=%.3f f1=%.3f | "
            "Worst: acc=%.3f%% prec=%.3f rec=%.3f f1=%.3f"
            % (
                val_agg["best"]["acc"], 
                val_agg["best"]["precision"], val_agg["best"]["recall"], val_agg["best"]["f1_score"],
                val_agg["mean"]["acc"], 
                val_agg["mean"]["precision"], val_agg["mean"]["recall"], val_agg["mean"]["f1_score"],
                val_agg["worst"]["acc"], 
                val_agg["worst"]["precision"], val_agg["worst"]["recall"], val_agg["worst"]["f1_score"],
            )
        )
        logger.info("="*50 + "\n")
             
        # 记录日志时 - 不再包含 frame_acc
        if rank == 0:
            csv_logger.log(
                epoch + 1,
                # Best metrics
                val_agg["best"]["acc"], 
                val_agg["best"]["precision"], 
                val_agg["best"]["recall"], 
                val_agg["best"]["f1_score"],
                # Mean metrics
                val_agg["mean"]["acc"], 
                val_agg["mean"]["precision"], 
                val_agg["mean"]["recall"], 
                val_agg["mean"]["f1_score"],
                # Worst metrics
                val_agg["worst"]["acc"], 
                val_agg["worst"]["precision"], 
                val_agg["worst"]["recall"], 
                val_agg["worst"]["f1_score"]
            )

        if val_only:
            return

        save_checkpoint(epoch + 1)


def aggregate_metrics(all_metrics):
    """Aggregate metrics across all classifiers to get best, mean, worst"""
    if len(all_metrics) == 1:
        # Single classifier case
        return {
            "best": all_metrics[0],
            "mean": all_metrics[0],
            "worst": all_metrics[0]
        }
    
    # Multiple classifiers case
    metric_names = ["acc", "precision", "recall", "f1_score"]
    
    # Find best and worst based on accuracy
    acc_values = [m["acc"] for m in all_metrics]
    best_idx = np.argmax(acc_values)
    worst_idx = np.argmin(acc_values)
    
    # Calculate mean
    mean_metrics = {}
    for metric in metric_names:
        values = [m[metric] for m in all_metrics]
        mean_metrics[metric] = np.mean(values)
    
    return {
        "best": all_metrics[best_idx],
        "mean": mean_metrics,
        "worst": all_metrics[worst_idx]
    }


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
):

    for c in classifiers:
        c.train(mode=training)

    criterion = torch.nn.CrossEntropyLoss()
    
    # Meters for each classifier
    acc_meters = [AverageMeter() for _ in classifiers]
    
    # Store predictions and labels for each classifier (only for validation)
    if not training:
        all_predictions = [[] for _ in classifiers]
        all_labels = [[] for _ in classifiers]
    
    # 获取总的iteration数量
    total_iters = len(data_loader)
    
    for itr, data in enumerate(data_loader):
        if training:
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):
            # Load data and put on GPU
            clips = [
                [dij.to(device, non_blocking=True) for dij in di]
                for di in data[0]
            ]
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
            labels = data[1].to(device)
            batch_size = len(labels)

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
                
                # Store predictions for metrics calculation (only during validation)
                if not training:
                    all_predictions[c_idx].extend(preds.cpu().numpy())
                    all_labels[c_idx].extend(labels.cpu().numpy())

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
            _agg_acc = np.array([am.avg for am in acc_meters])
            
            # 修改日志格式，添加epoch和iter信息
            logger.info(
                "[Epoch %d/%d, Iter %5d/%5d] accuracy: %.3f%% [mean: %.3f%% worst: %.3f%%] [mem: %.2e]"
                % (
                    epoch,
                    num_epochs,
                    itr + 1,
                    total_iters,
                    _agg_acc.max(),
                    _agg_acc.mean(),
                    _agg_acc.min(),
                    torch.cuda.max_memory_allocated() / 1024.0**2,
                )
            )

    # Calculate metrics for each classifier at the end of epoch
    all_metrics = []
    
    for c_idx in range(len(classifiers)):
        metrics = {
            "acc": acc_meters[c_idx].avg,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }
        
        # Calculate precision, recall, f1 only at epoch end and only for validation
        if not training and len(all_predictions[c_idx]) > 0:
            # Gather all predictions and labels from all ranks
            world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            
            if world_size > 1:
                gathered_predictions = [None for _ in range(world_size)]
                gathered_labels = [None for _ in range(world_size)]
                
                torch.distributed.all_gather_object(gathered_predictions, all_predictions[c_idx])
                torch.distributed.all_gather_object(gathered_labels, all_labels[c_idx])
                
                all_predictions[c_idx] = [pred for preds in gathered_predictions for pred in preds]
                all_labels[c_idx] = [label for labels in gathered_labels for label in labels]
            
            # Calculate metrics using macro averaging
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels[c_idx], 
                all_predictions[c_idx], 
                average='macro',
                zero_division=0
            )
            
            metrics["precision"] = precision * 100.0
            metrics["recall"] = recall * 100.0
            metrics["f1_score"] = f1 * 100.0
        
        all_metrics.append(metrics)
    
    if not training:
        # Log individual classifier performance at epoch end
        logger.info(f"\n[Epoch {epoch}/{num_epochs}] End of Epoch - Individual Classifier Performance:")
        for c_idx, metrics in enumerate(all_metrics):
            logger.info(
                f"Classifier {c_idx}: Accuracy: {metrics['acc']:.3f}%, "
                f"Precision: {metrics['precision']:.3f}%, "
                f"Recall: {metrics['recall']:.3f}%, "
                f"F1-Score: {metrics['f1_score']:.3f}%"
            )
        
        # Log aggregated metrics
        agg = aggregate_metrics(all_metrics)
        logger.info("\nAggregated Performance:")
        logger.info(
            f"Best - Accuracy: {agg['best']['acc']:.3f}%, "
            f"Precision: {agg['best']['precision']:.3f}%, "
            f"Recall: {agg['best']['recall']:.3f}%, "
            f"F1-Score: {agg['best']['f1_score']:.3f}%"
        )
        logger.info(
            f"Mean - Accuracy: {agg['mean']['acc']:.3f}%, "
            f"Precision: {agg['mean']['precision']:.3f}%, "
            f"Recall: {agg['mean']['recall']:.3f}%, "
            f"F1-Score: {agg['mean']['f1_score']:.3f}%"
        )
        logger.info(
            f"Worst - Accuracy: {agg['worst']['acc']:.3f}%, "
            f"Precision: {agg['worst']['precision']:.3f}%, "
            f"Recall: {agg['worst']['recall']:.3f}%, "
            f"F1-Score: {agg['worst']['f1_score']:.3f}%"
        )
    
    return {"all_metrics": all_metrics}


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

