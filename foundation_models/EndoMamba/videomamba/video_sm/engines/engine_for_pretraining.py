import math
import time
import sys
from typing import Iterable
import torch
import torch.nn as nn
import utils
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, tubelet_size=1, wandb_logger=None, 
                    teacher_model=None, embedding_weight=0.25):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = nn.MSELoss()
    embedding_loss = nn.CosineEmbeddingLoss(margin=0.15, reduction='mean')

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        videos, bool_masked_pos = batch
        videos = videos.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]
            unnorm_videos = videos * std + mean  # in [0, 1]

            if normlize_target:
                videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=tubelet_size, p1=patch_size, p2=patch_size)
                videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                    ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
            else:
                videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=tubelet_size, p1=patch_size, p2=patch_size)
                
            B, _, C = videos_patch.shape
            labels = videos_patch[bool_masked_pos].reshape(B, -1, C) # torch.Size([4, 2352, 768])
            
            if teacher_model is not None:
                ft = teacher_model(videos).detach()
                B, _, C = ft.shape
                ft_vis = ft[~bool_masked_pos].reshape(B, -1, C)

        loss = 0
        loss_embedding = 0
        with torch.cuda.amp.autocast():
            outputs, fs = model(videos, bool_masked_pos, tubelet_size) # torch.Size([4, 2352, 1536])
            loss_mse = loss_func(input=outputs, target=labels)
            loss += loss_mse
            if teacher_model is not None:
                target = torch.ones(fs.shape[0], fs.shape[1]).to(device)
                loss_embedding = embedding_loss(fs.reshape(-1, fs.shape[-1]),
                                                        ft_vis.reshape(-1, ft_vis.shape[-1]),
                                                        target.view(-1))
                loss += loss_embedding * embedding_weight

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger is not None:
            wandb_logger.set_step()
            wandb_logger.update(head="scalar", loss=loss, embedding=loss_embedding, mse=loss_mse)
            # Log outputs and labels as images
            if step % 1000 == 0: 
                patch_idx = 0
                videos_s = IMAGENET_DEFAULT_STD
                videos_m = IMAGENET_DEFAULT_MEAN

                label_patch = labels[0, patch_idx, :].detach().cpu().numpy()  # [768]
                label_patch = label_patch.reshape(patch_size, patch_size, 3)  # [16, 16, 3]
                # label_patch = label_patch.transpose(1, 2, 0)  # [16, 16, 3]
                label_patch = np.clip(label_patch / videos_s + videos_m, 0, 1)
                label_patch = (label_patch * 255).astype(np.uint8)

                output_patch = outputs[0, patch_idx, :].detach().cpu().numpy()  # [768]
                output_patch = output_patch.reshape(patch_size, patch_size, 3)  # [16, 16, 3]
                # output_patch = output_patch.transpose(1, 2, 0)  # [16, 16, 3]
                output_patch = np.clip(output_patch / videos_s + videos_m, 0, 1)
                output_patch = (output_patch * 255).astype(np.uint8)

                wandb_logger.update_image(head="image", outputs=output_patch, labels=label_patch)

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    timestep = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestep}] Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}