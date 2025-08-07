#!/usr/bin/env python
# coding: utf-8

"""
Surgical Phase Recognition with PyTorch Lightning
- 使用 LightningDataModule 管理数据
- 以 epoch 为单位训练/测试，由 --max_epochs 和 --min_epochs 控制
- 自动保存最佳模型，并在训练结束后用最佳模型做测试
"""

import os
import json
import argparse
import time
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoModelForImageClassification,
    get_cosine_schedule_with_warmup,
    set_seed,
)
from tqdm.auto import tqdm

import timm
import sys
sys.path.append(".")
from src.models.attentive_pooler import AttentiveClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Surgical Phase Recognition (epoch-based)")
    dg = parser.add_argument_group('Data')
    dg.add_argument("--data_dir", type=str, required=True)
    dg.add_argument("--train_csv", type=str, required=True)
    dg.add_argument("--val_csv", type=str, required=True)
    dg.add_argument("--pretrained_model_path", type=str, required=True)
    dg.add_argument("--output_dir", type=str, default="./output")
    dg.add_argument("--limit_train_batches", type=float, default=1.0)
    dg.add_argument("--limit_val_batches", type=float, default=1.0)

    eg = parser.add_argument_group('Epochs')
    eg.add_argument("--max_epochs", type=int, required=True, help="训练最大轮数")
    eg.add_argument("--min_epochs", type=int, default=5, help="训练最小轮数")
    eg.add_argument("--eval_epochs", type=int, default=1, help="评估频率(轮数)")

    tg = parser.add_argument_group('Training')
    tg.add_argument("--image_size", type=int, default=224)
    tg.add_argument("--train_bs", type=int, default=16)
    tg.add_argument("--eval_bs", type=int, default=16)
    tg.add_argument("--lr", type=float, default=2e-5)
    tg.add_argument("--weight_decay", type=float, default=0.0)
    tg.add_argument("--warmup_ratio", type=float, default=0.0)

    ag = parser.add_argument_group('Augmentation')
    ag.add_argument("--mean", nargs=3, type=float, default=[0.5,0.5,0.5])
    ag.add_argument("--std", nargs=3, type=float, default=[0.5,0.5,0.5])
    ag.add_argument("--crop_scale", nargs=2, type=float, default=[0.8,1.0])
    ag.add_argument("--color_jitter", nargs=4, type=float, default=[0.4,0.4,0.4,0.1])

    mg = parser.add_argument_group('Misc')
    mg.add_argument("--seed", type=int, default=42)
    mg.add_argument("--fp16", action="store_true")
    mg.add_argument("--gradient_checkpointing", action="store_true")
    mg.add_argument("--resume", action="store_true")
    mg.add_argument("--gpus", type=str, default="0")
    mg.add_argument("--class_weight", type=str, default="median",
                    choices=["median","frequency","uniform"])
    return parser.parse_args()


def save_hparams_json(args):
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "hyperparameters.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[I] Saved hyperparameters to {args.output_dir}/hyperparameters.json")


def save_inference_config(args):
    cfg = {
        "mean": tuple(args.mean),
        "std": tuple(args.std),
        "resize_size": int(args.image_size * 256 / 224),
        "crop_size": args.image_size,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "inference_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[I] Saved inference config to {args.output_dir}/inference_config.json")


def compute_case_and_macro_metrics(logits: np.ndarray, labels: np.ndarray, case_ids: List):
    preds = logits.argmax(axis=-1)
    df = pd.DataFrame({"Case_ID": case_ids, "label": labels, "pred": preds})
    df["correct"] = (df["label"] == df["pred"]).astype(int)
    case_acc = df.groupby("Case_ID")["correct"].mean().mean()
    p, r, f1, _ = precision_recall_fscore_support(labels, preds,
                                                   average="macro", zero_division=0)
    return {"case_accuracy": case_acc,
            "macro_precision": p,
            "macro_recall": r,
            "macro_f1": f1}


class SurgicalDataset(Dataset):
    def __init__(self, csv_path, data_dir, is_train, image_size,
                 class_weight_method, crop_scale, color_jitter, mean, std):
        self.df = pd.read_csv(os.path.join(data_dir, csv_path))
        for c in ("Frame_Path","Phase_GT","Case_ID"):
            if c not in self.df.columns:
                raise ValueError(f"CSV 缺少列: {c}")
        self.data_dir = data_dir
        self.is_train = is_train
        self.transforms = self._build_transforms(
            is_train, image_size, crop_scale, color_jitter, mean, std
        )
        self.class_weight_method = class_weight_method

    def _build_transforms(self, is_train, image_size, crop_scale, color_jitter, mean, std):
        if is_train:
            b,c,s,h = color_jitter
            return transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=crop_scale),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            resize = int(image_size * 256 / 224)
            return transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["Frame_Path"]
        if not os.path.isabs(path):
            path = os.path.join(self.data_dir, path)
        img = Image.open(path).convert("RGB")
        x = self.transforms(img)
        y = int(row["Phase_GT"])
        cid = row["Case_ID"]
        return {"pixel_values": x, "labels": y, "case_id": cid}

    def get_class_weights(self, eps=1e-6):
        counts = torch.tensor(np.bincount(self.df["Phase_GT"].tolist()),
                              dtype=torch.float).clamp_min(eps)
        if self.class_weight_method=="uniform":
            w = torch.ones_like(counts)
        elif self.class_weight_method=="frequency":
            w = 1.0/counts
            w *= counts.numel()/w.sum()
        else:  # median
            med = counts.median()
            w = med/counts
            w /= w.mean()
        return w.tolist()

    def print_statistics(self, name, other=None):
        print(f"\n=== {name} DATASET (train={self.is_train}) ===")
        print(f" samples={len(self.df):,}, cases={self.df['Case_ID'].nunique():,}")
        if other:
            print(f" other_samples={len(other.df):,}, other_cases={other.df['Case_ID'].nunique():,}")
        dist = self.df["Phase_GT"].value_counts().sort_index()
        for phase, cnt in dist.items():
            print(f"  Phase {phase:<2}: {cnt:>6,} ({cnt/len(self.df)*100:5.1f}%)")
        if self.is_train:
            print(" class_weights:", self.get_class_weights())


class SurgicalDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage=None):
        self.train_ds = SurgicalDataset(
            self.args.train_csv, self.args.data_dir, True,
            self.args.image_size, self.args.class_weight,
            tuple(self.args.crop_scale), tuple(self.args.color_jitter),
            tuple(self.args.mean), tuple(self.args.std),
        )
        self.val_ds = SurgicalDataset(
            self.args.val_csv, self.args.data_dir, False,
            self.args.image_size, self.args.class_weight,
            tuple(self.args.crop_scale), tuple(self.args.color_jitter),
            tuple(self.args.mean), tuple(self.args.std),
        )
        self.train_ds.print_statistics("TRAIN", other=self.val_ds)
        self.val_ds.print_statistics("VAL")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.args.train_bs,
            shuffle=True, num_workers=4, pin_memory=True, drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.args.eval_bs,
            shuffle=False, num_workers=4, pin_memory=True
        )

    test_dataloader = val_dataloader


class PrintMetricsCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        if not trainer.is_global_zero: return
        print("\n>>> Validation Metrics <<<")
        for k,v in trainer.callback_metrics.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            print(f"  {k:20s}: {val:.6f}")
        print("-"*40)

    def on_test_end(self, trainer, pl_module):
        if not trainer.is_global_zero: return
        print("\n>>> Test Metrics <<<")
        for k,v in trainer.callback_metrics.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            print(f"  {k:20s}: {val:.6f}")
        print("-"*40)


class StepProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()
        self._start_time = None

    def init_train_tqdm(self):
        return tqdm(desc="Training",
                    total=self.trainer.max_epochs,
                    unit="epoch",
                    leave=True,
                    dynamic_ncols=True)

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self._start_time = time.time()

    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        cm = trainer.callback_metrics
        if "train_loss" in cm:
            items["train_loss"] = f"{cm['train_loss']:.4f}"
        if trainer.optimizers:
            lr = trainer.optimizers[0].param_groups[0]["lr"]
            items["lr"] = f"{lr:.8e}"
        gs = trainer.current_epoch + 1
        total = trainer.max_epochs
        if self._start_time and gs and total:
            elapsed = time.time() - self._start_time
            frac = min(1.0, gs/total)
            if frac > 0:
                eta = elapsed/frac - elapsed
                h, rem = divmod(int(eta), 3600)
                m, s   = divmod(rem, 60)
                items["ETA"] = f"{h:d}:{m:02d}:{s:02d}"
        return items

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = timm.create_model('resnetv2_50', width_factor=1, pretrained=True)
        self.net.head = nn.Identity()
        print(self.net)

    def forward(self, x):
        fea = self.net(x)
        return fea


class SurgicalClassifier(pl.LightningModule):
    def __init__(self, pretrained_model_path, num_labels, weight_list,
                 learning_rate, weight_decay, warmup_ratio, gradient_checkpointing):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = Encoder()
        
        ckpt = torch.load('ckpts/model.ckpt', map_location="cpu")['model']
        ckpt_encoder_only = {}
        for k, v in ckpt.items():
            if k.startswith('encoder.'):
                ckpt_encoder_only[k.replace('encoder.', '')] = v
        
        self.model.load_state_dict(ckpt_encoder_only)
        print("loading distill ckpts")
        
        # freeze all layers
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        # set bn in eval mode to disable moving average
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        
        # if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
        #     self.model.gradient_checkpointing_enable()
        
        hidden_size = 2048
        # self.classifier = nn.Linear(hidden_size, num_labels)
        self.pool_layer = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        self.criterion = F.cross_entropy
        self.weight_list = torch.tensor(weight_list, dtype=torch.float)
        self._val_outputs = []
        self._test_outputs = []

    def forward(self, x):
        return self.model(x).logits

    def training_step(self, batch, batch_idx):
        x, y = batch["pixel_values"], batch["labels"]
        
        feat = self.model(x)
        # import ipdb; ipdb.set_trace()
        feat = self.pool_layer(feat).flatten(1)
        
        logits = self.classifier(feat)
        
        loss = self.criterion(logits, y, weight=self.weight_list.to(self.device))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, cid = batch["pixel_values"], batch["labels"], batch["case_id"]
        
        feat = self.model(x)
        feat = self.pool_layer(feat).flatten(1)
        logits = self.classifier(feat)
        
        loss = self.criterion(logits, y, weight=self.weight_list.to(self.device))
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self._val_outputs.append({"logits":logits.cpu(), "labels":y.cpu(), "case_ids":cid})

    def on_validation_epoch_end(self):
        if not self._val_outputs: return
        logits, labels, cids = self._gather(self._val_outputs)
        m = compute_case_and_macro_metrics(logits, labels, cids)
        for k,v in m.items():
            self.log(k, v, on_epoch=True, sync_dist=True)
        self._val_outputs.clear()

    def test_step(self, batch, batch_idx):
        x,y,cid = batch["pixel_values"], batch["labels"], batch["case_id"]
        logits = self(x)
        loss = self.criterion(logits, y, weight=self.weight_list.to(self.device))
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self._test_outputs.append({"logits":logits.cpu(), "labels":y.cpu(), "case_ids":cid})

    def on_test_epoch_end(self):
        if not self._test_outputs: return
        logits, labels, cids = self._gather(self._test_outputs)
        m = compute_case_and_macro_metrics(logits, labels, cids)
        for k,v in m.items():
            self.log(k, v, on_epoch=True, sync_dist=True)
        self._test_outputs.clear()

    def _gather(self, outputs):
        all_l, all_y, all_c = [], [], []
        for o in outputs:
            all_l.append(o["logits"])
            all_y.append(o["labels"])
            c = o["case_ids"]
            if isinstance(c, torch.Tensor): c = c.tolist()
            all_c.extend(c if isinstance(c,(list,tuple)) else [c])
        return torch.cat(all_l,0).numpy(), torch.cat(all_y,0).numpy(), all_c

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(),
                                 lr=self.hparams.learning_rate,
                                 weight_decay=self.hparams.weight_decay)
        # 计算总步数用于学习率调度
        total_batches = len(self.trainer.datamodule.train_dataloader())
        total = total_batches * self.trainer.max_epochs
        
        if self.hparams.warmup_ratio>0:
            w = int(total*self.hparams.warmup_ratio)
            sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=w, num_training_steps=total)
            return [opt], [{"scheduler":sched,"interval":"step"}]
        else:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
            return [opt], [{"scheduler":sched,"interval":"step"}]


def setup_environment(args):
    set_seed(args.seed)
    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus
    return len(args.gpus.split(','))


def setup_callbacks_and_logger(args):
    ckpt_cb = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="best-epoch{epoch:03d}-f1{macro_f1:.4f}",
        monitor="macro_f1", mode="max", save_top_k=1, save_last=True, verbose=True
    )
    cbs = [
        ckpt_cb,
        EarlyStopping(monitor="macro_f1", mode="max", patience=3, min_delta=1e-3, verbose=True),
        LearningRateMonitor(logging_interval="step"),
        PrintMetricsCallback(),
        StepProgressBar(),
    ]
    logger = TensorBoardLogger(args.output_dir, name="logs")
    return cbs, ckpt_cb, logger


def setup_trainer(args, num_gpus, callbacks, logger):
    strat = DDPStrategy(find_unused_parameters=True) if num_gpus>1 else "auto"
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=num_gpus,
        strategy=strat,
        precision=16 if args.fp16 else 32,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_val_batches,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        check_val_every_n_epoch=args.eval_epochs,
        num_sanity_val_steps=0,
    )
    print(f"[I] Trainer: max_epochs={args.max_epochs}, min_epochs={args.min_epochs}")
    return trainer


def get_checkpoint_path(args):
    if args.resume:
        last = os.path.join(args.output_dir, "last.ckpt")
        if os.path.exists(last):
            print(f"[I] Resuming from {last}")
            return last
    return None


def print_config(args, num_gpus, ckpt_path):
    print("\n" + "="*60)
    print("CONFIG:")
    print(f" GPUs           : {args.gpus} ({num_gpus})")
    print(f" max_epochs     : {args.max_epochs}")
    print(f" min_epochs     : {args.min_epochs}")
    print(f" eval_epochs    : {args.eval_epochs}")
    print(f" limit_batches  : train={args.limit_train_batches}, val/test={args.limit_val_batches}")
    print(f" lr/wd/warmup   : {args.lr:.2e}/{args.weight_decay}/{args.warmup_ratio}")
    print(f" batch_size     : train={args.train_bs}, eval={args.eval_bs}")
    print(f" fp16           : {args.fp16}")
    print(f" grad_ckpt      : {args.gradient_checkpointing}")
    print(f" output_dir     : {args.output_dir}")
    if ckpt_path: print(f" resume_ckpt    : {ckpt_path}")
    print("="*60 + "\n")


def main():
    args = parse_args()
    num_gpus = setup_environment(args)
    save_hparams_json(args)
    save_inference_config(args)

    dm = SurgicalDataModule(args)
    dm.setup("fit")

    model = SurgicalClassifier(
        pretrained_model_path=args.pretrained_model_path,
        num_labels=len(set(dm.train_ds.df["Phase_GT"].tolist())),
        weight_list=dm.train_ds.get_class_weights(),
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    callbacks, ckpt_cb, logger = setup_callbacks_and_logger(args)
    trainer = setup_trainer(args, num_gpus, callbacks, logger)

    ckpt_path = get_checkpoint_path(args)
    print_config(args, num_gpus, ckpt_path)

    # 训练
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)

    # 测试最佳模型
    best = ckpt_cb.best_model_path
    print(f"\n[I] Best checkpoint: {best}")
    trainer.test(ckpt_path=best, datamodule=dm)

    # 保存最终 weights
    best_model = SurgicalClassifier.load_from_checkpoint(best)
    outp = os.path.join(args.output_dir, "final_model")
    best_model.model.save_pretrained(outp)
    print(f"[I] Saved pretrained weights to {outp}")


if __name__ == "__main__":
    main()

