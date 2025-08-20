#!/usr/bin/env python
# coding: utf-8

"""
Surgical Phase Recognition (Video) with VJEPA2 & PyTorch Lightning

-- 使用 torch.hub 加载 VJEPA2 encoder + 手动分类头
-- Dataset 读取当前帧及前序 K-1 帧组成视频片段；不足时前面 pad 全零帧
-- LightningModule 调用 VJEPA2 encoder + 分类头
-- 按 epoch 而非 step 来控制训练轮数和验证频率
"""

import os
import json
import argparse
import time
from typing import Tuple, List, Optional

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
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    get_cosine_schedule_with_warmup,
    set_seed
)
from tqdm.auto import tqdm
from predictor import AttentiveClassifier

# =============================================================================
# Argument Parsing
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser("Video-based Surgical Phase Recognition with VJEPA2")
    # data
    dg = parser.add_argument_group('Data')
    dg.add_argument("--data_dir", required=True)
    dg.add_argument("--train_csv", required=True)
    dg.add_argument("--val_csv", required=True)
    dg.add_argument("--output_dir", default="./output")
    dg.add_argument("--limit_train_batches", type=float, default=1.0)
    dg.add_argument("--limit_val_batches", type=float, default=1.0)

    # model & training
    tg = parser.add_argument_group('Training')
    tg.add_argument("--num_frames", type=int, default=16,
        help="视频片段帧数 K (当前帧 + 前序 K-1 帧), VJEPA2 推荐使用 16")
    tg.add_argument("--image_size", type=int, default=224)
    tg.add_argument("--max_epochs", type=int, default=10,
        help="训练总轮数 (epoch)")
    tg.add_argument("--train_bs", type=int, default=4)
    tg.add_argument("--eval_bs", type=int, default=4)
    tg.add_argument("--accumulate_grad_batches", type=int, default=1)
    tg.add_argument("--lr", type=float, default=2e-5)
    tg.add_argument("--weight_decay", type=float, default=0.0)
    tg.add_argument("--warmup_steps", type=int, default=500)
    tg.add_argument("--freeze_encoder", action="store_true",
        help="是否冻结 VJEPA2 encoder 参数")

    # augmentation
    ag = parser.add_argument_group('Augmentation')
    ag.add_argument("--mean", nargs=3, type=float, default=[0.485, 0.456, 0.406])
    ag.add_argument("--std", nargs=3, type=float, default=[0.229, 0.224, 0.225])
    ag.add_argument("--crop_scale", nargs=2, type=float, default=[0.8,1.0])
    ag.add_argument("--color_jitter", nargs=4, type=float, default=[0.4,0.4,0.4,0.1])

    # misc
    mg = parser.add_argument_group('Misc')
    mg.add_argument("--seed", type=int, default=42)
    mg.add_argument("--fp16", action="store_true")
    mg.add_argument("--gradient_checkpointing", action="store_true")
    mg.add_argument("--resume", action="store_true")
    mg.add_argument("--gpus", type=str, default="0")
    mg.add_argument("--class_weight", choices=["median","frequency","uniform"],
                    default="median")
    return parser.parse_args()

# =============================================================================
# Utils
# =============================================================================
def save_json(o, path):
    with open(path, "w") as f:
        json.dump(o, f, indent=2)
    print(f"[I] Saved {path}")

def compute_metrics(logits: np.ndarray, labels: np.ndarray, case_ids: List):
    preds = logits.argmax(-1)
    df = pd.DataFrame({"case":case_ids,"label":labels,"pred":preds})
    df["correct"] = (df.label==df.pred).astype(int)
    case_acc = df.groupby("case").correct.mean().mean()
    p,r,f1,_ = precision_recall_fscore_support(labels,preds,average="macro", zero_division=0)
    return {"case_accuracy":case_acc,"macro_precision":p,"macro_recall":r,"macro_f1":f1}

# =============================================================================
# Dataset
# =============================================================================
class SurgicalVideoDataset(Dataset):
    def __init__(self,
        csv_path: str, data_dir: str,
        is_train: bool,
        num_frames: int,
        image_size: int,
        class_weight_method: str,
        crop_scale: Tuple[float,float],
        color_jitter: Tuple[float,float,float,float],
        mean: Tuple[float,float,float],
        std: Tuple[float,float,float],
    ):
        self.image_size = image_size
        self.num_frames = num_frames
        self.is_train = is_train
        self.data_dir = data_dir
        self.class_weight_method = class_weight_method

        df = pd.read_csv(os.path.join(data_dir, csv_path))
        df.sort_values(["Case_ID","Frame_Path"], inplace=True)
        df = df.reset_index(drop=True)
        self.df = df

        # build case -> idx list, idx -> position
        self.case2idx = {}
        self.idx2pos = {}
        for case, group in df.groupby("Case_ID").groups.items():
            lst = list(group)
            self.case2idx[case] = lst
            for pos, idx in enumerate(lst):
                self.idx2pos[idx] = pos

        # transforms per frame
        self.transforms = self._build_transforms(is_train, image_size,
                                                 crop_scale, color_jitter,
                                                 mean, std)

    def _build_transforms(self, is_train, image_size, crop_scale,
                          color_jitter, mean, std):
        if is_train:
            b,c,s,h = color_jitter
            return transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=crop_scale),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=b,
                                       contrast=c,
                                       saturation=s,
                                       hue=h),
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
        case = row["Case_ID"]
        pos = self.idx2pos[idx]
        idx_list = self.case2idx[case]
        start = pos - self.num_frames + 1
        frames = []

        # pad zero frames if start < 0
        for i in range(start, pos+1):
            if i < 0:
                frames.append(torch.zeros(3, self.image_size, self.image_size))
            else:
                vid_idx = idx_list[i]
                path = self.df.loc[vid_idx, "Frame_Path"]
                if not os.path.isabs(path):
                    path = os.path.join(self.data_dir, path)
                img = Image.open(path).convert("RGB")
                frames.append(self.transforms(img))

        # stack to (T, C, H, W) - VJEPA2 expects this format
        video = torch.stack(frames, dim=0)
        label = int(row["Phase_GT"])
        return {"pixel_values": video,
                "labels": label,
                "case_id": case}

    def get_class_weights(self, eps=1e-6):
        counts = torch.tensor(np.bincount(self.df["Phase_GT"]),
                              dtype=torch.float).clamp_min(eps)
        if self.class_weight_method == "uniform":
            w = torch.ones_like(counts)
        elif self.class_weight_method == "frequency":
            w = 1.0 / counts
            w *= counts.numel() / w.sum()
        else:  # median
            med = counts.median()
            w = med / counts
            w /= w.mean()
        return w.tolist()

    @rank_zero_only
    def print_statistics(self, name, other=None):
        print(f"\n=== {name} DATASET (train={self.is_train}) ===")
        print(f" samples={len(self.df):,}, cases={self.df['Case_ID'].nunique():,}")
        if other:
            print(f" other: samples={len(other.df):,}, cases={other.df['Case_ID'].nunique():,}")
        dist = self.df["Phase_GT"].value_counts().sort_index()
        for phase, cnt in dist.items():
            print(f"  Phase {phase}: {cnt:,} ({cnt/len(self.df)*100:.1f}%)")

# =============================================================================
# DataModule
# =============================================================================
class SurgicalVideoDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage: Optional[str] = None):
        self.train_ds = SurgicalVideoDataset(
            self.args.train_csv, self.args.data_dir, True,
            self.args.num_frames, self.args.image_size,
            self.args.class_weight,
            tuple(self.args.crop_scale),
            tuple(self.args.color_jitter),
            tuple(self.args.mean),
            tuple(self.args.std),
        )
        self.val_ds = SurgicalVideoDataset(
            self.args.val_csv, self.args.data_dir, False,
            self.args.num_frames, self.args.image_size,
            self.args.class_weight,
            tuple(self.args.crop_scale),
            tuple(self.args.color_jitter),
            tuple(self.args.mean),
            tuple(self.args.std),
        )
        self.train_ds.print_statistics("TRAIN", other=self.val_ds)
        self.val_ds.print_statistics("VAL")

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.args.train_bs,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.args.eval_bs,
                          shuffle=False,
                          num_workers=4,
                          pin_memory=True)
    
    def test_dataloader(self):
        # 复用验证集作为测试集
        return self.val_dataloader()


# =============================================================================
# Progress Bar
# =============================================================================
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

# =============================================================================
# Callbacks
# =============================================================================
class PrintMetricsCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        print("\n>>> Validation Metrics <<<")
        for k, v in trainer.callback_metrics.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            print(f"  {k:20s}: {val:.6f}")
        print("-" * 40)
    

    def on_test_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        print("\n>>> Test Metrics <<<")
        for k, v in trainer.callback_metrics.items():
            if k.startswith('test_'):
                val = v.item() if isinstance(v, torch.Tensor) else v
                print(f"  {k:20s}: {val:.6f}")
        print("-" * 40)

# =============================================================================
# LightningModule
# =============================================================================

class SurgicalVideoClassifier(pl.LightningModule):
    def __init__(self, args, num_labels, weight_list):
        super().__init__()
        self.save_hyperparameters(args)

        # Load VJEPA2 encoder
        print("[I] Loading VJEPA2 encoder...")
        self.encoder = torch.hub.load('facebookresearch/vjepa2', 
                                      'vjepa2_vit_large')[0]
        
        # Get the hidden size from the encoder
        # VJEPA2 ViT-Large typically has hidden_size of 1024
        hidden_size = 1024  # For ViT-Large
        
        # Freeze encoder if specified
        if args.freeze_encoder:
            print("[I] Freezing VJEPA2 encoder parameters...")
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Enable gradient checkpointing if specified
        if args.gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

        # Classifier head with dropout
        self.dropout = nn.Dropout(0.1)
        # self.classifier = nn.Linear(hidden_size, num_labels)
        self.classifier = AttentiveClassifier(
            embed_dim=hidden_size,
            num_heads=16,
            mlp_ratio=4.0,
            depth=1,
            norm_layer=nn.LayerNorm,
            num_classes=num_labels,
        )

        # Loss & weights
        self.criterion = F.cross_entropy
        self.weight = torch.tensor(weight_list, dtype=torch.float)

        # Collect validation outputs
        self._val_outputs = []
        # Collect test outputs
        self._test_outputs = []

    def forward(self, pixel_values):
        # pixel_values: (B, T, C, H, W)
        # VJEPA2 expects (B, T, C, H, W) format
        
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        
        # Extract features using VJEPA2 encoder
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):  # VJEPA2 may not support mixed precision well
            features = self.encoder(pixel_values)
        
        # features shape: (B, T, D) where D is the feature dimension
        # Pool over temporal dimension - combine mean and max pooling
        # features_mean = features.mean(dim=1)  # (B, D)
        # features_max = features.amax(dim=1)   # (B, D)
        # features_pooled = 0.5 * features_mean + 0.5 * features_max
        
        # Apply dropout and classifier
        # features_pooled = self.dropout(features_pooled)
        logits = self.classifier(features)
        
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch["pixel_values"], batch["labels"]
        # x already in (B, T, C, H, W) format from dataset
        logits = self(x)
        loss = self.criterion(logits, y, weight=self.weight.to(self.device))
        self.log("train_loss", loss,
                 on_epoch=False, on_step=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, cid = batch["pixel_values"], batch["labels"], batch["case_id"]
        # x already in (B, T, C, H, W) format from dataset
        logits = self(x)
        loss = self.criterion(logits, y, weight=self.weight.to(self.device))
        self.log("val_loss", loss, on_epoch=False, on_step=True, prog_bar=True, sync_dist=True)
        self._val_outputs.append({
            "logits": logits.detach().cpu(),
            "labels": y.detach().cpu(),
            "case_ids": cid
        })
    
    def test_step(self, batch, batch_idx):
        x, y, cid = batch["pixel_values"], batch["labels"], batch["case_id"]
        # x already in (B, T, C, H, W) format from dataset
        logits = self(x)
        loss = self.criterion(logits, y, weight=self.weight.to(self.device))
        self.log("test_loss", loss, on_epoch=False, on_step=True, prog_bar=True, sync_dist=True)
        self._test_outputs.append({
            "logits": logits.detach().cpu(),
            "labels": y.detach().cpu(),
            "case_ids": cid
        })

    def on_validation_epoch_end(self):
        if not self._val_outputs:
            return
        logits, labels, cids = self._gather(self._val_outputs)
        metrics = compute_metrics(logits, labels, cids)
        for k, v in metrics.items():
            self.log(k, v, on_epoch=True, sync_dist=True)
        self._val_outputs.clear()
    
    def on_test_epoch_end(self):
        if not self._test_outputs:
            return
        logits, labels, cids = self._gather(self._test_outputs)
        metrics = compute_metrics(logits, labels, cids)
        for k, v in metrics.items():
            # 为测试指标添加 test_ 前缀
            self.log(f"test_{k}", v, on_epoch=True, sync_dist=True)
        self._test_outputs.clear()

    def _gather(self, outs):
        ls, ys, cs = [], [], []
        for o in outs:
            ls.append(o["logits"])
            ys.append(o["labels"])
            # Handle case_ids which might be list of strings
            if isinstance(o["case_ids"], list):
                cs.extend(o["case_ids"])
            else:
                cs.extend(o["case_ids"].tolist() if hasattr(o["case_ids"], 'tolist') else list(o["case_ids"]))
        return (torch.cat(ls).numpy(),
                torch.cat(ys).numpy(),
                cs)

    def configure_optimizers(self):
        # Different learning rates for encoder and classifier
        params = []
        if not self.hparams.freeze_encoder:
            params.append({
                'params': self.encoder.parameters(),
                'lr': self.hparams.lr * 0.1  # Lower lr for pretrained encoder
            })
        params.append({
            'params': list(self.dropout.parameters()) + list(self.classifier.parameters()),
            'lr': self.hparams.lr
        })
        
        optimizer = torch.optim.AdamW(
            params,
            weight_decay=self.hparams.weight_decay
        )
        
        # Compute total training steps for scheduler
        train_loader = self.trainer.datamodule.train_dataloader()
        steps_per_epoch = len(train_loader) // self.trainer.accumulate_grad_batches
        total_steps = steps_per_epoch * self.trainer.max_epochs
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=total_steps
        )
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

# =============================================================================
# Training Script
# =============================================================================
def main():
    args = parse_args()
    num_gpus = len(args.gpus.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    set_seed(args.seed)
    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)

    save_json(vars(args), os.path.join(args.output_dir, "hyperparameters.json"))
    save_json({
        "mean": tuple(args.mean), "std": tuple(args.std),
        "resize_size": int(args.image_size * 256 / 224),
        "crop_size": args.image_size,
        "num_frames": args.num_frames,
        "model": "vjepa2_vit_large"
    }, os.path.join(args.output_dir, "inference_config.json"))

    dm = SurgicalVideoDataModule(args)
    dm.setup("fit")
    weight_list = dm.train_ds.get_class_weights()

    model = SurgicalVideoClassifier(
        args,
        num_labels=len(set(dm.train_ds.df["Phase_GT"])),
        weight_list=weight_list
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="epoch{epoch:02d}-f1{macro_f1:.4f}",
        monitor="macro_f1", mode="max",
        save_top_k=3, save_last=True, verbose=True
    )
    callbacks = [
        ckpt_cb,
        EarlyStopping(monitor="macro_f1", mode="max", patience=3, min_delta=1e-3, verbose=True),
        LearningRateMonitor("step"),
        PrintMetricsCallback(),
        StepProgressBar(),
    ]
    logger = TensorBoardLogger(args.output_dir, name="logs")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=num_gpus,
        strategy=DDPStrategy(find_unused_parameters=True) if num_gpus > 1 else "auto",
        precision=16 if args.fp16 else 32,
        callbacks=callbacks,
        logger=logger,
        max_epochs=args.max_epochs,
        min_epochs=int(args.max_epochs*0.9),
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        check_val_every_n_epoch=1,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    ckpt_path = os.path.join(args.output_dir, "last.ckpt") if args.resume else None
    trainer.fit(model, dm, ckpt_path=ckpt_path)

    best_ckpt = ckpt_cb.best_model_path
    print(f"[I] Best checkpoint: {best_ckpt}")
    trainer.test(ckpt_path=best_ckpt, datamodule=dm)

if __name__ == "__main__":
    main()