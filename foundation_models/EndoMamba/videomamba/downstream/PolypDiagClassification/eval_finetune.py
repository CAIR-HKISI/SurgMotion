import argparse
import json
import os
from cv2 import line
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from torch import nn
from tqdm import tqdm
import numpy as np
import random

from sklearn.metrics import f1_score

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from timm.models import create_model
from datasets import UCF101, HMDB51, Kinetics
from models import get_vit_base_patch16_224, get_aux_token_vit, SwinTransformer3D
from video_sm.models.videomamba import videomamba_small
from models.endomamba_classification import endomamba_small
from video_sm.models.videomae_v2 import vit_small_patch16_224
from utils import utils
from utils.meters import TestMeter
from utils.parser import load_config


def eval_finetune(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    args.output_dir = args.output_dir + args.arch + '_s' + str(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(vars(args), open(f"{args.output_dir}/config.json", "w"), indent=4)

    # ============ preparing data ... ============ 
    config = load_config(args)
    config.TEST.NUM_SPATIAL_CROPS = 1
    config.DATA.PATH_TO_DATA_DIR = args.data_path + 'splits'
    config.DATA.PATH_PREFIX = args.data_path + 'videos'
    config.DATA.USE_FLOW = False
    if args.dataset == "ucf101":
        dataset_train = UCF101(cfg=config, mode="train", num_retries=10)
        dataset_val = UCF101(cfg=config, mode="val", num_retries=10)
        config.TEST.NUM_SPATIAL_CROPS = 3
    elif args.dataset == "hmdb51":
        dataset_train = HMDB51(cfg=config, mode="train", num_retries=10)
        dataset_val = HMDB51(cfg=config, mode="val", num_retries=10)
        config.TEST.NUM_SPATIAL_CROPS = 3
    elif args.dataset == "kinetics400":
        dataset_train = Kinetics(cfg=config, mode="train", num_retries=10)
        dataset_val = Kinetics(cfg=config, mode="val", num_retries=10)
        config.TEST.NUM_SPATIAL_CROPS = 3
    else:
        raise NotImplementedError(f"invalid dataset: {args.dataset}")

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=train_sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False
    )

    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============ 
    if config.DATA.USE_FLOW or config.MODEL.TWO_TOKEN:
        model = get_aux_token_vit(cfg=config, no_head=True)
        model_embed_dim = 2 * model.embed_dim
    else:
        if args.arch == "vit_base":
            model = get_vit_base_patch16_224(cfg=config, no_head=True)
            model_embed_dim = model.embed_dim
        elif args.arch == "swin":
            model = SwinTransformer3D(depths=[2, 2, 18, 2], embed_dim=128, num_heads=[4, 8, 16, 32])
            model_embed_dim = 1024
        # elif args.arch == "endomamba":
        #     model = get_endomamba_small(cfg=config, no_head=False, pretrained=not args.scratch, 
        #                                 num_classes=args.num_labels)  
        #     model_embed_dim = 384  # Set the embed dimension to 384 as per endomamba configuration
        else:
            if args.arch == "videomaev2":
                args.arch = "vit_small_patch16_224"
            model = create_model(args.arch, no_head=False, pretrained=not args.scratch, num_classes=args.num_labels)
            model_embed_dim = model.embed_dim  # Set the embed dimension to 384 as per endomamba configuration
        # else:
        #     raise Exception(f"invalid model: {args.arch}")

    model.cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")

    # ============ Remove Linear Classifier for "endomamba" case ... ============ 
    if "endomamba" in args.arch or "videomamba" in args.arch or "vit_small_patch16_224" in args.arch:
        linear_classifier = None  # Directly set to None for endomamba since it includes its own head
    else:
        linear_classifier = LinearClassifier(model_embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens)),
                                             num_labels=args.num_labels)
        linear_classifier = linear_classifier.cuda()
        linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    # If test mode, load weights and perform testing
    if args.test:
        model.eval()
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        print('best_f1', state_dict["best_f1"])
        model.load_state_dict(state_dict["backbone_state_dict"])
        test_stats, f1 = validate_network(val_loader, model, linear_classifier, args.n_last_blocks,
                                          args.avgpool_patchtokens)
        print(f"F1 score of the network on the {len(dataset_val)} test images: {f1 * 100:.1f}%")
        exit(0)

    scaled_lr = args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.

    # set optimizer
    optimizer = torch.optim.SGD(
        [{'params': model.parameters(), 'lr': scaled_lr}],
        momentum=0.9,
        weight_decay=0,  # we do not apply weight decay
    )
    if linear_classifier is not None:
        optimizer.add_param_group(
            {'params': linear_classifier.parameters(), 'lr': scaled_lr}
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0., "best_f1": 0.}
    # utils.restart_from_checkpoint(
    #     os.path.join(args.output_dir, "checkpoint.pth.tar"),
    #     run_variables=to_restore,
    #     state_dict=linear_classifier,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    # )
    start_epoch = to_restore["epoch"]
    best_f1 = to_restore["best_acc"]

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        train_stats = train(args, model, linear_classifier, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats, f1 = validate_network(val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)
            print(f"F1 score at epoch {epoch} of the network on the {len(dataset_val)} test images: {f1 * 100:.1f}%")
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}

            if f1 > best_f1 and utils.is_main_process():
                with (Path(args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                best_f1 = max(best_f1, f1)
                save_dict = {
                    "epoch": epoch + 1,
                    "backbone_state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_f1": best_f1,
                }
                if linear_classifier is not None:
                    save_dict["linear_classifier"] = linear_classifier.state_dict()
                torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar"))

            best_f1 = max(best_f1, f1)
            print(f'Max F1 score so far: {best_f1 * 100:.1f}%')

# Training function
def train(args, model, linear_classifier, optimizer, loader, epoch, n, avgpool):
    model.train()
    if linear_classifier is not None:
        linear_classifier.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, target, sample_idx, meta) in metric_logger.log_every(loader, 20, header):
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(inp)
        if linear_classifier is not None:
            output = linear_classifier(output)

        # print("Output shape:", output.shape)
        # print("Target shape:", target.shape)
        loss = nn.CrossEntropyLoss()(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# Validation function
@torch.no_grad()
def validate_network(val_loader, model, linear_classifier, n, avgpool):
    model.eval()
    if linear_classifier is not None:
        linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    all_target = []
    all_output = []
    for (inp, target, sample_idx, meta) in metric_logger.log_every(val_loader, 20, header):
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            output = model(inp)
        if linear_classifier is not None:
            output = linear_classifier(output)
        loss = nn.CrossEntropyLoss()(output, target)

        acc1, = utils.accuracy(output, target, topk=(1,))
        all_target.extend(target.detach().cpu().numpy())
        all_output.extend(np.argmax(output.detach().cpu().numpy(), axis=1))

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    f1 = f1_score(all_target, all_output)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, f1


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=1, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
                        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='endomamba_small', type=str)
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--lc_pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=4, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/data/tqy/PolypDiag/', type=str)
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default="/data/tqy/out/Classification/MIX12", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=2, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--dataset', default="ucf101", help='Dataset: ucf101 / hmdb51')
    parser.add_argument('--use_flow', default=False, type=utils.bool_flag, help="use flow teacher")

    parser.add_argument('--scratch', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--pretrained_model_weights', default='/mnt/tqy/out/Classification/EndoMamba_NF32_s3/checkpoint.pth.tar', type=str, help='pre-trained weights')
    parser.add_argument('--seed', type=int,
                        default=3, help='random seed')
    
    # config file
    parser.add_argument("--cfg", dest="cfg_file", help="Path to the config file", type=str,
                        default="./models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml")
    parser.add_argument("--opts", help="See utils/defaults.py for all options", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    eval_finetune(args)
