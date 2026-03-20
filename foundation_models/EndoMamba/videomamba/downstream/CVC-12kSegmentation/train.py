import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.endomamba_seg_modeling import endomambaseg_small
# from networks.videomamba_seg_modeling import videomambaseg_small
# from networks.videomae_seg_modeling import videomaeseg_small
from timm.models import create_model
from trainer import trainer_synapse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str,
                        default='', help='experiment_name')
    parser.add_argument('--root_path', type=str,
                        default='/mnt/tqy/CVC-ClinicVideoDB/CVC-ClinicVideoDB/', help='root dir for data')
    parser.add_argument('--dataset', type=str,
                        default='CVC', help='experiment_name')
    parser.add_argument('--list_dir', type=str,
                        default='./lists/lists_Synapse', help='list dir')
    parser.add_argument('--out_dir', type=str,
                        default='', help='save checkpoint dir')
    parser.add_argument('--num_classes', type=int,
                        default=2, help='output channel of network')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int,
                        default=150, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int,
                        default=1, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--gpu', default="1", help='total gpu')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=1e-4,
                        help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')
    parser.add_argument('--seed', type=int,
                        default=9041, help='random seed')
    parser.add_argument('--n_skip', type=int,
                        default=0, help='using number of skip-connect, default is num')
    parser.add_argument('--model', type=str,
                        default='endomambaseg_small', help='select one vit model')
    parser.add_argument('--wandb', type=bool,
                        default=False, help='using wandb logger')
    parser.add_argument('--test', action='store_true', help='test the pretrained model')
    parser.add_argument('--pretrained_model_weights', 
                        default="/mnt/tqy/out/segmentation/EndoMamba_NF8_s1/best_model.pth",
                        help='test pretrained model weight')
    args = parser.parse_args()
    # print(args)
    # exit(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'CVC': {
            'root_path': args.root_path,
            'list_dir': './lists/lists_Synapse',
            'num_classes': 2,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    # args.exp = 'EndoMamba (MIX12 w. teacher)'
    snapshot_path = args.out_dir + args.exp
    snapshot_path = snapshot_path + '_s'+str(args.seed) + '_skip' + str(args.n_skip)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        
    # net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    
    # net = endomambaseg_small(pretrained=True, num_classes=args.num_classes, n_skip=args.n_skip).cuda()
    net = create_model(
        args.model,
        pretrained=True,
        num_classes=args.num_classes,
        n_skip=args.n_skip
    ).cuda()
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')

    trainer = {'CVC': trainer_synapse,}
    trainer[dataset_name](args, net, snapshot_path)
