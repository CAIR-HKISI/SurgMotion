import argparse
# from networkx import project
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import json
import os
from pathlib import Path
import sys, time
from torch.utils.data import DataLoader
from timm.utils import accuracy

sys.path.append("/home/tqy/VideoMamba/videomamba/downstream/SurgicalPhase/Surgformer")
sys.path.append("/home/tqy/VideoMamba/videomamba/")

from timm.models import create_model

from downstream_phase.datasets_phase import build_smart_test_dataset as build_dataset
import utils

from model.endomamba import endomamba_small
from mamba_ssm.utils.generation import InferenceParams


def get_args():
    parser = argparse.ArgumentParser(
        "SurgVideoMAE fine-tuning and evaluation script for video phase recognition",
        add_help=False,
    )
 
    # Model parameters
    parser.add_argument(
        "--model",
        default="endomamba_small",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument("--smart_test", default=False, action="store_true")
    parser.add_argument(
        "--pretrained_path",
        default="/mnt/tqy/out/AutoLaparo/endomamba_small_AutoLaparo_5e-05_0.75_online_all_frame_frame32_Fixed_Stride_1_5e-05_train5/checkpoint-best.pth",
        type=str,
        metavar="Parameters",
        help="Name of parameters to load",
    )
    parser.add_argument("--input_size", default=224, type=int, help="videos input size")

    # Evaluation parameters
    parser.add_argument("--crop_pct", type=float, default=None)
    parser.add_argument("--short_side_size", type=int, default=224)
    
    # Dataset parameters
    parser.add_argument(
        "--data_path",
        default="/mnt/tqy/AutoLaparo/AutoLaparo_Task1",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--eval_video_id",
        default=["15", "16", "17", "18", "19", "20", "21"], 
        nargs="+",  
        type=str,  
        help="sequences for evaluation (e.g., --eval_video_id 15 16 17)"
    )
    parser.add_argument(
        "--nb_classes", default=7, type=int, help="number of the classification types"
    )
    parser.add_argument(
        "--imagenet_default_mean_and_std", default=True, action="store_true"
    )

    parser.add_argument(
        "--data_strategy", type=str, default="online"
    )  # online/offline
    parser.add_argument(
        "--output_mode", type=str, default="all_frame"
    )  # key_frame/all_frame
    # parser.add_argument("--cut_black", action="store_true")  # True/False
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument(
        "--sampling_rate", type=int, default=4
    )  # 0表示指数级间隔，-1表示随机间隔设置, -2表示递增间隔
    parser.add_argument(
        "--data_set",
        default="AutoLaparo",
        choices=["Cholec80", "AutoLaparo"],
        type=str,
        help="dataset",
    )
    parser.add_argument(
        "--data_fps",
        default="1fps",
        choices=["", "5fps", "1fps"],
        type=str,
        help="dataset",
    )
    parser.add_argument("--cut_black", default=True, action="store_true")  # True/False
    parser.add_argument(
        "--output_dir",
        default="./result/",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--device", default="cuda:2", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--num_workers", default=2, type=int)
    
    parser.add_argument("--train_seq_len", required=True, type=int)
    
    parser.add_argument("--only_cls_token", action="store_true", default=False)
    
    known_args, _ = parser.parse_known_args()

    return parser.parse_args()


def test_per_sequence(data_loader, model, device="cuda:1", train_seq_len=16, smart_test=True, txt_path='.result.txt'):
    """
    smart test would always save the result of last <train_seq_len // 2> predictions of each <train_seq_len> to ensure enough memory for references.
    """
    final_result = []
    
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    
    infer_params = InferenceParams(max_seqlen=64, max_batch_size=1)
    frame_counter = 0
        
    current_batch = 0
    
    with open(txt_path, 'w') as f:
        f.write('')
    
    with torch.no_grad():
        t1 = time.time()
        for batch in metric_logger.log_every(data_loader, 10, header):
            time_loader = time.time() - t1
            videos = batch[0]
            target = batch[1]
            ids = batch[2]
            flags = batch[3]
            
            videos= videos.to(device)
            target = target.to(device)
                
            videos = torch.concatenate([videos, videos], dim=0)

            t1 = time.time()
            if frame_counter % train_seq_len == 0 and frame_counter > train_seq_len:
                if smart_test:
                    current_batch = 1  # switch final output to batch 1
                for layer_idx in infer_params.key_value_memory_dict.keys():
                    params = infer_params.key_value_memory_dict[layer_idx]
                    updated_params = []  
                    for i in range(len(params)):
                        element = params[i]
                        element[0] = torch.zeros_like(element[0]).to(element.device)  # clear batch 0 states
                        updated_params.append(element) 
                    infer_params.key_value_memory_dict[layer_idx] = tuple(updated_params)
                
            elif smart_test and frame_counter % train_seq_len == train_seq_len // 2 and frame_counter > train_seq_len:
                current_batch  = 0 # switch final output to batch 0
                for layer_idx in infer_params.key_value_memory_dict.keys():
                    params = infer_params.key_value_memory_dict[layer_idx]
                    updated_params = []
                    for i in range(len(params)):
                        element = params[i]
                        element[1] = torch.zeros_like(element[1]).to(element.device)  # clear batch 1 states
                        updated_params.append(element) 
                    infer_params.key_value_memory_dict[layer_idx] = tuple(updated_params)
            time_mem = time.time() - t1
            
            t1 = time.time()
            if model.return_last_state:
                output, infer_params = model(videos, infer_params)  
            else:
                output = model(videos)  
            output = output[current_batch]
            time_com = time.time() - t1
            # print("Used time:", time.time() - t1)
            # print(output.size(), target.size())
            if model.return_last_state:
                output = output.view(-1, output.size(-1))  # take only the last frame output
                target = target.view(-1)
            else:
                output = output.view(-1, output.size(-1))[-1].unsqueeze(0)  # take only the last frame output
                target = target[:, -1]
            # print(output.size(), target.size())
            loss = criterion(output, target)

            # unique_id, video_id, frame_id = ids[0].strip().split('_')
            # if flags[i]:
            #     if target[i] == 0:
            #         output.data[i] = torch.tensor([1, 0, 0, 0, 0, 0, 0])
            #     elif target[i] == 1:
            #         output.data[i] = torch.tensor([0, 1, 0, 0, 0, 0, 0])
            #     elif target[i] == 2:
            #         output.data[i] = torch.tensor([0, 0, 1, 0, 0, 0, 0])
            
            _, pred = output.topk(1, 1, True, True)
                        
            string = "{} {} {} {}\n".format(
                str(frame_counter),
                str(pred.cpu().item()),
                str(target.cpu().item()),
                str(output.data[0].cpu().numpy().tolist()),
            )
            
            with open(txt_path, 'a') as f:
                f.write(string)
            
            # print(string)
            final_result.append(string)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            batch_size = videos.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            metric_logger.meters["time_loader"].update(time_loader, n=batch_size)
            metric_logger.meters["time_mem"].update(time_mem, n=batch_size)
            metric_logger.meters["time_com"].update(time_com, n=batch_size)
            
            frame_counter += 1
            t1 = time.time()

    return final_result


def main(args):
    
    if args.smart_test:
        args.output_dir += "/smart_numframe{}_trainlen{}/".format(args.num_frames, args.train_seq_len)
    else:
        args.output_dir += "/last_numframe{}_trainlen{}/".format(args.num_frames, args.train_seq_len)
        
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    txt_file = open(os.path.join(args.output_dir, "val_hyerparamter.txt"), "w")
    txt_file.write(str(args))
        
    device = torch.device(args.device)

    cudnn.benchmark = True

    # Cholec80后40个数据集用于测试：2452890
    # print("Total Test Dataset Length: ", sum([len(d) for d in dataset_test_list]))
    
    return_last_state = args.num_frames < 2

    if "endomamba" in args.model:
        model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        all_frames=args.num_frames,
        with_head=True,
        return_last_state=return_last_state,
        only_cls_token=args.only_cls_token,
    )
    
    txt_file.write(str(model))
    txt_file.close()
    
    state_dict = torch.load(args.pretrained_path, map_location="cpu")['model']
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print("number of params:", n_parameters)

    with torch.cuda.device(device):
        for i in range(len(args.eval_video_id)):
            vid_id = args.eval_video_id[i]
            dataset_test, _ = build_dataset(
                is_train=False, test_mode=True, fps=args.data_fps, args=args, current_video_id=vid_id)
            preds_file = args.output_dir +  "/video_" + vid_id + ".txt"
            test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=args.num_workers)
            result = test_per_sequence(test_loader, model, device, args.train_seq_len, args.smart_test, preds_file)
            print("Save Files: ", preds_file)


if __name__ == "__main__":
    opts = get_args()
    
    main(opts)