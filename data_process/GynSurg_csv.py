
import os
import pandas as pd
import numpy as np
from collections import defaultdict

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

def split_cases(case_names, ratio=(0.7, 0.15, 0.15), seed=42):
    """
    按比例划分case到train/val/test
    每个类别内分别以 ratio 分割，保证类别均匀分布于三个 split。
    """
    case_names = sorted(case_names)
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(case_names))
    rng.shuffle(idxs)
    n_total = len(case_names)
    n_train = int(n_total * ratio[0] + 0.5)
    n_val = int(n_total * ratio[1] + 0.5)
    n_test = n_total - n_train - n_val
    splits = {}
    for i, idx in enumerate(idxs):
        if i < n_train:
            splits[case_names[idx]] = 'train'
        elif i < n_train + n_val:
            splits[case_names[idx]] = 'val'
        else:
            splits[case_names[idx]] = 'test'
    return splits

def gather_metadata(frame_root, class2id, dataname, out_dir,
                   year=2025, ratio=(0.7,0.15,0.15), seed=42,
                   caseid_map=None):
    # 收集类别下的所有病例(case)文件夹
    cat2cases = defaultdict(list)
    for class_name in class2id:
        class_dir = os.path.join(frame_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for case in sorted(os.listdir(class_dir)):
            case_path = os.path.join(class_dir, case)
            if os.path.isdir(case_path):
                cat2cases[class_name].append(case)

    # 按每个类别分别 split 各自的 case
    case2split = dict()
    for classname, case_list in cat2cases.items():
        splits = split_cases(case_list, ratio=ratio, seed=seed)
        for case, split in splits.items():
            case2split[(classname, case)] = split

    # 若未提供 caseid_map，则在当前「单个数据集」内，根据 case 名字中的数字串生成本地映射；
    # 若你已经在外部为该数据集准备好了统一的映射（例如视频编号 → ID），
    # 也可以通过 caseid_map 参数传进来复用。
    if caseid_map is None:
        # 统计全部case名字中所有独特的数字串作为视频id
        # 例如 '01_05' -> '0105'；对所有 case 去掉非数字后按字符串去重
        digit_set = set()
        for class_cases in cat2cases.values():
            for case in class_cases:
                digits = ''.join([c for c in case if c.isdigit()])
                if digits != '':
                    digit_set.add(digits)
        # 给所有不同id字符串确定唯一编号(数字形式)
        sorted_ids = sorted(list(digit_set))
        caseid_map = {d: i+1 for i, d in enumerate(sorted_ids)}

    # 采集 frame 信息
    records = []
    index = 0
    for class_name in class2id:
        class_dir = os.path.join(frame_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        class_id = class2id[class_name]
        for case in sorted(os.listdir(class_dir)):
            case_path = os.path.join(class_dir, case)
            if not os.path.isdir(case_path):
                continue
            split = case2split.get((class_name, case), "train")
            # 重新生成Case_ID: 取case名字中所有数字作为id字符串, 然后查找caseid_map。若无数字则为0。
            digits = ''.join([c for c in case if c.isdigit()])
            if digits in caseid_map:
                case_id = caseid_map[digits]
            else:
                case_id = 0
            # 遍历 case 目录下所有图片
            img_files = [f for f in sorted(os.listdir(case_path))
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img in img_files:
                full_path = os.path.join(case_path, img)
                record = {
                    "Index": index,
                    "DataName": dataname,
                    "Year": year,
                    "Case_Name": case,
                    "Case_ID": case_id,
                    "Frame_Path": full_path,
                    "Phase_GT": class_id,
                    "Phase_Name": class_name,
                    "Split": split
                }
                records.append(record)
                index += 1

    # 保存到 train/val/test csv
    ensure_dir(out_dir)
    df = pd.DataFrame(records)
    for split in ['train', 'val', 'test']:
        split_df = df[df["Split"] == split]
        out_csv = os.path.join(out_dir, f"{split}_metadata.csv")
        split_df.to_csv(out_csv, index=False)
        print(f"{dataname}: 保存 {len(split_df)} 条数据到 {out_csv}")


def main():
    # 1. GynSurg_Action
    action_classes = [
        "NeedlePassing", "Coagulation", "Irrigation", "Suction",
        "Transection", "Rest"
    ]
    action_mapping = {
        "NeedlePassing": 0,
        "Coagulation": 1,
        "Irrigation": 2,
        "Suction": 2,
        "Transection": 3,
        "Rest": 4
    }
    action_root = "data/Surge_Frames/GynSurg_Action/frames"
    action_out = "data/Surge_Frames/GynSurg_Action"
    ensure_dir(action_out)
    # 1) GynSurg_Action（在该数据集内部自动生成视频 ID 映射）
    gather_metadata(
        frame_root=action_root,
        class2id=action_mapping,
        dataname="GynSurg_Action",
        out_dir=action_out,
        ratio=(0.7,0.15,0.15),
        seed=42,
    )

    # 2. GynSurg_Somke
    smoke_mapping = {
        "Non-smoke": 0,
        "Smoke": 1
    }
    smoke_root = "data/Surge_Frames/GynSurg_Somke/frames"
    smoke_out = "data/Surge_Frames/GynSurg_Somke"
    ensure_dir(smoke_out)
    gather_metadata(
        frame_root=smoke_root,
        class2id=smoke_mapping,
        dataname="GynSurg_Somke",
        out_dir=smoke_out,
        ratio=(0.7,0.15,0.15),
        seed=42,
    )

    # 3. GynSurg_Bleeding
    bleed_mapping = {
        "Non-bleeding": 0,
        "Bleeding": 1
    }
    bleed_root = "data/Surge_Frames/GynSurg_Bleeding/frames"
    bleed_out = "data/Surge_Frames/GynSurg_Bleeding"
    ensure_dir(bleed_out)
    gather_metadata(
        frame_root=bleed_root,
        class2id=bleed_mapping,
        dataname="GynSurg_Bleeding",
        out_dir=bleed_out,
        ratio=(0.7,0.15,0.15),
        seed=42,
    )

if __name__ == "__main__":
    main()
