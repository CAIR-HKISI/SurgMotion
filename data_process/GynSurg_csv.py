
import os
import pandas as pd
import numpy as np
from collections import defaultdict

# 配置路径
FRAME_ROOT = "data/Surge_Frames/GynSurg_Action/frames"
DATANAME = "GynSurg_Action"
YEAR = 2025
OUT_DIR = "data/Surge_Frames/GynSurg_Action"

def main():
    # 第一层：phase名
    phase_dirs = [d for d in sorted(os.listdir(FRAME_ROOT)) 
                  if os.path.isdir(os.path.join(FRAME_ROOT, d))]

    phase_names = sorted(list(set(phase_dirs)))
    phase2id = {name: i for i, name in enumerate(phase_names)}
    print("Phase2id mapping:", phase2id)

    # 获取每个 phase 下的 case 名列表
    phase2cases = defaultdict(list)
    for phase_name in phase_dirs:
        phase_path = os.path.join(FRAME_ROOT, phase_name)
        if not os.path.isdir(phase_path):
            continue
        # 第二层：case名
        case_names = [d for d in sorted(os.listdir(phase_path)) 
                      if os.path.isdir(os.path.join(phase_path, d))]
        phase2cases[phase_name].extend(case_names)

    # 按 case_name 划分 train/val/test，且所有 phase 下的case独立分配
    all_cases = []
    for phase, cases in phase2cases.items():
        # 存储 phase, case_name
        for case in cases:
            all_cases.append((phase, case))

    # 随机分配到 train/val/test
    rng = np.random.default_rng(seed=42)
    idxs = np.arange(len(all_cases))
    rng.shuffle(idxs)

    n_total = len(all_cases)
    n_train = int(n_total * 0.7 + 0.5)
    n_test = int(n_total * 0.15 + 0.5)
    n_val = n_total - n_train - n_test

    train_idx = idxs[:n_train]
    test_idx = idxs[n_train:n_train+n_test]
    val_idx = idxs[n_train+n_test:]

    split_lists = {'train': train_idx, 'val': val_idx, 'test': test_idx}
    case2split = {}

    for split, idxlist in split_lists.items():
        for idx in idxlist:
            phase, case = all_cases[idx]
            # 唯一键为 phase:case
            case2split[f"{phase}:{case}"] = split

    print(f"训练集: {len(train_idx)} cases, 验证集: {len(val_idx)} cases, 测试集: {len(test_idx)} cases")

    all_data = []
    idx = 0

    # 遍历第一层 phase 目录
    for phase_name in sorted(os.listdir(FRAME_ROOT)):
        phase_path = os.path.join(FRAME_ROOT, phase_name)
        if not os.path.isdir(phase_path):
            continue
        phase_gt = phase2id[phase_name]

        # 遍历第二层 case 目录
        for case_name in sorted(os.listdir(phase_path)):
            case_path = os.path.join(phase_path, case_name)
            if not os.path.isdir(case_path):
                continue

            # 自动生成Case_ID，仅数字部分，否则使用自增idx+1
            try:
                case_id = int(''.join([c for c in case_name if c.isdigit()])) if any(i.isdigit() for i in case_name) else idx+1
            except Exception:
                case_id = idx+1

            # 依据 phase 和 case 名来划分 split
            case_key = f"{phase_name}:{case_name}"
            split = case2split.get(case_key, "train")

            # 遍历第三层（图片）
            for file in sorted(os.listdir(case_path)):
                if not (file.lower().endswith('.jpg') or file.lower().endswith('.jpeg') or file.lower().endswith('.png')):
                    continue
                frame_path = os.path.join(case_path, file)
                data = {
                    "Index": idx,
                    "DataName": DATANAME,
                    "Year": YEAR,
                    "Case_Name": case_name,
                    "Case_ID": case_id,
                    "Frame_Path": frame_path,
                    "Phase_GT": phase_gt,
                    "Phase_Name": phase_name,
                    "Split": split
                }
                all_data.append(data)
                idx += 1

    # 保存按照split划分的csv
    df = pd.DataFrame(all_data)
    for split in ["train", "val", "test"]:
        out_csv = os.path.join(OUT_DIR, f"{split}_metadata.csv")
        df_split = df[df["Split"] == split]
        df_split.to_csv(out_csv, index=False)
        print(f"Saved {len(df_split)} rows to {out_csv}")

if __name__ == "__main__":
    main()
