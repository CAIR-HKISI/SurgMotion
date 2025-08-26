import os
import pickle
import csv

'''
{"BBP18": [{'Frame_id': '00000067',
 'Original_frame_id': 66,
 'Phase_gt': 0,
 'Step_gt': 1,
 'unique_id': 686995500066,
 'Event_ID': [],
 'Overall': 0,
 'Bleeding': 0,
 'Mechanical injury': 0,
 'Thermal injury': 0,
 'Ischemic injury': 0,
 'Insufficient closure of anastomosis': 0,
 'Bleeding - 1': 0,
 'Bleeding - 2': 0,
 'Bleeding - 3': 0,
 'Bleeding - 4': 0,
 'Bleeding - 5': 0,
 'Thermal injury - 1': 0,
 'Thermal injury - 2': 0,
 'Thermal injury - 3': 0,
 'Thermal injury - 4': 0,
 'Thermal injury - 5': 0,
 'Mechanical injury - 1': 0,
 'Mechanical injury - 2': 0,
 'Mechanical injury - 3': 0,
 'Mechanical injury - 4': 0,
 'Mechanical injury - 5': 0,
 'Ischemic injury - 0': 0,
 'Ischemic injury - 1': 0,
 'Ischemic injury - 2': 0,
 'Insufficient closure of anastomosis - 1': 0,
 'Bleeding - rectified': 0}]
 }
 
'''


def read_pkl_data_to_csv(pkl_path, img_path, csv_output_path, hospital="Bern", year="2023"):
    """
    读取pickle文件并转换为CSV格式
    
    Args:
        pkl_path: pickle文件路径
        img_path: 图片根目录路径
        csv_output_path: CSV输出文件路径
        hospital: 医院名称
        year: 年份
    """
    # 读取pickle文件
    with open(pkl_path, "rb") as fp:
        data = pickle.load(fp)
    
    # 处理图片路径
    root_dir = img_path
    if not os.path.exists(root_dir):
        root_dir = root_dir.replace('train', '').replace('val', '').replace('test', '')
    
    # Phase阶段名称映射
    phase_name_map = {
        0: "preparation",
        1: "gastric_pouch_creation",
        2: "omentum_division",
        3: "gastrojejunal_anastomosis",
        4: "anastomosis_test",
        5: "jejunal_separation",
        6: "closure_petersen_space",
        7: "jejunojejunal_anastomosis",
        8: "closure_mesenteric_defect",
        9: "cleaning_coagulation",
        10: "disassembling",
        11: "other_intervention",
        12: "out_of_body",
        13: "severe_index"
    }
    
    # Step步骤名称映射
    step_name_map = {
        0: "s0_null_step",
        1: "s1_cavity_exploration",
        2: "s2_trocar_placement",
        3: "s3_retractor_placement",
        4: "s4_crura_dissection",
        5: "s5_his_angle_dissection",
        6: "s6_horizontal_stapling",
        7: "s7_retrogastric_dissection",
        8: "s8_vertical_stapling",
        9: "s9_gastric_remnant_reinforcement",
        10: "s10_gastric_pouch_reinforcement",
        11: "s11_gastric_opening",
        12: "s12_omental_lifting",
        13: "s13_omental_section",
        14: "s14_adhesiolysis",
        15: "s15_treitz_angle_identification",
        16: "s16_biliary_limb_measurement",
        17: "s17_jejunum_opening",
        18: "s18_gastrojejunal_stapling",
        19: "s19_gastrojejunal_defect_closing",
        20: "s20_mesenteric_opening",
        21: "s21_jejunal_section",
        22: "s22_gastric_tube_placement",
        23: "s23_jejunal_clamping",
        24: "s24_ink_injection",
        25: "s25_visual_assessment",
        26: "s26_gastrojejunal_anastomosis_reinforcement",
        27: "s27_petersen_space_exposure",
        28: "s28_petersen_space_closing",
        29: "s29_biliary_limb_opening",
        30: "s30_alimentary_limb_measurement",
        31: "s31_alimentary_limb_opening",
        32: "s32_jejunojejunal_stapling",
        33: "s33_jejunojejunal_defect_closing",
        34: "s34_jejunojejunal_anastomosis_reinforcement",
        35: "s35_staple_line_reinforcement",
        36: "s36_mesenteric_defect_exposure",
        37: "s37_mesenteric_defect_closing",
        38: "s38_anastomosis_fixation",
        39: "s39_coagulation",
        40: "s40_irrigation_aspiration",
        41: "s41_parietal_closure",
        42: "s42_trocar_removal",
        43: "s43_calibration",
        44: "s44_drainage_insertion",
        45: "s45_specimen_retrieval"
    }
    
    # 准备CSV数据
    csv_data = []
    index = 0
    
    for vid_name in sorted(data.keys()):
        case_id = int(vid_name[-2:])
        
        for item in data[vid_name]:
            frame_path = os.path.join(root_dir, 'frames', vid_name, f"{item['Frame_id']}.jpg")
            if not os.path.exists(frame_path):
                print(f"Frame not found: {frame_path}")
            
            phase_gt = item['Phase_gt']
            step_gt = item['Step_gt']
            phase_name = phase_name_map.get(phase_gt, f"unknown_phase_{phase_gt}")
            step_name = step_name_map.get(step_gt, f"unknown_step_{step_gt}")
            
            csv_row = {
                'index': index,
                'Hospital': hospital,
                'Year': year,
                'Case_Name': vid_name,
                'Case_ID': case_id,
                'Frame_Path': frame_path,
                'Phase_GT': int(phase_gt),
                'Phase_Name': phase_name,
                'Step_GT': int(step_gt),
                'Step_Name': step_name
            }
            csv_data.append(csv_row)
            index += 1
    
    # 写入CSV文件
    fieldnames = ['index', 'Hospital', 'Year', 'Case_Name', 'Case_ID', 'Frame_Path', 
                  'Phase_GT', 'Phase_Name', 'Step_GT', 'Step_Name']
    
    with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"CSV文件已保存: {csv_output_path}, 共 {len(csv_data)} 行数据")
    return csv_data

def process_dataset(img_path, train_pkl_path, val_pkl_path, test_pkl_path, 
                   output_dir="./", hospital="Bern", year="2023"):
    """
    处理完整数据集
    
    Args:
        img_path: 图片根目录路径
        train_pkl_path: 训练集pickle文件路径
        val_pkl_path: 验证集pickle文件路径  
        test_pkl_path: 测试集pickle文件路径
        output_dir: CSV文件输出目录
        hospital: 医院名称
        year: 年份
    """
    pkl_paths = {
        'train': train_pkl_path,
        'val': val_pkl_path,
        'test': test_pkl_path
    }
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理各个数据集
    for split_name, pkl_path in pkl_paths.items():
        csv_output_path = os.path.join(output_dir, f"{split_name}_metadata.csv")
        read_pkl_data_to_csv(
            pkl_path=pkl_path,
            img_path=img_path,
            csv_output_path=csv_output_path,
            hospital=hospital,
            year=year
        )

if __name__ == "__main__":
    # 配置参数 - 您可以在这里修改路径
    dir_name = "BernBypass70"
    label_name = "bern"
    
    dir_name = "StrasBypass70"
    label_name = "strasbourg"
    
    img_path = f"data/Surge_Frames/MultiBypass140/{dir_name}"
    train_pkl_path = f"data/bypass_label/{label_name}/labels_by70_splits/labels/train/1fps_100_0.pickle"
    val_pkl_path = f"data/bypass_label/{label_name}/labels_by70_splits/labels/val/1fps_0.pickle"
    test_pkl_path = f"data/bypass_label/{label_name}/labels_by70_splits/labels/test/1fps_0.pickle"
    output_dir = f"data/Surge_Frames/MultiBypass140/{dir_name}/"  # CSV文件输出目录
    
    year = "2023"
    
    # 执行处理
    process_dataset(
        img_path=img_path,
        train_pkl_path=train_pkl_path,
        val_pkl_path=val_pkl_path,
        test_pkl_path=test_pkl_path,
        output_dir=output_dir,
        hospital=dir_name,
        year=year
    )






