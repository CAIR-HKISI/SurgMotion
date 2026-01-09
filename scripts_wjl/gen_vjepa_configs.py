import os
import re

# VJEPA Checkpoint 配置
CHECKPOINT_PATH = 'logs/cooldown_vitg-256px-64f_40epoch/last.pt'
MODEL_NAME = 'vit_giant_xformers'  # 256px 使用 xformers 版本
MODEL_SUFFIX = 'vjepa_vitg'
TAG_SUFFIX = 'vjepa_vitg_64f'
WANDB_TAG = 'vjepa_vitg_64f'

# 输出目录
OUTPUT_BASE_DIR = 'configs/fdtn_probing/vjepa_cpt'

# 参考的模板目录（使用 dinov3 配置作为模板获取数据集信息）
TEMPLATE_BASE_DIR = 'configs/fdtn_probing/dinov3'

# VJEPA 的 model_kwargs 模板
MODEL_KWARGS_TEMPLATE = """model_kwargs:
  checkpoint: {checkpoint}
  module_name: evals.video_classification_frozen.modelcustom.vit_encoder_multiclip
  pretrain_kwargs:
    encoder:
      checkpoint_key: target_encoder
      img_temporal_dim_size: null
      model_name: {model_name}
      patch_size: 16
      tubelet_size: 2
      uniform_power: true
      use_rope: true
  wrapper_kwargs:
    max_frames: 128
    use_pos_embed: false"""

def extract_dataset_config(content):
    """从配置内容中提取数据集相关配置"""
    # 提取数据集相关的字段
    dataset_info = {}
    
    # 提取 dataset_train
    match = re.search(r'dataset_train:\s*(.+)', content)
    if match:
        dataset_info['dataset_train'] = match.group(1).strip()
    
    # 提取 dataset_val
    match = re.search(r'dataset_val:\s*(.+)', content)
    if match:
        dataset_info['dataset_val'] = match.group(1).strip()
    
    # 提取 num_classes
    match = re.search(r'num_classes:\s*(\d+)', content)
    if match:
        dataset_info['num_classes'] = int(match.group(1))
    
    # 提取 frames_per_clip
    match = re.search(r'frames_per_clip:\s*(\d+)', content)
    if match:
        dataset_info['frames_per_clip'] = int(match.group(1))
    
    # 提取 num_segments
    match = re.search(r'num_segments:\s*(\d+)', content)
    if match:
        dataset_info['num_segments'] = int(match.group(1))
    
    # 提取 batch_size
    match = re.search(r'batch_size:\s*(\d+)', content)
    if match:
        dataset_info['batch_size'] = int(match.group(1))
    
    # 提取 task_type（如果有）
    match = re.search(r'task_type:\s*(\w+)', content)
    if match:
        dataset_info['task_type'] = match.group(1)
    
    # 提取 use_weighted_loss（如果有）
    match = re.search(r'use_weighted_loss:\s*(true|false)', content)
    if match:
        dataset_info['use_weighted_loss'] = match.group(1)
    
    return dataset_info

def process_template_folder(dataset_name, template_folder_path, output_folder_path):
    """处理模板文件夹，生成 vjepa 配置"""
    # 找到模板文件（优先使用 vitl）
    files = [f for f in os.listdir(template_folder_path) if f.endswith('.yaml')]
    if not files:
        print(f"No yaml files in {template_folder_path}")
        return
    
    # 优先选择 vitl 文件作为模板
    template_file = None
    for f in files:
        if 'vitl' in f:
            template_file = f
            break
    if not template_file:
        template_file = files[0]
    
    template_path = os.path.join(template_folder_path, template_file)
    with open(template_path, 'r') as f:
        content = f.read()
    
    # 提取数据集配置
    dataset_info = extract_dataset_config(content)
    
    # 从文件名提取数据集后缀
    match = re.search(r'dinov3_(?:vitl|vith(?:16plus)?|.+)_64f_(.+)\.yaml', template_file)
    if match:
        dataset_suffix = match.group(1)
    else:
        dataset_suffix = dataset_name.lower().replace('-', '_')
    
    print(f"Processing {dataset_name} using {template_file} as template -> {dataset_suffix}")
    
    # 创建输出目录
    os.makedirs(output_folder_path, exist_ok=True)
    
    # 生成新配置
    new_content = content
    
    # 1. 替换 eval_name
    new_content = re.sub(r'eval_name:\s*\S+', 'eval_name: foundation_phase_probing', new_content)
    
    # 2. 替换 folder
    new_content = re.sub(
        r'folder:\s*logs/foundation/\S+',
        f'folder: logs/foundation/{MODEL_SUFFIX}_{dataset_suffix}',
        new_content
    )
    
    # 3. 替换 tag
    new_content = re.sub(
        r'tag:\s*\S+',
        f'tag: {TAG_SUFFIX}_{dataset_suffix}',
        new_content
    )
    
    # 4. 替换 wandb name
    new_content = re.sub(
        r'name:\s*dinov3[^\n]+',
        f'name: {TAG_SUFFIX}_{dataset_suffix}',
        new_content
    )
    
    # 5. 替换 wandb tags
    new_content = re.sub(
        r'- dinov3[-_][a-zA-Z0-9_]+-64f',
        f'- {WANDB_TAG}',
        new_content
    )
    new_content = re.sub(
        r'- dinov3_vitl_64f',
        f'- {WANDB_TAG}',
        new_content
    )
    
    # 6. 替换 wandb project（如果需要）
    new_content = re.sub(
        r'project:\s*foundation-model-probing',
        'project: nsjepa-probing',
        new_content
    )
    
    # 7. 替换 resolution 为 256（vjepa 使用 256px）
    new_content = re.sub(r'resolution:\s*\d+', 'resolution: 256', new_content)
    
    # 8. 替换 model_kwargs 部分
    # 先删除旧的 model_kwargs
    new_content = re.sub(r'model_kwargs:.*', '', new_content, flags=re.DOTALL)
    
    # 添加新的 model_kwargs
    model_kwargs = MODEL_KWARGS_TEMPLATE.format(
        checkpoint=CHECKPOINT_PATH,
        model_name=MODEL_NAME
    )
    new_content = new_content.rstrip() + '\n\n' + model_kwargs + '\n'
    
    # 删除 resume_iter（如果有）
    new_content = re.sub(r'resume_iter:\s*\d+\n', '', new_content)
    
    # 删除 id（如果有）
    new_content = re.sub(r'id:\s*\S+\n', '', new_content)
    
    # 设置 resume_checkpoint 为 false
    new_content = re.sub(r'resume_checkpoint:\s*\S+', 'resume_checkpoint: false', new_content)
    
    # 更新 notes
    new_content = re.sub(
        r'notes:\s*"[^"]*"',
        f'notes: "Multi-head probing on {dataset_name} dataset with VJEPA ViT-G encoder"',
        new_content
    )
    
    # 写入文件
    output_filename = f"{MODEL_SUFFIX}_64f_{dataset_suffix}.yaml"
    output_path = os.path.join(output_folder_path, output_filename)
    
    with open(output_path, 'w') as f:
        f.write(new_content)
    
    print(f"  Generated: {output_path}")

def main():
    if not os.path.exists(TEMPLATE_BASE_DIR):
        print(f"Template dir not found: {TEMPLATE_BASE_DIR}")
        return
    
    # 创建输出基目录
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    # 遍历模板目录中的所有数据集文件夹
    for item in sorted(os.listdir(TEMPLATE_BASE_DIR)):
        template_folder_path = os.path.join(TEMPLATE_BASE_DIR, item)
        if os.path.isdir(template_folder_path):
            output_folder_path = os.path.join(OUTPUT_BASE_DIR, item)
            process_template_folder(item, template_folder_path, output_folder_path)

if __name__ == "__main__":
    main()
