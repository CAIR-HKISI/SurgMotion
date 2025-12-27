import os
import re

SRC_BASE = 'configs/fdtn_probing/dinov3'
DST_BASE = 'configs/fdtn_probing/surgvlp'

MODELS = [
    {
        'file_prefix': 'surgvlp_res50',
        'model_name': 'surgvlp',
        'description': 'SurgVLP ResNet50'
    }
]

def process_file(dataset_name, src_file):
    with open(src_file, 'r') as f:
        content = f.read()

    # Determine filename
    base_name = os.path.basename(src_file)
    
    # Extract DatasetName part
    # Expected format: dinov3_vith_64f_DatasetName.yaml or dinov3_vith16plus_64f_DatasetName.yaml
    match = re.search(r'dinov3_(?:vith|vith16plus|vit\w+)_64f_(.+)\.yaml', base_name)
    if not match:
        # Try generic match
        match = re.search(r'dinov3_.+_64f_(.+)\.yaml', base_name)
    
    if match:
        dataset_suffix = match.group(1)
    else:
        # Fallback: use folder name
        dataset_suffix = dataset_name

    for model_cfg in MODELS:
        new_content = content
        
        file_prefix = model_cfg['file_prefix']
        m_name = model_cfg['model_name']
        desc = model_cfg['description']
        
        # 1. Update basic fields
        # logs/foundation/dinov3_vith16plus_PolypDiag -> logs/foundation/surgvlp_res50_PolypDiag
        new_content = re.sub(r'logs/foundation/dinov3_[^/\n]+', f'logs/foundation/{file_prefix}_{dataset_suffix}', new_content)
        
        # tag: dinov3_vith16plus_64f_PolypDiag -> tag: surgvlp_res50_64f_PolypDiag
        new_content = re.sub(r'tag: dinov3_[^\n]+', f'tag: {file_prefix}_64f_{dataset_suffix}', new_content)
        
        # 2. Update WandB
        # name: dinov3_vith16plus_64f_PolypDiag -> name: surgvlp_res50_64f_PolypDiag
        new_content = re.sub(r'name: dinov3_[^\n]+', f'name: {file_prefix}_64f_{dataset_suffix}', new_content)
        
        # tags: ...
        # - dinov3_vith16plus-64f -> - surgvlp_res50-64f
        new_content = re.sub(r'tags:\n\s+- ([^\n]+)\n\s+- phase-probing\n\s+- dinov3_[^\n]+', 
                             f'tags:\n    - \\1\n    - phase-probing\n    - {file_prefix}-64f', new_content)
        
        # notes
        new_content = re.sub(r'notes: "Multi-head probing on ([^"]+)"', 
                             f'notes: "Multi-head probing on \\1 with {desc}"', new_content)

        # 3. Update model_kwargs
        # Remove old encoder block and replace model_type
        new_content = re.sub(r'model_type: dinov3', 'model_type: surgvlp', new_content)
        
        # Regex to replace encoder section
        # Assuming format:
        #   encoder:
        #     model_name: dinov3_vith16plus
        encoder_pattern = r'  encoder:\n    model_name: [^\n]+'
        encoder_replacement = f'  encoder:\n    model_name: {m_name}'
        new_content = re.sub(encoder_pattern, encoder_replacement, new_content)

        # 4. Save
        dst_dir = os.path.join(DST_BASE, dataset_name)
        os.makedirs(dst_dir, exist_ok=True)
        
        dst_filename = f"{file_prefix}_64f_{dataset_suffix}.yaml"
        dst_path = os.path.join(dst_dir, dst_filename)
        
        with open(dst_path, 'w') as f:
            f.write(new_content)
        print(f"Generated: {dst_path}")

def main():
    if not os.path.exists(SRC_BASE):
        print(f"Source not found: {SRC_BASE}")
        return

    for item in os.listdir(SRC_BASE):
        src_dir = os.path.join(SRC_BASE, item)
        if os.path.isdir(src_dir):
            # Find yaml file
            yaml_files = [f for f in os.listdir(src_dir) if f.endswith('.yaml')]
            if not yaml_files:
                continue
            
            # Use the first one (usually there is one main one or multiple similar ones)
            # Prefer ones with '64f' if multiple
            target_yaml = next((f for f in yaml_files if '64f' in f), yaml_files[0])
            
            process_file(item, os.path.join(src_dir, target_yaml))

if __name__ == '__main__':
    main()

