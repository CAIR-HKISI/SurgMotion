import os
import re

SRC_BASE = 'configs/fdtn_probing/dinov3'
DST_BASE = 'configs/fdtn_probing/videomaev2'

MODELS = {
    'large': 'videomaev2_large',
    'huge': 'videomaev2_huge',
    'giant': 'videomaev2_giant',
}

def process_file(dataset_name, src_file):
    with open(src_file, 'r') as f:
        content = f.read()

    # Determine filename
    base_name = os.path.basename(src_file)
    # Expected format: dinov3_vith_64f_DatasetName.yaml
    # We want: videomaev2_{size}_64f_DatasetName.yaml
    
    # Extract DatasetName part
    match = re.search(r'dinov3_vith(?:16plus)?_64f_(.+)\.yaml', base_name)
    if not match:
        # Try generic match
        match = re.search(r'dinov3_.+_64f_(.+)\.yaml', base_name)
    
    if match:
        dataset_suffix = match.group(1)
    else:
        # Fallback: use folder name or part of filename
        dataset_suffix = dataset_name

    for size, model_name in MODELS.items():
        new_content = content
        
        # 1. Update basic fields
        new_content = re.sub(r'logs/foundation/dinov3_[^/\n]+', f'logs/foundation/{model_name}_{dataset_suffix}', new_content)
        new_content = re.sub(r'tag: dinov3_[^\n]+', f'tag: {model_name}_64f_{dataset_suffix}', new_content)
        
        # 2. Update WandB
        new_content = re.sub(r'name: dinov3_[^\n]+', f'name: {model_name}_64f_{dataset_suffix}', new_content)
        new_content = re.sub(r'tags:\n\s+- ([^\n]+)\n\s+- phase-probing\n\s+- dinov3_[^\n]+', 
                             f'tags:\n    - \\1\n    - phase-probing\n    - {model_name}-64f', new_content)
        
        new_content = re.sub(r'notes: "Multi-head probing on ([^"]+)"', 
                             f'notes: "Multi-head probing on \\1 with VideoMAEv2 {size.capitalize()}"', new_content)

        # 3. Update model_kwargs
        # Remove old encoder block and replace model_type
        new_content = re.sub(r'model_type: dinov3', 'model_type: videomae', new_content)
        
        # Regex to replace encoder section
        # Assuming format:
        #   encoder:
        #     model_name: dinov3_vith16plus
        encoder_pattern = r'  encoder:\n    model_name: [^\n]+'
        encoder_replacement = f'  encoder:\n    model_name: {model_name}'
        new_content = re.sub(encoder_pattern, encoder_replacement, new_content)

        # 4. Save
        dst_dir = os.path.join(DST_BASE, dataset_name)
        os.makedirs(dst_dir, exist_ok=True)
        
        dst_filename = f"{model_name}_64f_{dataset_suffix}.yaml"
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

