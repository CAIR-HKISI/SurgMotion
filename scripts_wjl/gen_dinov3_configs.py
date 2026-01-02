import os
import re

BASE_DIR = 'configs/fdtn_probing/dinov3'

MODELS = {
    'vitl': {
        'model_name': 'dinov3_vitl14',
        'folder_suffix': 'dinov3_vitl',
        'tag_suffix': 'dinov3_vitl_64f',
        'wandb_tag': 'dinov3_vitl_64f'
    },
    'vith': {
        'model_name': 'dinov3_vith16plus',
        'folder_suffix': 'dinov3_vith16plus',
        'tag_suffix': 'dinov3_vith16plus_64f',
        'wandb_tag': 'dinov3_vith16plus-64f'
    }
}

def process_folder(dataset_name, folder_path):
    # Find a source file
    files = [f for f in os.listdir(folder_path) if f.endswith('.yaml')]
    if not files:
        print(f"No yaml files in {folder_path}")
        return

    # Prefer one that has '64f' and looks like a dinov3 config
    # Try to find one that is EITHER vitl or vith to use as template
    source_file = None
    
    # Priority: vitl -> vith -> any
    for f in files:
        if 'vitl' in f and '64f' in f:
            source_file = f
            break
    
    if not source_file:
        for f in files:
            if 'vith' in f and '64f' in f:
                source_file = f
                break
    
    if not source_file:
        source_file = files[0]

    with open(os.path.join(folder_path, source_file), 'r') as f:
        content = f.read()

    # Extract dataset suffix from filename if possible
    # Format usually: dinov3_{size}_64f_{Dataset}.yaml
    # Regex to capture Dataset part
    match = re.search(r'dinov3_(?:vitl|vith(?:16plus)?|.+)_64f_(.+)\.yaml', source_file)
    if match:
        dataset_suffix = match.group(1)
    else:
        # If filename doesn't match expected pattern, try to infer from dataset_name
        # or just use the part after the last underscore if reasonable
        parts = source_file.replace('.yaml', '').split('_')
        if len(parts) > 1:
            dataset_suffix = parts[-1]
        else:
            dataset_suffix = dataset_name

    print(f"Processing {dataset_name} using {source_file} as template")

    for target_key, target_config in MODELS.items():
        new_content = content
        
        # 1. Folder
        # Pattern: folder: logs/foundation/dinov3_[^/\n]+
        # We replace the whole dinov3 part
        new_content = re.sub(r'folder: logs/foundation/dinov3_[^/\n]+', 
                             f'folder: logs/foundation/{target_config["folder_suffix"]}_{dataset_suffix}', new_content)

        # 2. Tag
        # Pattern: tag: dinov3_[^\n]+
        new_content = re.sub(r'tag: dinov3_[^\n]+', 
                             f'tag: {target_config["tag_suffix"]}_{dataset_suffix}', new_content)

        # 3. WandB Name
        # Pattern: name: dinov3_[^\n]+
        new_content = re.sub(r'name: dinov3_[^\n]+', 
                             f'name: {target_config["tag_suffix"]}_{dataset_suffix}', new_content)

        # 4. WandB Tags
        # Pattern: - dinov3-[^-\n]+-64f OR - dinov3_[^-\n]+-64f
        # We look for something starting with - dinov3... and ending with -64f
        new_content = re.sub(r'- dinov3[-_][a-zA-Z0-9]+-64f', 
                             f'- {target_config["wandb_tag"]}', new_content)
        
        # 5. Model Name
        # Pattern: model_name: dinov3.*
        # This handles cases like 'dinov3', 'dinov3_vitl14', 'dinov3_vith16plus'
        new_content = re.sub(r'model_name: dinov3[^\n]*', 
                             f'model_name: {target_config["model_name"]}', new_content)

        # 6. Handle eval_checkpoint or other paths containing model names
        # We do a best-effort string replacement for these, avoiding the keys we already handled.
        # We only target specific tokens to avoid breaking things.
        
        if target_key == 'vitl':
             # Replace vith variants with vitl
             new_content = new_content.replace('dinov3_vith16plus', 'dinov3_vitl')
             # new_content = new_content.replace('dinov3_vith', 'dinov3_vitl') # potentially dangerous if part of other word, but unlikely here
        elif target_key == 'vith':
             # Replace vitl variants with vith16plus
             new_content = new_content.replace('dinov3_vitl', 'dinov3_vith16plus')
             # Force batch_size to 1 for vith
             new_content = re.sub(r'batch_size:\s*\d+', 'batch_size: 1', new_content)

        # Write file
        dst_filename = f"dinov3_{target_key}_64f_{dataset_suffix}.yaml"
        dst_path = os.path.join(folder_path, dst_filename)
        
        with open(dst_path, 'w') as f:
            f.write(new_content)
        print(f"  Generated: {dst_filename}")

def main():
    if not os.path.exists(BASE_DIR):
        print(f"Base dir not found: {BASE_DIR}")
        return

    for item in os.listdir(BASE_DIR):
        path = os.path.join(BASE_DIR, item)
        if os.path.isdir(path):
            process_folder(item, path)

if __name__ == "__main__":
    main()

