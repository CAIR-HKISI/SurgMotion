import os
import re

SRC_BASE = 'configs/fdtn_probing/dinov3'
DST_BASE = 'configs/fdtn_probing/internvideo'

MODELS = [
    {
        'file_prefix': 'internvideo2_1b',
        'model_name': 'InternVideo2-Stage2_1B-224p-f4',
        'description': 'InternVideo2 Stage2 1B 224p 4f'
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
        # logs/foundation/dinov3_vith16plus_PolypDiag -> logs/foundation/internvideo2_1b_PolypDiag
        new_content = re.sub(r'logs/foundation/dinov3_[^/\n]+', f'logs/foundation/{file_prefix}_{dataset_suffix}', new_content)
        
        # tag: dinov3_vith16plus_64f_PolypDiag -> tag: internvideo2_1b_64f_PolypDiag
        new_content = re.sub(r'tag: dinov3_[^\n]+', f'tag: {file_prefix}_64f_{dataset_suffix}', new_content)
        
        # 2. Update WandB
        # name: dinov3_vith16plus_64f_PolypDiag -> name: internvideo2_1b_64f_PolypDiag
        new_content = re.sub(r'name: dinov3_[^\n]+', f'name: {file_prefix}_64f_{dataset_suffix}', new_content)
        
        # tags: ...
        # - dinov3_vith16plus-64f -> - internvideo2_1b-64f
        new_content = re.sub(r'tags:\n\s+- ([^\n]+)\n\s+- phase-probing\n\s+- dinov3_[^\n]+', 
                             f'tags:\n    - \\1\n    - phase-probing\n    - {file_prefix}-64f', new_content)
        
        # notes
        new_content = re.sub(r'notes: "Multi-head probing on ([^"]+)"', 
                             f'notes: "Multi-head probing on \\1 with {desc}"', new_content)

        # 3. Update model_kwargs
        # Remove old encoder block and replace model_type
        new_content = re.sub(r'model_type: dinov3', 'model_type: internvideo', new_content)
        
        # Regex to replace encoder section
        # Assuming format:
        #   encoder:
        #     model_name: dinov3_vith16plus
        encoder_pattern = r'  encoder:\n    model_name: [^\n]+'
        encoder_replacement = f'  encoder:\n    model_name: {m_name}'
        new_content = re.sub(encoder_pattern, encoder_replacement, new_content)
        
        # InternVideo specific settings
        # Ensure resolution is 224 (DINOv3 was usually 224 or 518)
        # InternVideo2-1B-224p-f4 uses 224
        new_content = re.sub(r'resolution: \d+', 'resolution: 224', new_content)
        
        # InternVideo requires frames_per_clip = 4 if using f4 checkpoint, 
        # But our pipeline might be using 64f. 
        # The adapter has 'frames_per_clip' argument which defaults to 4. 
        # If the input clip has 64 frames (probing pipeline), adapter will internally handle it (process as batch of 4-frame clips or just assume it is long video?)
        # 
        # Wait, probing pipeline loads `frames_per_clip` frames.
        # DINOv3 usually takes 64 frames. 
        # InternVideo2 f4 model is trained on 4 frames.
        # If we feed 64 frames, adapter needs to handle it.
        # In adapter.py: "B, C, F, H, W = x.shape... if F != self.num_frames: pass"
        # It seems adapter just runs it. InternVideo2 accepts variable frames if pos_embed is interpolated (which we added).
        # So we can keep `frames_per_clip: 64` in the config if we want to probe on 64 frames.
        # Or we should reduce it?
        # Usually probing on foundation models uses a set number of frames (e.g. 64).
        # We will keep 64.
        
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

