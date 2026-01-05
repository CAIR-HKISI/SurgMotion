import os
import re

ROOT_DIR = 'configs/fdtn_probing'
# Skill dataset config
TRAIN_CSV = 'data/Surge_Frames/Private_SYSU_Brochiscopy_labeled/skill_train_metadata.csv'
TEST_CSV = 'data/Surge_Frames/Private_SYSU_Brochiscopy_labeled/skill_test_metadata.csv'
NUM_CLASSES = 2
DATASET_TAG = 'SYSU_Skill'
DATASET_NAME_LOWER = 'sysu_skill'

# 统一配置: 8 segments × 16 frames = 128 frames 
NUM_SEGMENTS = 8
FRAMES_PER_CLIP = 16
BATCH_SIZE = 2
MAX_FRAMES = 128  # 需要 >= num_segments * frames_per_clip
NUM_EPOCHS = 4
# 大模型配置
LARGE_MODELS = ['huge', 'giant']
LARGE_MODEL_BATCH_SIZE = 1

def is_large_model(filename):
    """检查文件名是否对应大模型 (huge, giant)"""
    filename_lower = filename.lower()
    return any(model in filename_lower for model in LARGE_MODELS)

def process_content(content, filename=''):
    # 1. Replace dataset paths
    # Handle list format first (vjepa)
    if re.search(r'dataset_train:\s*\n\s*-', content):
        content = re.sub(r'(dataset_train:\s*\n\s*- ).*', f'\\g<1>{TRAIN_CSV}', content)
    else:
        content = re.sub(r'dataset_train: .*', f'dataset_train: {TRAIN_CSV}', content)

    if re.search(r'dataset_val:\s*\n\s*-', content):
        content = re.sub(r'(dataset_val:\s*\n\s*- ).*', f'\\g<1>{TEST_CSV}', content)
    else:
        content = re.sub(r'dataset_val: .*', f'dataset_val: {TEST_CSV}', content)

    # 2. Replace num_classes
    if re.search(r'num_classes:\s*\n\s*-', content):
        content = re.sub(r'(num_classes:\s*\n\s*- )\d+', f'\\g<1>{NUM_CLASSES}', content)
    else:
        content = re.sub(r'num_classes: \d+', f'num_classes: {NUM_CLASSES}', content)

    # Update task_type - replace if exists, or add after tasks_per_node (top level)
    if re.search(r'task_type: .*', content):
        content = re.sub(r'task_type: .*', 'task_type: action', content)
    else:
        # Add task_type after tasks_per_node line (top level config)
        content = re.sub(r'(tasks_per_node: \d+)', r'\1\ntask_type: action', content)
    
    # Update num_segments (统一配置)
    if re.search(r'num_segments: .*', content):
        content = re.sub(r'num_segments: \d+', f'num_segments: {NUM_SEGMENTS}', content)
    
    # Update frames_per_clip (统一配置)
    if re.search(r'frames_per_clip: .*', content):
        content = re.sub(r'frames_per_clip: \d+', f'frames_per_clip: {FRAMES_PER_CLIP}', content)
    
    # Update batch_size (根据模型大小调整)
    if is_large_model(filename):
        batch_size = LARGE_MODEL_BATCH_SIZE
    else:
        batch_size = BATCH_SIZE
    if re.search(r'batch_size: .*', content):
        content = re.sub(r'batch_size: \d+', f'batch_size: {batch_size}', content)
    
    # Update max_frames (需要 >= num_segments * frames_per_clip)
    if re.search(r'max_frames: .*', content):
        content = re.sub(r'max_frames: \d+', f'max_frames: {MAX_FRAMES}', content)
    
    # Update num_epochs
    if re.search(r'num_epochs: .*', content):
        content = re.sub(r'num_epochs: \d+', f'num_epochs: {NUM_EPOCHS}', content)
    
    # Disable wrapper use_pos_embed to avoid index out of bounds
    # (clip_indices can exceed max_frames/tubelet_size)
    content = re.sub(r'(wrapper_kwargs:.*?use_pos_embed:) true', r'\1 false', content, flags=re.DOTALL)

    # 3. Replace PitVis/SYSU references
    content = content.replace('PitVis', DATASET_TAG)
    content = content.replace('pitvis', DATASET_NAME_LOWER)
    
    return content

def main():
    if not os.path.exists(ROOT_DIR):
        print(f"Directory not found: {ROOT_DIR}")
        return

    for root, dirs, files in os.walk(ROOT_DIR):
        # SKIP processing if we are currently INSIDE a PitVis, SYSU, or SYSU_Skill folder
        basename = os.path.basename(root)
        if basename in ['PitVis', 'SYSU', 'SYSU_Skill']:
            continue
            
        # Skip Private-* folders
        if 'Private-' in basename:
            continue

        # Check if PitVis folder exists in current dir
        if 'PitVis' in dirs:
            print(f"Found PitVis folder in {root}")
            src_dir = os.path.join(root, 'PitVis')
            dst_dir = os.path.join(root, 'SYSU_Skill')
            os.makedirs(dst_dir, exist_ok=True)
            
            for f in os.listdir(src_dir):
                if f.endswith('.yaml'):
                    # Skip if it's already a generated sysu file
                    if 'sysu' in f.lower(): 
                        continue
                    
                    # Skip eval files
                    if 'eval' in f.lower() or 'pumch' in f.lower() or 'pwh' in f.lower() or 'tss' in f.lower():
                        continue

                    with open(os.path.join(src_dir, f), 'r') as file:
                        content = file.read()
                    
                    new_content = process_content(content, filename=f)
                    
                    # Rename file
                    new_filename = f.replace('pitvis', 'sysu_skill').replace('PitVis', 'SYSU_Skill')
                    # If filename didn't contain pitvis, append _sysu_skill
                    if new_filename == f:
                        name, ext = os.path.splitext(f)
                        new_filename = f"{name}_sysu_skill{ext}"
                    
                    dst_path = os.path.join(dst_dir, new_filename)
                    with open(dst_path, 'w') as file:
                        file.write(new_content)
                    print(f"  Generated {dst_path}")
            
        else:
            # Case 2: Flat structure
            pitvis_files = [f for f in files if 'pitvis' in f.lower() and f.endswith('.yaml')]
            
            for f in pitvis_files:
                if 'sysu' in f.lower():
                    continue
                if 'eval' in f.lower() or 'pumch' in f.lower() or 'pwh' in f.lower() or 'tss' in f.lower():
                    continue

                src_path = os.path.join(root, f)
                with open(src_path, 'r') as file:
                    content = file.read()
                
                new_content = process_content(content, filename=f)
                new_filename = f.replace('pitvis', 'sysu_skill').replace('PitVis', 'SYSU_Skill')
                if new_filename == f:
                    name, ext = os.path.splitext(f)
                    new_filename = f"{name}_sysu_skill{ext}"
                
                dst_path = os.path.join(root, new_filename)
                
                if dst_path != src_path:
                    with open(dst_path, 'w') as file:
                        file.write(new_content)
                    print(f"  Generated {dst_path}")

if __name__ == '__main__':
    main()
