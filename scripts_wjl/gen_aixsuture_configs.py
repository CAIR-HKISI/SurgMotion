import os
import re

ROOT_DIR = 'configs/fdtn_probing'
TRAIN_CSV = 'data/Surge_Frames/AIxsuture_v1/train_clip_metadata.csv'
TEST_CSV = 'data/Surge_Frames/AIxsuture_v1/test_clip_metadata.csv'
NUM_CLASSES = 3
DATASET_TAG = 'AIxsuture'
DATASET_NAME_LOWER = 'aixsuture'

# 统一配置: 4 segments × 16 frames = 64 frames (与PitVis相同的总帧数)
NUM_SEGMENTS = 4
FRAMES_PER_CLIP = 16
BATCH_SIZE = 2
MAX_FRAMES = 128  # 需要 >= num_segments * frames_per_clip
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
    # dataset_train:
    # - path
    if re.search(r'dataset_train:\s*\n\s*-', content):
        content = re.sub(r'(dataset_train:\s*\n\s*- ).*', f'\\g<1>{TRAIN_CSV}', content)
    else:
        content = re.sub(r'dataset_train: .*', f'dataset_train: {TRAIN_CSV}', content)

    # dataset_val:
    # - path
    if re.search(r'dataset_val:\s*\n\s*-', content):
        content = re.sub(r'(dataset_val:\s*\n\s*- ).*', f'\\g<1>{TEST_CSV}', content)
    else:
        content = re.sub(r'dataset_val: .*', f'dataset_val: {TEST_CSV}', content)

    # 2. Replace num_classes
    # Handle list format
    # num_classes:
    # - 15
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

    # 3. Replace PitVis references
    content = content.replace('PitVis', DATASET_TAG)
    content = content.replace('pitvis', DATASET_NAME_LOWER)
    
    return content

def main():
    if not os.path.exists(ROOT_DIR):
        print(f"Directory not found: {ROOT_DIR}")
        return

    for root, dirs, files in os.walk(ROOT_DIR):
        # SKIP processing if we are currently INSIDE a PitVis or SYSU folder
        # We only want to process from the parent directory
        if os.path.basename(root) in ['PitVis', 'SYSU', DATASET_TAG]:
            continue
        
        # Skip Private-* folders (e.g. for surgenet where these are different surgeries)
        if 'Private-' in os.path.basename(root):
            continue

        # Case 1: Standard structure with PitVis subfolder
        if 'PitVis' in dirs:
            print(f"Found PitVis folder in {root}")
            src_dir = os.path.join(root, 'PitVis')
            dst_dir = os.path.join(root, DATASET_TAG)
            os.makedirs(dst_dir, exist_ok=True)
            
            for f in os.listdir(src_dir):
                if f.endswith('.yaml'):
                    # Skip if it's already a generated sysu file (just in case)
                    if DATASET_NAME_LOWER in f.lower(): 
                        continue
                    
                    # Skip files related to cross-dataset evaluation
                    if 'eval' in f.lower() or 'pumch' in f.lower() or 'pwh' in f.lower() or 'tss' in f.lower():
                        continue

                    with open(os.path.join(src_dir, f), 'r') as file:
                        content = file.read()
                    
                    new_content = process_content(content, filename=f)
                    
                    # Rename file
                    new_filename = f.replace('pitvis', DATASET_NAME_LOWER).replace('PitVis', DATASET_TAG)
                    # If filename didn't contain pitvis, append _sysu
                    if new_filename == f:
                        name, ext = os.path.splitext(f)
                        new_filename = f"{name}_{DATASET_NAME_LOWER}{ext}"
                    
                    dst_path = os.path.join(dst_dir, new_filename)
                    with open(dst_path, 'w') as file:
                        file.write(new_content)
                    print(f"  Generated {dst_path}")
            
        else:
            # Case 2: Flat structure (e.g. vjepa) where configs are in the model root
            # Only process if this directory does NOT have a PitVis folder (handled above)
            
            # Filter for pitvis yaml files
            pitvis_files = [f for f in files if 'pitvis' in f.lower() and f.endswith('.yaml')]
            
            for f in pitvis_files:
                # Double check we aren't picking up something we shouldn't
                if DATASET_NAME_LOWER in f.lower():
                    continue

                # Skip files related to cross-dataset evaluation
                if 'eval' in f.lower() or 'pumch' in f.lower() or 'pwh' in f.lower() or 'tss' in f.lower():
                    continue

                src_path = os.path.join(root, f)
                with open(src_path, 'r') as file:
                    content = file.read()
                
                new_content = process_content(content, filename=f)
                new_filename = f.replace('pitvis', DATASET_NAME_LOWER).replace('PitVis', DATASET_TAG)
                 # If filename didn't contain pitvis (unlikely given filter), append _sysu
                if new_filename == f:
                    name, ext = os.path.splitext(f)
                    new_filename = f"{name}_{DATASET_NAME_LOWER}{ext}"
                
                dst_path = os.path.join(root, new_filename)
                
                # Avoid overwriting source
                if dst_path != src_path:
                    with open(dst_path, 'w') as file:
                        file.write(new_content)
                    print(f"  Generated {dst_path}")

if __name__ == '__main__':
    main()
