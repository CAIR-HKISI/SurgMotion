import os
import re

ROOT_DIR = 'configs/fdtn_probing'
# JIGSAWS config
TRAIN_CSV = 'data/Surge_Frames/JIGSAWS/clips_64f/train_dense_64f_detailed.csv'
TEST_CSV = 'data/Surge_Frames/JIGSAWS/clips_64f/test_dense_64f_detailed.csv'
DATASET_TAG = 'JIGSAWS'
DATASET_NAME_LOWER = 'jigsaws'

def process_content(content):
    # 0. Clean up deprecated fields
    # Remove metric_aggregation if present (deprecated in eval.py)
    if 'metric_aggregation' in content:
        content = re.sub(r'metric_aggregation:.*\n', '', content)

    # 1. Inject task_type: regression (assuming JIGSAWS is used for regression here)
    if 'task_type:' not in content:
        content = re.sub(r'(tasks_per_node: \d+)', r'\1\ntask_type: regression', content)
    else:
        content = re.sub(r'task_type:.*', 'task_type: regression', content)

    # 2. Replace dataset paths
    # Handle list format
    if re.search(r'dataset_train:\s*\n\s*-', content):
        content = re.sub(r'(dataset_train:\s*\n\s*- ).*', f'\\g<1>{TRAIN_CSV}', content)
    else:
        content = re.sub(r'dataset_train: .*', f'dataset_train: {TRAIN_CSV}', content)

    if re.search(r'dataset_val:\s*\n\s*-', content):
        content = re.sub(r'(dataset_val:\s*\n\s*- ).*', f'\\g<1>{TEST_CSV}', content)
    else:
        content = re.sub(r'dataset_val: .*', f'dataset_val: {TEST_CSV}', content)

    # 3. Replace num_classes
    # The reference uses list format:
    # num_classes:
    # - 1
    if re.search(r'num_classes:\s*\n\s*-', content):
        content = re.sub(r'(num_classes:\s*\n\s*- ).*', r'\1 1   # Placeholder for regression', content)
    else:
        # If it was single line, convert to list or just set to 1?
        # The reference file uses list format explicitly.
        # Let's try to match the style if possible, or just set `num_classes: 1` if the original was simple.
        # But wait, foundation models might expect specific format.
        # Let's just set it to 1.
        content = re.sub(r'num_classes: .*', 'num_classes: 1   # Placeholder for regression', content)

    # 4. Replace PitVis references
    content = content.replace('PitVis', DATASET_TAG)
    content = content.replace('pitvis', DATASET_NAME_LOWER)
    
    # 5. Add/Update specific fields from reference if missing
    # quick_run: false
    # quick_run_num_videos: 2
    
    return content

def main():
    if not os.path.exists(ROOT_DIR):
        print(f"Directory not found: {ROOT_DIR}")
        return

    for root, dirs, files in os.walk(ROOT_DIR):
        # SKIP processing if we are currently INSIDE a PitVis, SYSU, SYSU_Skill, or JIGSAWS folder
        basename = os.path.basename(root)
        if basename in ['PitVis', 'SYSU', 'SYSU_Skill', 'JIGSAWS']:
            continue
            
        # Skip Private-* folders
        if 'Private-' in basename:
            continue

        # Check if PitVis folder exists in current dir (Use PitVis as template source)
        if 'PitVis' in dirs:
            print(f"Found PitVis folder in {root}")
            src_dir = os.path.join(root, 'PitVis')
            dst_dir = os.path.join(root, 'JIGSAWS')
            os.makedirs(dst_dir, exist_ok=True)
            
            for f in os.listdir(src_dir):
                if f.endswith('.yaml'):
                    # Skip if it's already a generated sysu/jigsaws file
                    if 'sysu' in f.lower() or 'jigsaws' in f.lower(): 
                        continue
                    
                    # Skip eval files
                    if 'eval' in f.lower() or 'pumch' in f.lower() or 'pwh' in f.lower() or 'tss' in f.lower():
                        continue

                    with open(os.path.join(src_dir, f), 'r') as file:
                        content = file.read()
                    
                    new_content = process_content(content)
                    
                    # Rename file
                    new_filename = f.replace('pitvis', 'jigsaws').replace('PitVis', 'jigsaws')
                    if new_filename == f:
                        name, ext = os.path.splitext(f)
                        new_filename = f"{name}_jigsaws{ext}"
                    
                    dst_path = os.path.join(dst_dir, new_filename)
                    with open(dst_path, 'w') as file:
                        file.write(new_content)
                    print(f"  Generated {dst_path}")
            
        else:
            # Case 2: Flat structure
            pitvis_files = [f for f in files if 'pitvis' in f.lower() and f.endswith('.yaml')]
            
            for f in pitvis_files:
                if 'sysu' in f.lower() or 'jigsaws' in f.lower():
                    continue
                if 'eval' in f.lower() or 'pumch' in f.lower() or 'pwh' in f.lower() or 'tss' in f.lower():
                    continue

                src_path = os.path.join(root, f)
                with open(src_path, 'r') as file:
                    content = file.read()
                
                new_content = process_content(content)
                new_filename = f.replace('pitvis', 'jigsaws').replace('PitVis', 'jigsaws')
                if new_filename == f:
                    name, ext = os.path.splitext(f)
                    new_filename = f"{name}_jigsaws{ext}"
                
                dst_path = os.path.join(root, new_filename)
                
                if dst_path != src_path:
                    with open(dst_path, 'w') as file:
                        file.write(new_content)
                    print(f"  Generated {dst_path}")

if __name__ == '__main__':
    main()

