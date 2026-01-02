import os
import re

ROOT_DIR = 'configs/fdtn_probing'
# Skill dataset config
TRAIN_CSV = 'data/Surge_Frames/Private_SYSU_Brochiscopy_labeled/skill_train_metadata.csv'
TEST_CSV = 'data/Surge_Frames/Private_SYSU_Brochiscopy_labeled/skill_test_metadata.csv'
NUM_CLASSES = 2
DATASET_TAG = 'SYSU_Skill'
DATASET_NAME_LOWER = 'sysu_skill'

def process_content(content):
    # 0. Set metric_aggregation to global
    # Find tasks_per_node: 1 and insert metric_aggregation after it if it doesn't exist
    if 'metric_aggregation' not in content:
        content = re.sub(r'(tasks_per_node: \d+)', r'\1\nmetric_aggregation: "global" # Options: "global", "per_video", or null (defaults to per_video)', content)
    else:
        # If it exists, update it
        content = re.sub(r'metric_aggregation:.*', 'metric_aggregation: "global"', content)
        
    # Also force batch_size to 1 as seen in AVOS config
    content = re.sub(r'batch_size: \d+', 'batch_size: 1', content)

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

    # 3. Replace PitVis/SYSU references
    # We are generating FROM PitVis templates, so replace PitVis/pitvis with SYSU_Skill/sysu_skill
    content = content.replace('PitVis', DATASET_TAG)
    content = content.replace('pitvis', DATASET_NAME_LOWER)
    
    # 4. Update folder/tag/name/wandb to reflect skill task
    # This might already be covered by step 3 if the template used 'PitVis' in those fields
    # But just in case, let's ensure 'sysu' becomes 'sysu_skill' if we are re-using sysu files?
    # Actually, we should stick to using PitVis templates as source for consistency with previous step.
    
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
                    
                    new_content = process_content(content)
                    
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
                
                new_content = process_content(content)
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

