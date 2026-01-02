import os
import re

ROOT_DIR = 'configs/fdtn_probing'
TRAIN_CSV = 'data/Surge_Frames/Private_SYSU_Brochiscopy_labeled/step_train_metadata.csv'
TEST_CSV = 'data/Surge_Frames/Private_SYSU_Brochiscopy_labeled/step_test_metadata.csv'
NUM_CLASSES = 10
DATASET_TAG = 'SYSU'
DATASET_NAME_LOWER = 'sysu'

def process_content(content):
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
        if os.path.basename(root) in ['PitVis', 'SYSU']:
            continue
        
        # Skip Private-* folders (e.g. for surgenet where these are different surgeries)
        if 'Private-' in os.path.basename(root):
            continue

        # Case 1: Standard structure with PitVis subfolder
        if 'PitVis' in dirs:
            print(f"Found PitVis folder in {root}")
            src_dir = os.path.join(root, 'PitVis')
            dst_dir = os.path.join(root, 'SYSU')
            os.makedirs(dst_dir, exist_ok=True)
            
            for f in os.listdir(src_dir):
                if f.endswith('.yaml'):
                    # Skip if it's already a generated sysu file (just in case)
                    if 'sysu' in f.lower(): 
                        continue
                    
                    # Skip files related to cross-dataset evaluation
                    if 'eval' in f.lower() or 'pumch' in f.lower() or 'pwh' in f.lower() or 'tss' in f.lower():
                        continue

                    with open(os.path.join(src_dir, f), 'r') as file:
                        content = file.read()
                    
                    new_content = process_content(content)
                    
                    # Rename file
                    new_filename = f.replace('pitvis', 'sysu').replace('PitVis', 'SYSU')
                    # If filename didn't contain pitvis, append _sysu
                    if new_filename == f:
                        name, ext = os.path.splitext(f)
                        new_filename = f"{name}_sysu{ext}"
                    
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
                if 'sysu' in f.lower():
                    continue

                # Skip files related to cross-dataset evaluation
                if 'eval' in f.lower() or 'pumch' in f.lower() or 'pwh' in f.lower() or 'tss' in f.lower():
                    continue

                src_path = os.path.join(root, f)
                with open(src_path, 'r') as file:
                    content = file.read()
                
                new_content = process_content(content)
                new_filename = f.replace('pitvis', 'sysu').replace('PitVis', 'SYSU')
                 # If filename didn't contain pitvis (unlikely given filter), append _sysu
                if new_filename == f:
                    name, ext = os.path.splitext(f)
                    new_filename = f"{name}_sysu{ext}"
                
                dst_path = os.path.join(root, new_filename)
                
                # Avoid overwriting source
                if dst_path != src_path:
                    with open(dst_path, 'w') as file:
                        file.write(new_content)
                    print(f"  Generated {dst_path}")

if __name__ == '__main__':
    main()
