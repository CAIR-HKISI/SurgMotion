import os

# configs = [
#     "autolaparo_probe_attentive_64f.yaml",
#     "egosurgery_probe_attentive_64f.yaml",
#     "pmlr50_probe_attentive_64f.yaml",
#     "aIxsuture-5s_probe_attentive_64f.yaml",
#     "avos_probe_attentive_64f.yaml",
#     "polypdiag_probe_attentive_64f.yaml",
#     "surgicalactions160-25fps_probe_attentive_64f.yaml"
# ]


# global_vars = {
#     "CKPTL_DIR": "logs/cpt_vitl-256px-16f_100epoch/cooldown-e40_vitl-256px-e400/",
#     "CKPT_EPOCH": "latest.pt",
#     "MODEL_NAME": "vit_large",
#     "timestamp": "1203"
# }

# global_vars = {
#     "CKPTL_DIR": "logs/cpt_vitg-256px-16f_100epoch/cooldown-e40_vitg-xformers-256px-64f-e600",
#     "CKPT_EPOCH": "latest.pt",
#     "MODEL_NAME": "vit_giant_xformers",
#     "timestamp": "1203"
# }

# global_vars = {
#     "CKPTL_DIR": "logs/cpt_vitg-384px-16f_100epoch/cooldown-e40_vitg-384px-64f-e100",
#     "CKPT_EPOCH": "latest.pt",
#     "MODEL_NAME": "vit_giant",
#     "timestamp": "1205",
#     "TASK": "probing_cb_softmax"
# }


# global_vars = {
#     "CKPTL_DIR": "logs/cpt_vitl-256px-16f_100epoch/cooldown-e40_vitl-256px-e200",
#     "CKPT_EPOCH": "latest.pt",
#     "MODEL_NAME": "vit_large",
#     "timestamp": "1205",
#     "TASK": "probing_cb_softmax"
# }

# global_vars = {
#     "CKPTL_DIR": "logs/cpt_vitl-256px-16f_100epoch/cooldown-e40_vitl-256px-e300",
#     "CKPT_EPOCH": "latest.pt",
#     "MODEL_NAME": "vit_large",
#     "timestamp": "1205",
#     "TASK": "probing_cb_softmax"
# }

# global_vars = {
#     "CKPTL_DIR": "logs/cpt_vitl-256px-16f_100epoch/cooldown-e40_vitl-256px-e100",
#     "CKPT_EPOCH": "latest.pt",
#     "MODEL_NAME": "vit_large",
#     "timestamp": "1205",
#     "TASK": "probing_cb_softmax"
# }

# global_vars = {
#     "CKPTL_DIR": "logs/cpt_vitg-256px-16f_100epoch/cooldown-e40_vitg-xformers-256px-64f-e200",
#     "CKPT_EPOCH": "latest.pt",
#     "MODEL_NAME": "vit_giant_xformers",
#     "timestamp": "1205",
#     "TASK": "probing_cb_softmax"
# }



# global_vars = {
#     "CKPTL_DIR": "logs/cpt_vitg-256px-16f_100epoch/cooldown-e40_vitg-xformers-256px-64f-e200",
#     "CKPT_EPOCH": "latest.pt",
#     "MODEL_NAME": "vit_giant",
#     "timestamp": "1205",
#     "TASK": "probing_cb_softmax-0.99999"
# }

# global_vars = {
#     "CKPTL_DIR": "logs/cpt_vitg-256px-16f_100epoch/cooldown-e40_vitg-xformers-256px-64f-e200",
#     "CKPT_EPOCH": "latest.pt",
#     "MODEL_NAME": "vit_giant",
#     "timestamp": "1205",
#     "TASK": "probing_cb_focal"
# }



# global_vars = {
#     "CKPTL_DIR": "logs/pred-motion-v3_vitg-256px-64f_motion-weight-5",
#     "CKPT_EPOCH": "latest.pt",
#     "MODEL_NAME": "vit_giant_xformers",
#     "timestamp": "1209",
#     "TASK": "probing_pred_motion"
# }


# global_vars = {
#     "CKPTL_DIR": "logs/pred-motion-v3-multi-scale_vitg-256px-64f",
#     "CKPT_EPOCH": "latest.pt",
#     "MODEL_NAME": "vit_giant_xformers",
#     "timestamp": "1209",
#     "TASK": "probing_pred_motion"
# }

# global_vars = {
#     "CKPTL_DIR": "logs/pred-motion-v3_vitg-256px-64f-motion-weight-1",
#     "CKPT_EPOCH": "latest.pt",
#     "MODEL_NAME": "vit_giant_xformers",
#     "timestamp": "1209",
#     "TASK": "probing_pred_motion"
# }

# global_vars = {
#     "CKPTL_DIR": "logs/pred-motion-v3_vitg-256px-64f_jepaloss-l2",
#     "CKPT_EPOCH": "latest.pt",
#     "MODEL_NAME": "vit_giant_xformers",
#     "timestamp": "1209",
#     "TASK": "probing_pred_motion"
# }

# global_vars = {
#     "CKPTL_DIR": "pred-motion-v3_vitg-256px-64f_motion-weight-10",
#     "CKPT_EPOCH": "latest.pt",
#     "MODEL_NAME": "vit_giant_xformers",
#     "timestamp": "1209",
#     "TASK": "probing_pred_motion"
# }


configs = [
    "m2cai_probe_attentive_64f.yaml",
    "atlas_probe_attentive_64f.yaml",
    "cholec80_probe_attentive_64f.yaml",
    "jigsaws_probe_attentive_64f.yaml",
    "ophnet_probe_attentive_64f.yaml",
    "pitvis_probe_attentive_64f.yaml",
]

global_vars = {
    "CKPTL_DIR": "logs/pred-motion-v3_vitg-256px-64f_jepaloss-l2",
    "CKPT_EPOCH": "latest.pt",
    "MODEL_NAME": "vit_giant_xformers",
    "timestamp": "1210",
    "TASK": "probing_pred_motion"
}


task = global_vars["TASK"]
ckptl_name = os.path.basename(global_vars["CKPTL_DIR"].rstrip("/"))

source_script = "scripts_wjl/run_probing.sh"
output_dir = f"scripts_wjl/{task}_{global_vars['timestamp']}_{ckptl_name}"


# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

with open(source_script, 'r') as f:
    lines = f.readlines()

# 分离头部（Shebang + SBATCH）和主体
header = []
body = []
in_header = True
skip_check = False

for line in lines:
    if in_header:
        if line.startswith("#!") or line.startswith("#SBATCH") or line.strip() == "":
            header.append(line)
        else:
            in_header = False
            # Start of body processing for the first line of body
            # Check if this is the start of the check block
            if "if [ -z \"$FNAME\" ]" in line:
                skip_check = True
            
            if not skip_check:
                body.append(line)
            elif line.strip() == "fi":
                skip_check = False
    else:
        # Continue body processing
        if "if [ -z \"$FNAME\" ]" in line:
            skip_check = True
        
        if not skip_check:
            body.append(line)
        
        if skip_check and line.strip() == "fi":
            skip_check = False

for config in configs:
    data_name = config.split('_')[0]
    job_name = f"prb_{data_name}"
    filename = f"submit_{data_name}.sh"
    filepath = os.path.join(output_dir, filename)
    
    new_content = []
    
    # 1. Shebang
    if header and header[0].startswith("#!"):
        new_content.append(header[0])
    else:
        new_content.append("#!/bin/bash\n")
        
    # 2. Job Name (inserted early)
    new_content.append(f"#SBATCH --job-name={job_name}\n")
    
    # 3. Rest of header
    for line in header:
        if not line.startswith("#!"):
            new_content.append(line)
            
    # 5. Variables (WITHOUT export)
    new_content.append("\n# ========================\n")
    new_content.append("# 任务特定配置\n")
    new_content.append("# ========================\n")
    new_content.append(f'FNAME="{config}"\n')
    new_content.append(f'TASK="{task}"\n')
    for k, v in global_vars.items():
        if k == "TASK":
            continue
        new_content.append(f'{k}="{v}"\n')
    new_content.append(f'CKPTL_NAME="{ckptl_name}"\n')
        
    # 6. Body (with inserted commands)
    for line in body:
        new_content.append(line)
        # 插入 folder 和 checkpoint 定义
        if line.strip().startswith('LOG_FILE='):
            new_content.append('\nfolder="${CKPTL_DIR}/${DATA_NAME}"\n')
            new_content.append('checkpoint="${CKPTL_DIR}/${CKPT_EPOCH}"\n')
            new_content.append('\n')
            new_content.append('mkdir -p "${folder}"\n')
            new_content.append('mkdir -p "$(dirname "${LOG_FILE}")"\n')
            new_content.append('\n')
    
    with open(filepath, 'w') as f:
        f.writelines(new_content)
    
    # Make executable
    os.chmod(filepath, 0o755)

print(f"Generated {len(configs)} scripts in {output_dir}")
