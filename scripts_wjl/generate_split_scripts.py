import os

configs = [
    "autolaparo_probe_attentive_64f.yaml",
    "egosurgery_probe_attentive_64f.yaml",
    "pmlr50_probe_attentive_64f.yaml",
    "aIxsuture-5s_probe_attentive_64f.yaml",
    "avos_probe_attentive_64f.yaml",
    "polypdiag_probe_attentive_64f.yaml",
    "surgicalactions160-10fps_probe_attentive_64f.yaml"
]

global_vars = {
    "LOG_ROOT": '"logs"',
    "CKPTL_NAME": '"cooldown_vitl-256px-64f_21-dataset_40epoch"',
    "MODEL_NAME": '"vit_large"',
    "CKPT_EPOCH": '"latest.pt"'
}

source_script = "scripts_wjl/run_probing.sh"
output_dir = "scripts_wjl/probing_vitl_tasks"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

with open(source_script, 'r') as f:
    lines = f.readlines()

# 分离头部（Shebang + SBATCH）和主体
header = []
body = []
in_header = True
for line in lines:
    if in_header:
        if line.startswith("#!") or line.startswith("#SBATCH") or line.strip() == "":
            header.append(line)
        else:
            in_header = False
            body.append(line)
    else:
        body.append(line)

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
            
    # 4. Set Working Directory explicitly to project root (assuming script is in scripts_wjl/probing_vitl_tasks)
    new_content.append("\n# ========================\n")
    new_content.append("# 切换到项目根目录\n")
    new_content.append("# ========================\n")
    new_content.append('cd "$(dirname "$0")/../.."\n')
    new_content.append('echo "Current working directory: $(pwd)"\n')

    # 5. Variables
    new_content.append("\n# ========================\n")
    new_content.append("# 任务特定配置\n")
    new_content.append("# ========================\n")
    new_content.append(f'export FNAME="{config}"\n')
    for k, v in global_vars.items():
        new_content.append(f'export {k}={v}\n')
        
    # 6. Body (Original run_probing logic)
    new_content.extend(body)
    
    with open(filepath, 'w') as f:
        f.writelines(new_content)
    
    # Make executable
    os.chmod(filepath, 0o755)

print(f"Generated {len(configs)} scripts in {output_dir}")

