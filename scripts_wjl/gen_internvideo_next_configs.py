import os
import re

# Template configs (source) and output configs (destination)
SRC_BASE = "configs/fdtn_probing/dinov3"
DST_BASE = "configs/fdtn_probing/internvideo_next"

# InternVideoNext adapter expects `model_type: internvideo_next` and `encoder.model_name`
# (or explicit checkpoint path via `model_kwargs.checkpoint`).
MODEL_CFG = {
    "file_prefix": "internvideo_next",
    "model_name": "internvideo_next_large_p14_res224_f16",
    "description": "InternVideoNext",
    # Important: InternVideoNext returns per-frame tokens with tubelet_size=1 in our probing wrapper
    "tubelet_size": 1,
}


def _extract_dataset_suffix(dataset_name: str, src_filename: str) -> str:
    """
    从模板文件名中提取数据集后缀（用于生成目标文件名/日志tag等）。
    期望格式：dinov3_xxx_64f_<DatasetSuffix>.yaml
    """
    match = re.search(r"dinov3_.+_64f_(.+)\.yaml", src_filename)
    if match:
        return match.group(1)
    return dataset_name


def process_file(dataset_name: str, src_file: str):
    with open(src_file, "r") as f:
        content = f.read()

    base_name = os.path.basename(src_file)
    dataset_suffix = _extract_dataset_suffix(dataset_name, base_name)

    file_prefix = MODEL_CFG["file_prefix"]
    m_name = MODEL_CFG["model_name"]
    desc = MODEL_CFG["description"]
    tubelet_size = MODEL_CFG["tubelet_size"]

    new_content = content

    # 1) Update output/log identifiers
    # folder: logs/foundation/dinov3_xxx_<Dataset> -> logs/foundation/internvideo_next_<Dataset>
    new_content = re.sub(
        r"folder:\s*logs/foundation/dinov3_[^/\n]+",
        f"folder: logs/foundation/{file_prefix}_{dataset_suffix}",
        new_content,
    )

    # tag: dinov3_xxx_64f_<Dataset> -> internvideo_next_64f_<Dataset>
    new_content = re.sub(
        r"tag:\s*dinov3_[^\n]+",
        f"tag: {file_prefix}_64f_{dataset_suffix}",
        new_content,
    )

    # 2) Update WandB
    new_content = re.sub(
        r"name:\s*dinov3_[^\n]+",
        f"name: {file_prefix}_64f_{dataset_suffix}",
        new_content,
    )

    # wandb.tags third item: dinov3_xxx-64f -> internvideo_next-64f
    new_content = re.sub(
        # 把 tags 里类似 "- dinov3_vith16plus-64f" 或 "- dinov3-vitl-64f" 统一替换掉
        r"(?m)^\s+-\s*dinov3[^\n]*64f[^\n]*$",
        f"    - {file_prefix}-64f",
        new_content,
    )

    # notes: add model description
    new_content = re.sub(
        r'notes:\s*"Multi-head probing on ([^"]+)"',
        rf'notes: "Multi-head probing on \1 with {desc}"',
        new_content,
    )

    # 3) Update model_kwargs
    new_content = re.sub(r"model_type:\s*dinov3", "model_type: internvideo_next", new_content)

    # Replace encoder section (only model_name)
    encoder_pattern = r"  encoder:\n    model_name: [^\n]+"
    encoder_replacement = f"  encoder:\n    model_name: {m_name}"
    new_content = re.sub(encoder_pattern, encoder_replacement, new_content)

    # Ensure wrapper_kwargs.tubelet_size is set to 1 for InternVideoNext
    new_content = re.sub(
        r"wrapper_kwargs:\n\s+tubelet_size:\s*\d+",
        f"wrapper_kwargs:\n    tubelet_size: {tubelet_size}",
        new_content,
    )

    # 4) Save
    dst_dir = os.path.join(DST_BASE, dataset_name)
    os.makedirs(dst_dir, exist_ok=True)

    dst_filename = f"{file_prefix}_64f_{dataset_suffix}.yaml"
    dst_path = os.path.join(dst_dir, dst_filename)

    with open(dst_path, "w") as f:
        f.write(new_content)

    print(f"Generated: {dst_path}")


def main():
    if not os.path.exists(SRC_BASE):
        print(f"Source not found: {SRC_BASE}")
        return

    for item in os.listdir(SRC_BASE):
        src_dir = os.path.join(SRC_BASE, item)
        if not os.path.isdir(src_dir):
            continue

        yaml_files = [f for f in os.listdir(src_dir) if f.endswith(".yaml")]
        if not yaml_files:
            continue

        # Prefer ones with '64f' if multiple
        target_yaml = next((f for f in yaml_files if "64f" in f), yaml_files[0])
        process_file(item, os.path.join(src_dir, target_yaml))


if __name__ == "__main__":
    main()


