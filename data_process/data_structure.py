import os

def print_folder_tree(dir_path, prefix="", is_root=True, skip_keywords=("clip", "frame")):
    """
    打印文件夹树结构：
    ✅ 显示文件夹层级
    ✅ 显示 CSV 文件
    🚫 不展开 clip / frame 文件夹
    """
    if is_root:
        print(f"📁 {os.path.basename(dir_path)}")

    try:
        entries = sorted(os.listdir(dir_path))
    except PermissionError:
        print(prefix + "🚫 [No access]")
        return

    # 分离文件夹与文件
    folders = [e for e in entries if os.path.isdir(os.path.join(dir_path, e))]
    files = [e for e in entries if os.path.isfile(os.path.join(dir_path, e))]

    for i, folder in enumerate(folders):
        path = os.path.join(dir_path, folder)
        connector = "└── " if i == len(folders) - 1 and not files else "├── "
        print(f"{prefix}{connector}📂 {folder}")

        # 如果包含 clip 或 frame，不去递归展开
        if any(k.lower() in folder.lower() for k in skip_keywords):
            continue

        # 递归文件夹
        new_prefix = prefix + ("    " if i == len(folders) - 1 and not files else "│   ")
        print_folder_tree(path, new_prefix, is_root=False, skip_keywords=skip_keywords)

    # 显示当前目录下的 CSV 文件
    csv_files = [f for f in files if f.lower().endswith(".csv")]
    for j, f in enumerate(csv_files):
        connector = "└── " if j == len(csv_files) - 1 else "├── "
        print(f"{prefix}{connector}🧾 {f}")


if __name__ == "__main__":
    root = input("请输入要查看的目录路径：").strip()
    if not os.path.exists(root):
        print("❌ 路径不存在，请检查后重试。")
    else:
        print_folder_tree(root)

