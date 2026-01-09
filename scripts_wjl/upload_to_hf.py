import os
from pathlib import Path
from huggingface_hub import HfApi, login

# ============ 配置区域 ============
# Hugging Face 用户名或组织名
HF_USERNAME = "CAIR-HKISI" 

# Hugging Face Token
# 建议: 优先读取环境变量，如果环境变量没有，再使用硬编码的默认值
# 这里的 "xxx" 只是占位符，如果环境变量没设，代码会提示输入
HF_TOKEN = "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" 

# 仓库名称
REPO_NAME = "Surgical_Video_Dataset"

# 本地文件路径
LOCAL_FILE = "data/Private_KCH_Colonoscopy_labeled.tar"

# 仓库类型: "dataset" 或 "model"
REPO_TYPE = "dataset"

# 是否为私有仓库
PRIVATE = True
# ==================================


def main():
    # 1. 处理 Token
    token = HF_TOKEN
    # 如果 Token 是占位符或者为空，尝试交互式输入
    if not token or token == "xxx":
        print("未检测到有效 Token。")
        print("请输入 Hugging Face Token (在 https://huggingface.co/settings/tokens 获取):")
        token = input("Token: ").strip()
    
    if not token:
        print("错误: 未提供 Token，程序退出。")
        return

    # 2. 登录 Hugging Face
    try:
        # add_to_git_credential=True 可以帮助在本地 git 操作时免密
        login(token=token, add_to_git_credential=True)
        print("✅ 登录成功!")
    except Exception as e:
        print(f"❌ 登录失败: {e}")
        return

    # 3. 检查文件是否存在
    local_path = Path(LOCAL_FILE)
    if not local_path.exists():
        print(f"❌ 错误: 本地文件不存在 - {local_path.absolute()}")
        return

    file_size_gb = local_path.stat().st_size / (1024**3)
    print(f"📦 准备上传文件: {LOCAL_FILE}")
    print(f"📊 文件大小: {file_size_gb:.2f} GB")

    # 4. 初始化 API
    api = HfApi()

    # 仓库 ID
    repo_id = f"{HF_USERNAME}/{REPO_NAME}"

    # 5. 创建仓库（如果不存在）
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type=REPO_TYPE,
            private=PRIVATE,
            exist_ok=True,
        )
        print(f"✅ 仓库已就绪: {repo_id} (私有: {PRIVATE})")
    except Exception as e:
        print(f"❌ 创建仓库时出错: {e}")
        return

    # 6. 上传文件
    print(f"🚀 开始上传到 {repo_id}...")
    print("⏳ 这可能需要一些时间，取决于网络速度...")

    try:
        # upload_file 会自动处理大文件 (LFS)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=local_path.name, # 文件在仓库中的名字
            repo_id=repo_id,
            repo_type=REPO_TYPE,
        )
        print(f"\n🎉 上传成功!")
        
        # 根据仓库类型生成正确的 URL
        url_prefix = "datasets/" if REPO_TYPE == "dataset" else ""
        print(f"🔗 访问链接: https://huggingface.co/{url_prefix}{repo_id}")
        
    except Exception as e:
        print(f"\n❌ 上传失败: {e}")
        return


if __name__ == "__main__":
    main()