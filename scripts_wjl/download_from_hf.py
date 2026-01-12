import os
import ssl
import urllib3
from pathlib import Path

# ============ 禁用 SSL 验证 (解决企业网络 SSL 证书问题) ============
# 方法1: 设置环境变量
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'

# 方法2: 禁用 urllib3 的 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 方法3: 修改默认 SSL 上下文 (如果上面方法不生效)
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

from huggingface_hub import hf_hub_download, login

# ============ 配置区域 ============
# Hugging Face 用户名或组织名
HF_USERNAME = "CAIR-HKISI"

# Hugging Face Token
# 建议: 优先读取环境变量，如果环境变量没有，再使用硬编码的默认值
HF_TOKEN = "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"

# 仓库名称
REPO_NAME = "Surgical_Video_Dataset"

# 仓库中的文件名 (即上传时在仓库里的名字)
REMOTE_FILENAME = "Private_KCH_Colonoscopy_labeled.tar"

# 本地保存目录
LOCAL_DIR = "data"

# 仓库类型: "dataset" 或 "model"
REPO_TYPE = "dataset"
# ==================================


def main():
    # 1. 处理 Token
    token = HF_TOKEN
    # 如果 Token 是占位符或者为空，尝试交互式输入
    if not token or token == "xxx":
        print("未检测到有效 Token。")
        print("请输入 Hugging Face Token (下载私有仓库需要):")
        token = input("Token: ").strip()
    
    # 2. 登录 Hugging Face (对于私有仓库是必须的)
    if token:
        try:
            # add_to_git_credential=True 可以帮助在本地 git 操作时免密
            login(token=token, add_to_git_credential=True)
            print("✅ 登录成功!")
        except Exception as e:
            print(f"❌ 登录失败: {e}")
            return

    # 3. 准备下载
    repo_id = f"{HF_USERNAME}/{REPO_NAME}"
    local_dir_path = Path(LOCAL_DIR)
    
    # 确保保存目录存在
    if not local_dir_path.exists():
        print(f"📂 创建目录: {local_dir_path.absolute()}")
        local_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"📦 准备从仓库 {repo_id} ({REPO_TYPE}) 下载文件: {REMOTE_FILENAME}")
    print(f"📥 保存位置: {local_dir_path.absolute()}")
    print("⏳ 开始下载 (支持断点续传)...")

    # 4. 执行下载
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=REMOTE_FILENAME,
            repo_type=REPO_TYPE,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False,  # False: 下载实际文件; True: 使用缓存软链接
            resume_download=True,          # 开启断点续传
            force_download=False,          # 如果本地已有且最新，则不重新下载
        )
        print(f"\n🎉 下载成功!")
        print(f"📄 文件路径: {downloaded_path}")
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        # 常见错误提示
        if "401" in str(e) or "403" in str(e):
            print("💡 提示: 请检查 Token 是否正确，以及是否有该仓库的访问权限。")
        elif "404" in str(e):
            print(f"💡 提示: 仓库 '{repo_id}' 或文件 '{REMOTE_FILENAME}' 不存在。")


if __name__ == "__main__":
    main()
