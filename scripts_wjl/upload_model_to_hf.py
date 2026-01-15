import os
import ssl
import urllib3
from pathlib import Path

# ============ 启用 hf_transfer 加速上传 ============
# hf_transfer 是基于 Rust 的高速传输工具，支持并行传输
# 安装: pip install hf_transfer
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

# ============ 禁用 SSL 验证 (解决企业网络 SSL 证书问题) ============
# 必须在导入 requests 之前设置环境变量
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'

# 禁用 urllib3 的 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 修改默认 SSL 上下文
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# Monkey-patch requests 库来禁用 SSL 验证
import requests
from requests.adapters import HTTPAdapter

# 保存原始方法
_original_request = requests.Session.request

def _patched_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return _original_request(self, method, url, **kwargs)

# 应用 patch
requests.Session.request = _patched_request

# 同时 patch requests 模块级别的函数
_original_get = requests.get
_original_post = requests.post
_original_head = requests.head
_original_put = requests.put

def _patched_get(url, **kwargs):
    kwargs['verify'] = False
    return _original_get(url, **kwargs)

def _patched_post(url, **kwargs):
    kwargs['verify'] = False
    return _original_post(url, **kwargs)

def _patched_head(url, **kwargs):
    kwargs['verify'] = False
    return _original_head(url, **kwargs)

def _patched_put(url, **kwargs):
    kwargs['verify'] = False
    return _original_put(url, **kwargs)

requests.get = _patched_get
requests.post = _patched_post
requests.head = _patched_head
requests.put = _patched_put

from huggingface_hub import HfApi, login
import shutil
import tempfile

# 检查 hf_transfer 是否可用
try:
    import hf_transfer
    HF_TRANSFER_AVAILABLE = True
except ImportError:
    HF_TRANSFER_AVAILABLE = False

# ============ 配置区域 ============
# Hugging Face 用户名或组织名
HF_USERNAME = "CAIR-HKISI"

# Hugging Face Token
HF_TOKEN = "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"

# 仓库名称 (模型仓库)
REPO_NAME = "NSJepa"

# 本地模型文件路径
LOCAL_FILE = "/home/projects/med-multi-llm/jinlin_wu/NSJepa_20251112/logs/cooldown_vitg-256px-64f_40epoch/latest.pt"

# 仓库中的文件名 (可以和本地文件名不同)
# 如果设置为 None，则使用本地文件的原名
REMOTE_FILENAME = "cooldown_vitg-256px-64f_40epoch.pt"

# 仓库类型: "model" 用于模型
REPO_TYPE = "model"

# 是否为私有仓库
PRIVATE = True
# ==================================


def main():
    # 1. 处理 Token
    token = HF_TOKEN
    if not token or token == "xxx":
        print("未检测到有效 Token。")
        print("请输入 Hugging Face Token (在 https://huggingface.co/settings/tokens 获取):")
        token = input("Token: ").strip()
    
    if not token:
        print("错误: 未提供 Token，程序退出。")
        return

    # 2. 登录 Hugging Face
    try:
        login(token=token, add_to_git_credential=True)
        print("✅ 登录成功!")
    except Exception as e:
        print(f"❌ 登录失败: {e}")
        return

    # 检查 hf_transfer 状态
    if HF_TRANSFER_AVAILABLE:
        print("⚡ hf_transfer 已启用 (高速传输模式)")
    else:
        print("⚠️  hf_transfer 未安装，使用默认上传模式")
        print("💡 安装方法: pip install hf_transfer")

    # 3. 检查文件是否存在
    local_path = Path(LOCAL_FILE)
    if not local_path.exists():
        print(f"❌ 错误: 本地文件不存在 - {local_path.absolute()}")
        return

    file_size_mb = local_path.stat().st_size / (1024**2)
    file_size_gb = local_path.stat().st_size / (1024**3)
    
    print(f"📦 准备上传模型: {LOCAL_FILE}")
    if file_size_gb >= 1:
        print(f"📊 文件大小: {file_size_gb:.2f} GB")
    else:
        print(f"📊 文件大小: {file_size_mb:.2f} MB")

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

    # 确定在仓库中的文件名
    path_in_repo = REMOTE_FILENAME if REMOTE_FILENAME else local_path.name

    # 6. 使用 upload_folder 上传 (支持断点续传和多线程)
    print(f"🚀 开始上传到 {repo_id}...")
    print(f"📄 仓库中的文件名: {path_in_repo}")
    print("⏳ 使用多线程上传，支持断点续传...")

    # 创建临时目录，将文件软链接到目标名称
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        # 创建软链接或复制文件到临时目录，使用目标文件名
        link_path = tmp_path / path_in_repo
        
        try:
            # 优先使用软链接（节省空间和时间）
            os.symlink(local_path.absolute(), link_path)
        except (OSError, NotImplementedError):
            # 如果软链接失败，则复制文件
            print("📋 正在准备文件...")
            shutil.copy2(local_path, link_path)

        try:
            # upload_folder 支持多线程和断点续传
            api.upload_folder(
                folder_path=str(tmp_path),
                repo_id=repo_id,
                repo_type=REPO_TYPE,
                multi_commits=True,           # 大文件分多次提交，支持断点续传
                multi_commits_verbose=True,   # 显示进度
            )
            print(f"\n🎉 上传成功!")
            print(f"🔗 访问链接: https://huggingface.co/{repo_id}")
            print(f"📥 下载链接: https://huggingface.co/{repo_id}/resolve/main/{path_in_repo}")
            
        except Exception as e:
            print(f"\n❌ 上传失败: {e}")
            if "401" in str(e) or "403" in str(e):
                print("💡 提示: 请检查 Token 是否正确，以及是否有该仓库的写入权限。")
            return


if __name__ == "__main__":
    main()
