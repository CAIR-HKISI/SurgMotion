import os
import ssl
import urllib3
from pathlib import Path
import tempfile
import hashlib

# ============ 启用 hf_transfer 加速上传 ============
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

# ============ 禁用 SSL 验证 (解决企业网络 SSL 证书问题) ============
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

import requests
from requests.adapters import HTTPAdapter

_original_request = requests.Session.request

def _patched_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return _original_request(self, method, url, **kwargs)

requests.Session.request = _patched_request

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

try:
    import hf_transfer
    HF_TRANSFER_AVAILABLE = True
except ImportError:
    HF_TRANSFER_AVAILABLE = False

# ============ 配置区域 ============
HF_USERNAME = "CAIR-HKISI"
HF_TOKEN = "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"
REPO_NAME = "NSJepa"

# 本地模型文件路径
LOCAL_FILE = "/home/projects/med-multi-llm/jinlin_wu/NSJepa_20251112/logs/cooldown_vitg-256px-64f_40epoch/latest.pt"

# 仓库中的文件名 (可以和本地文件名不同)
REMOTE_FILENAME = "cooldown_vitg-256px-64f_40epoch.pt"

# 仓库类型
REPO_TYPE = "model"
PRIVATE = True

# ============ 分卷配置 ============
# 分卷大小 (MB)，设为 190MB 以确保低于 200MB 限制
CHUNK_SIZE_MB = 190
# 临时分卷目录 (设为 None 则使用系统临时目录)
TEMP_SPLIT_DIR = None
# 上传后是否删除临时分卷文件
CLEANUP_AFTER_UPLOAD = True
# ==================================


def split_file(file_path: Path, chunk_size_mb: int, output_dir: Path) -> list:
    """
    将大文件分割成多个小分卷
    返回分卷文件路径列表
    """
    chunk_size = chunk_size_mb * 1024 * 1024  # 转换为字节
    file_size = file_path.stat().st_size
    num_chunks = (file_size + chunk_size - 1) // chunk_size
    
    base_name = file_path.name
    chunks = []
    
    print(f"📦 正在分割文件为 {num_chunks} 个分卷 (每卷 {chunk_size_mb}MB)...")
    
    with open(file_path, 'rb') as f:
        for i in range(num_chunks):
            chunk_name = f"{base_name}.part{i:03d}"
            chunk_path = output_dir / chunk_name
            
            # 如果分卷已存在且大小正确，跳过
            expected_size = min(chunk_size, file_size - i * chunk_size)
            if chunk_path.exists() and chunk_path.stat().st_size == expected_size:
                print(f"  ⏭️  分卷 {i+1}/{num_chunks} 已存在，跳过: {chunk_name}")
                chunks.append(chunk_path)
                f.seek(chunk_size, 1)  # 跳过这部分
                continue
            
            print(f"  ✂️  创建分卷 {i+1}/{num_chunks}: {chunk_name}")
            data = f.read(chunk_size)
            with open(chunk_path, 'wb') as chunk_file:
                chunk_file.write(data)
            chunks.append(chunk_path)
    
    # 创建元数据文件
    metadata_path = output_dir / f"{base_name}.metadata"
    with open(metadata_path, 'w') as f:
        f.write(f"original_filename={base_name}\n")
        f.write(f"total_size={file_size}\n")
        f.write(f"num_chunks={num_chunks}\n")
        f.write(f"chunk_size_mb={chunk_size_mb}\n")
        # 计算原始文件的 MD5 校验和
        print("🔐 计算文件校验和...")
        md5_hash = hashlib.md5()
        with open(file_path, 'rb') as orig_f:
            for block in iter(lambda: orig_f.read(8192 * 1024), b''):
                md5_hash.update(block)
        f.write(f"md5={md5_hash.hexdigest()}\n")
    
    chunks.append(metadata_path)
    print(f"✅ 文件分割完成，共 {num_chunks} 个分卷 + 1 个元数据文件")
    
    return chunks


def check_remote_exists(api: HfApi, repo_id: str, filename: str, repo_type: str) -> bool:
    """检查远程仓库中是否已存在某个文件"""
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
        return filename in files
    except Exception:
        return False


def main():
    token = HF_TOKEN
    if not token or token == "xxx":
        print("未检测到有效 Token。")
        print("请输入 Hugging Face Token:")
        token = input("Token: ").strip()
    
    if not token:
        print("错误: 未提供 Token，程序退出。")
        return

    try:
        login(token=token, add_to_git_credential=True)
        print("✅ 登录成功!")
    except Exception as e:
        print(f"❌ 登录失败: {e}")
        return

    if HF_TRANSFER_AVAILABLE:
        print("⚡ hf_transfer 已启用 (高速传输模式)")
    else:
        print("⚠️  hf_transfer 未安装")
        print("💡 安装方法: pip install hf_transfer")

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

    api = HfApi()
    repo_id = f"{HF_USERNAME}/{REPO_NAME}"

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

    path_in_repo = REMOTE_FILENAME if REMOTE_FILENAME else local_path.name

    # 判断是否需要分卷
    if file_size_mb <= 200:
        # 文件小于 200MB，直接上传
        print(f"🚀 文件小于 200MB，直接上传...")
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=REPO_TYPE,
            )
            print(f"\n🎉 上传成功!")
            print(f"🔗 访问链接: https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"\n❌ 上传失败: {e}")
            return
    else:
        # 需要分卷上传
        print(f"📦 文件超过 200MB 限制，将进行分卷上传 (每卷 {CHUNK_SIZE_MB}MB)...")
        
        # 确定分卷目录
        if TEMP_SPLIT_DIR:
            split_dir = Path(TEMP_SPLIT_DIR)
            split_dir.mkdir(parents=True, exist_ok=True)
        else:
            split_dir = Path(tempfile.gettempdir()) / "hf_upload_chunks"
            split_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📂 分卷目录: {split_dir}")
        
        # 分割文件
        chunks = split_file(local_path, CHUNK_SIZE_MB, split_dir)
        
        # 在仓库中创建子目录存放分卷
        remote_dir = path_in_repo.replace('.pt', '_chunks')
        
        print(f"\n🚀 开始上传 {len(chunks)} 个文件到 {repo_id}/{remote_dir}/...")
        
        success_count = 0
        for i, chunk_path in enumerate(chunks):
            chunk_name = chunk_path.name
            remote_path = f"{remote_dir}/{chunk_name}"
            
            # 检查是否已上传
            if check_remote_exists(api, repo_id, remote_path, REPO_TYPE):
                print(f"  ⏭️  [{i+1}/{len(chunks)}] 已存在，跳过: {chunk_name}")
                success_count += 1
                continue
            
            print(f"  📤 [{i+1}/{len(chunks)}] 上传中: {chunk_name}", end=" ", flush=True)
            
            try:
                api.upload_file(
                    path_or_fileobj=str(chunk_path),
                    path_in_repo=remote_path,
                    repo_id=repo_id,
                    repo_type=REPO_TYPE,
                )
                print("✅")
                success_count += 1
            except Exception as e:
                print(f"❌ 失败: {e}")
                print(f"\n⚠️  上传中断，已完成 {success_count}/{len(chunks)} 个文件")
                print("💡 重新运行脚本可以继续上传剩余分卷（支持断点续传）")
                return
        
        print(f"\n🎉 全部上传成功! ({success_count}/{len(chunks)} 个文件)")
        print(f"🔗 访问链接: https://huggingface.co/{repo_id}")
        print(f"📁 分卷目录: {remote_dir}/")
        
        # 清理临时文件
        if CLEANUP_AFTER_UPLOAD:
            print("\n🧹 清理临时分卷文件...")
            for chunk_path in chunks:
                try:
                    chunk_path.unlink()
                except Exception:
                    pass
            print("✅ 清理完成")


if __name__ == "__main__":
    main()
