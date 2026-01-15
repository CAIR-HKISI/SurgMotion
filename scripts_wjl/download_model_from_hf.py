import os
import ssl
import urllib3
from pathlib import Path
import hashlib

# ============ 启用 hf_transfer 加速下载 ============
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

def _patched_get(url, **kwargs):
    kwargs['verify'] = False
    return _original_get(url, **kwargs)

def _patched_post(url, **kwargs):
    kwargs['verify'] = False
    return _original_post(url, **kwargs)

def _patched_head(url, **kwargs):
    kwargs['verify'] = False
    return _original_head(url, **kwargs)

requests.get = _patched_get
requests.post = _patched_post
requests.head = _patched_head

from huggingface_hub import hf_hub_download, HfApi, login

try:
    import hf_transfer
    HF_TRANSFER_AVAILABLE = True
except ImportError:
    HF_TRANSFER_AVAILABLE = False

# ============ 配置区域 ============
HF_USERNAME = "CAIR-HKISI"
HF_TOKEN = "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"
REPO_NAME = "NSJepa"

# 要下载的模型文件名 (不含 _chunks 后缀)
REMOTE_FILENAME = "cooldown_vitg-256px-64f_40epoch.pt"

# 本地保存目录
LOCAL_DIR = "checkpoints"

# 仓库类型
REPO_TYPE = "model"

# 下载后是否删除分卷文件 (仅在分卷下载模式下有效)
CLEANUP_CHUNKS = True
# ==================================


def parse_metadata(metadata_path: Path) -> dict:
    """解析元数据文件"""
    metadata = {}
    with open(metadata_path, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                metadata[key] = value
    return metadata


def merge_chunks(chunks_dir: Path, metadata: dict, output_path: Path) -> bool:
    """合并分卷文件"""
    num_chunks = int(metadata['num_chunks'])
    original_filename = metadata['original_filename']
    expected_size = int(metadata['total_size'])
    expected_md5 = metadata.get('md5', None)
    
    print(f"🔗 正在合并 {num_chunks} 个分卷...")
    
    with open(output_path, 'wb') as outfile:
        for i in range(num_chunks):
            chunk_name = f"{original_filename}.part{i:03d}"
            chunk_path = chunks_dir / chunk_name
            
            if not chunk_path.exists():
                print(f"❌ 缺少分卷: {chunk_name}")
                return False
            
            print(f"  📦 合并分卷 {i+1}/{num_chunks}: {chunk_name}")
            with open(chunk_path, 'rb') as chunk_file:
                outfile.write(chunk_file.read())
    
    # 验证文件大小
    actual_size = output_path.stat().st_size
    if actual_size != expected_size:
        print(f"❌ 文件大小不匹配! 期望: {expected_size}, 实际: {actual_size}")
        return False
    
    # 验证 MD5
    if expected_md5:
        print("🔐 验证文件完整性...")
        md5_hash = hashlib.md5()
        with open(output_path, 'rb') as f:
            for block in iter(lambda: f.read(8192 * 1024), b''):
                md5_hash.update(block)
        actual_md5 = md5_hash.hexdigest()
        
        if actual_md5 != expected_md5:
            print(f"❌ MD5 校验失败! 期望: {expected_md5}, 实际: {actual_md5}")
            return False
        print("✅ MD5 校验通过")
    
    return True


def main():
    token = HF_TOKEN
    if not token or token == "xxx":
        print("未检测到有效 Token。")
        print("请输入 Hugging Face Token (下载私有仓库需要):")
        token = input("Token: ").strip()
    
    if token:
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

    repo_id = f"{HF_USERNAME}/{REPO_NAME}"
    local_dir_path = Path(LOCAL_DIR)
    local_dir_path.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    
    # 检查是单文件还是分卷模式
    chunks_dir_name = REMOTE_FILENAME.replace('.pt', '_chunks')
    
    try:
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type=REPO_TYPE)
    except Exception as e:
        print(f"❌ 获取仓库文件列表失败: {e}")
        return
    
    # 检查是否存在分卷目录
    chunk_files = [f for f in repo_files if f.startswith(f"{chunks_dir_name}/")]
    
    if chunk_files:
        # 分卷下载模式
        print(f"📦 检测到分卷模式，共 {len(chunk_files)} 个文件")
        
        chunks_local_dir = local_dir_path / chunks_dir_name
        chunks_local_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📥 下载分卷到: {chunks_local_dir}")
        print("⏳ 开始下载 (支持断点续传)...")
        
        for i, remote_file in enumerate(sorted(chunk_files)):
            filename = remote_file.split('/')[-1]
            local_file = chunks_local_dir / filename
            
            # 检查是否已下载
            if local_file.exists():
                print(f"  ⏭️  [{i+1}/{len(chunk_files)}] 已存在，跳过: {filename}")
                continue
            
            print(f"  📥 [{i+1}/{len(chunk_files)}] 下载中: {filename}", end=" ", flush=True)
            
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=remote_file,
                    repo_type=REPO_TYPE,
                    local_dir=LOCAL_DIR,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    force_download=False,
                )
                print("✅")
            except Exception as e:
                print(f"❌ 失败: {e}")
                print(f"\n⚠️  下载中断，重新运行可继续下载")
                return
        
        print(f"\n✅ 所有分卷下载完成!")
        
        # 读取元数据
        metadata_file = chunks_local_dir / f"{REMOTE_FILENAME}.metadata"
        if not metadata_file.exists():
            print(f"❌ 缺少元数据文件: {metadata_file}")
            return
        
        metadata = parse_metadata(metadata_file)
        
        # 合并分卷
        output_path = local_dir_path / REMOTE_FILENAME
        if merge_chunks(chunks_local_dir, metadata, output_path):
            file_size_gb = output_path.stat().st_size / (1024**3)
            print(f"\n🎉 合并成功!")
            print(f"📄 文件路径: {output_path}")
            print(f"📊 文件大小: {file_size_gb:.2f} GB")
            
            # 清理分卷文件
            if CLEANUP_CHUNKS:
                print("\n🧹 清理分卷文件...")
                import shutil
                shutil.rmtree(chunks_local_dir)
                print("✅ 清理完成")
        else:
            print(f"\n❌ 合并失败!")
            return
    
    elif REMOTE_FILENAME in repo_files:
        # 单文件下载模式
        print(f"📦 检测到单文件模式")
        print(f"📥 下载到: {local_dir_path}")
        print("⏳ 开始下载 (支持断点续传)...")
        
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=REMOTE_FILENAME,
                repo_type=REPO_TYPE,
                local_dir=LOCAL_DIR,
                local_dir_use_symlinks=False,
                resume_download=True,
                force_download=False,
            )
            print(f"\n🎉 下载成功!")
            print(f"📄 文件路径: {downloaded_path}")
            
            file_size_gb = Path(downloaded_path).stat().st_size / (1024**3)
            print(f"📊 文件大小: {file_size_gb:.2f} GB")
            
        except Exception as e:
            print(f"\n❌ 下载失败: {e}")
            if "401" in str(e) or "403" in str(e):
                print("💡 提示: 请检查 Token 是否正确")
            elif "404" in str(e):
                print(f"💡 提示: 文件 '{REMOTE_FILENAME}' 不存在")
    
    else:
        print(f"❌ 未找到文件 '{REMOTE_FILENAME}' 或其分卷")
        print(f"📂 仓库中的文件列表:")
        for f in repo_files[:20]:
            print(f"    - {f}")
        if len(repo_files) > 20:
            print(f"    ... 共 {len(repo_files)} 个文件")


if __name__ == "__main__":
    main()
