"""
Tar 包图像读取工具。
支持通过 tar URI（tar:///path/to/file.tar::inner/path.jpg）直接从 tar 包中读取图像，
避免解压到磁盘，节省存储空间。
"""
import tarfile
import io
from pathlib import Path
from typing import Optional
from PIL import Image
from functools import lru_cache

# tar URI 分隔符：tar:///path/to/archive.tar::inner/relative/path.jpg
TAR_URI_PREFIX = "tar://"
TAR_URI_SEPARATOR = "::"


def is_tar_uri(path: str) -> bool:
    """判断路径是否为 tar URI 格式。"""
    return path.startswith(TAR_URI_PREFIX)


def make_tar_uri(tar_path: str, inner_path: str) -> str:
    """
    构建 tar URI。

    Args:
        tar_path: tar 文件的绝对路径
        inner_path: tar 内部的相对路径

    Returns:
        tar URI 字符串，如 "tar:///data/images/data_000000.tar::珐琅/xxx.jpg"
    """
    return f"{TAR_URI_PREFIX}{tar_path}{TAR_URI_SEPARATOR}{inner_path}"


def parse_tar_uri(uri: str) -> tuple[str, str]:
    """
    解析 tar URI，返回 (tar_path, inner_path)。

    Args:
        uri: tar URI 字符串

    Returns:
        (tar 文件路径, tar 内部相对路径) 元组
    """
    body = uri[len(TAR_URI_PREFIX):]
    tar_path, inner_path = body.split(TAR_URI_SEPARATOR, 1)
    return tar_path, inner_path


# ---- 带 LRU 缓存的 TarFile 句柄管理 ----
# 避免同一个 tar 文件被反复打开/关闭，大幅提升批量读取性能

@lru_cache(maxsize=64)
def _get_tar_handle(tar_path: str) -> tarfile.TarFile:
    """获取缓存的 TarFile 句柄（只读模式）。"""
    return tarfile.open(tar_path, "r")


def load_image(path: str) -> Optional[Image.Image]:
    """
    统一的图像加载接口：自动判断是普通路径还是 tar URI，返回 PIL Image。

    Args:
        path: 图像路径（普通文件路径或 tar URI）

    Returns:
        PIL Image 对象，加载失败时返回 None
    """
    try:
        if is_tar_uri(path):
            tar_path, inner_path = parse_tar_uri(path)
            tf = _get_tar_handle(tar_path)
            member = tf.getmember(inner_path)
            file_obj = tf.extractfile(member)
            if file_obj is None:
                return None
            image_data = file_obj.read()
            return Image.open(io.BytesIO(image_data)).convert("RGB")
        else:
            return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[WARNING] 无法加载图像 {path}: {e}")
        return None


def load_image_bytes(path: str) -> Optional[bytes]:
    """
    加载图像的原始字节数据（用于 Streamlit st.image 等场景，避免二次编解码）。

    Args:
        path: 图像路径（普通文件路径或 tar URI）

    Returns:
        图像的原始字节数据，失败时返回 None
    """
    try:
        if is_tar_uri(path):
            tar_path, inner_path = parse_tar_uri(path)
            tf = _get_tar_handle(tar_path)
            member = tf.getmember(inner_path)
            file_obj = tf.extractfile(member)
            if file_obj is None:
                return None
            return file_obj.read()
        else:
            return Path(path).read_bytes()
    except Exception:
        return None
