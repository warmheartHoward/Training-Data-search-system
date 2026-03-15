"""
全局配置文件：集中管理模型路径、索引路径、推理参数等。
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    """模型相关配置"""
    # DINOv2 视觉模型
    vision_model_name: str = "facebook/dinov2-large"
    vision_embedding_dim: int = 1024  # dinov2-large 的 CLS token 维度

    # BGE-M3 文本模型
    text_model_name: str = "BAAI/bge-m3"
    text_embedding_dim: int = 1024  # bge-m3 的输出维度

    # 推理参数
    vision_batch_size: int = 32   # 每张 4090 上的图像 batch size
    text_batch_size: int = 128    # 每张 4090 上的文本 batch size
    max_text_length: int = 256    # 文本最大 token 长度


@dataclass
class IndexConfig:
    """索引存储相关配置"""
    index_dir: str = "indexes"
    image_index_file: str = "image_index.faiss"
    text_index_file: str = "text_index.faiss"


@dataclass
class PipelineConfig:
    """离线建库流水线配置"""
    num_gpus: int = 8                          # 使用的 GPU 数量
    gpu_ids: List[int] = field(default_factory=lambda: list(range(8)))
    data_root_dir: str = "data/train"          # 训练集根目录（包含子文件夹）


@dataclass
class AppConfig:
    """Streamlit 应用配置"""
    retrieval_device: str = "cuda:0"  # 在线检索使用的 GPU
    default_top_k: int = 5
    max_top_k: int = 50
