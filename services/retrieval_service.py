"""
在线检索服务（单例模式）。
应用启动时加载模型和 FAISS 索引，对外提供以图搜图和以文搜文接口。
"""
import os
from pathlib import Path
from typing import List, Optional, Union
from PIL import Image
import threading

from models.vision_encoder import VisionEncoder
from models.text_encoder import TextEncoder
from indexing.index_manager import FaissIndexManager, SearchResult
from configs.config import ModelConfig, IndexConfig, AppConfig


class RetrievalService:
    """
    检索服务单例类。
    - 在 Streamlit 等应用中保证全局只加载一次模型和索引。
    - 提供 search_by_image 和 search_by_text 两个核心接口。
    """

    _instance: Optional["RetrievalService"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> "RetrievalService":
        """线程安全的单例实现。"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        index_config: Optional[IndexConfig] = None,
        app_config: Optional[AppConfig] = None,
    ):
        # 避免重复初始化
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.model_config = model_config or ModelConfig()
        self.index_config = index_config or IndexConfig()
        self.app_config = app_config or AppConfig()

        device = self.app_config.retrieval_device
        print(f"[INFO] 正在加载模型到 {device} ...")

        # ---- 加载视觉编码器 ----
        self.vision_encoder = VisionEncoder(
            model_name=self.model_config.vision_model_name,
            device=device,
        )
        print(f"[INFO] DINOv2 视觉模型加载完成 (dim={self.vision_encoder.embedding_dim})")

        # ---- 加载文本编码器 ----
        self.text_encoder = TextEncoder(
            model_name=self.model_config.text_model_name,
            device=device,
        )
        print(f"[INFO] BGE-M3 文本模型加载完成 (dim={self.text_encoder.embedding_dim})")

        # ---- 加载 FAISS 索引 ----
        index_dir = self.index_config.index_dir
        image_index_path = os.path.join(index_dir, self.index_config.image_index_file)
        text_index_path = os.path.join(index_dir, self.index_config.text_index_file)
        image_metadata_path = os.path.join(index_dir, "image_metadata.pkl")
        text_metadata_path = os.path.join(index_dir, "text_metadata.pkl")

        self.image_index = FaissIndexManager.load(image_index_path, image_metadata_path)
        self.text_index = FaissIndexManager.load(text_index_path, text_metadata_path)
        print("[INFO] FAISS 索引加载完成")

        self._initialized = True

    def search_by_image(
        self,
        image: Union[str, Path, Image.Image],
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        以图搜图：提取 query 图像的视觉特征，在图像索引中检索最相似的训练集样本。

        Args:
            image: 查询图像（路径或 PIL Image）
            top_k: 返回的 Top-K 数量

        Returns:
            SearchResult 列表（包含相似度、训练集原图路径、GT 词条）
        """
        query_embedding = self.vision_encoder.encode_single(image)
        return self.image_index.search(query_embedding, top_k=top_k)

    def search_by_text(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        以文搜文：提取 query 文本的语义特征，在文本索引中检索最相似的训练集词条。

        Args:
            query_text: 查询文本（中文或英文均可）
            top_k: 返回的 Top-K 数量

        Returns:
            SearchResult 列表（包含相似度、训练集原图路径、GT 词条）
        """
        query_embedding = self.text_encoder.encode_single(query_text)
        return self.text_index.search(query_embedding, top_k=top_k)
