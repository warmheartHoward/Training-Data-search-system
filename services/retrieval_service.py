"""
在线检索服务。
支持多版本索引的动态加载与合并检索。
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import numpy as np
from PIL import Image

from models.vision_encoder import VisionEncoder
from models.text_encoder import TextEncoder
from indexing.index_manager import FaissIndexManager, SearchResult
from configs.config import ModelConfig, IndexConfig, AppConfig


def discover_versions(index_dir: str) -> List[str]:
    """
    自动扫描索引根目录，发现所有可用的版本子目录。

    判定规则：子目录中包含 image_index.faiss 文件即视为有效版本。
    如果根目录本身直接包含索引文件（无版本子目录的旧格式），返回 ["default"]。

    Args:
        index_dir: 索引根目录路径

    Returns:
        版本名列表（排序后），如 ["default"], ["v1", "v2_museum"]
    """
    if not os.path.isdir(index_dir):
        return []

    versions: List[str] = []

    # 检查根目录是否直接有索引文件（兼容无版本的旧格式）
    if os.path.isfile(os.path.join(index_dir, "image_index.faiss")):
        versions.append("default")

    # 扫描子目录
    for entry in sorted(os.listdir(index_dir)):
        sub_dir = os.path.join(index_dir, entry)
        if os.path.isdir(sub_dir) and os.path.isfile(
            os.path.join(sub_dir, "image_index.faiss")
        ):
            versions.append(entry)

    return versions


class RetrievalService:
    """
    检索服务：管理模型加载与多版本索引检索。

    - 模型只加载一次，常驻显存。
    - 索引按版本懒加载，缓存在内存中。
    - 检索时合并所选版本的结果，按分数全局排序取 Top-K。
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        index_config: Optional[IndexConfig] = None,
        app_config: Optional[AppConfig] = None,
    ):
        self.model_config = model_config or ModelConfig()
        self.index_config = index_config or IndexConfig()
        self.app_config = app_config or AppConfig()

        device = self.app_config.retrieval_device
        print(f"[INFO] 正在加载模型到 {device} ...")

        # ---- 加载编码器（只加载一次） ----
        self.vision_encoder = VisionEncoder(
            model_name=self.model_config.vision_model_name,
            device=device,
        )
        print(f"[INFO] DINOv2 视觉模型加载完成 (dim={self.vision_encoder.embedding_dim})")

        self.text_encoder = TextEncoder(
            model_name=self.model_config.text_model_name,
            device=device,
        )
        print(f"[INFO] BGE-M3 文本模型加载完成 (dim={self.text_encoder.embedding_dim})")

        # ---- 索引缓存：version -> (image_index, text_index) ----
        self._index_cache: Dict[str, tuple[FaissIndexManager, FaissIndexManager]] = {}

        # ---- 发现可用版本 ----
        self.available_versions = discover_versions(self.index_config.index_dir)
        print(f"[INFO] 发现索引版本: {self.available_versions}")

        # 默认加载所有版本
        for v in self.available_versions:
            self._load_version(v)

    def _version_dir(self, version: str) -> str:
        """返回版本对应的磁盘目录。"""
        if version == "default":
            return self.index_config.index_dir
        return os.path.join(self.index_config.index_dir, version)

    def _load_version(self, version: str) -> None:
        """加载单个版本的索引到缓存。"""
        if version in self._index_cache:
            return

        vdir = self._version_dir(version)
        img_index_path = os.path.join(vdir, self.index_config.image_index_file)
        txt_index_path = os.path.join(vdir, self.index_config.text_index_file)
        img_meta_path = os.path.join(vdir, "image_metadata.pkl")
        txt_meta_path = os.path.join(vdir, "text_metadata.pkl")

        img_index = FaissIndexManager.load(img_index_path, img_meta_path)

        txt_index = None
        if os.path.isfile(txt_index_path) and os.path.isfile(txt_meta_path):
            txt_index = FaissIndexManager.load(txt_index_path, txt_meta_path)

        self._index_cache[version] = (img_index, txt_index)
        print(f"[INFO] 版本 '{version}' 索引已加载")

    def get_active_versions(self, selected: Optional[List[str]] = None) -> List[str]:
        """返回实际参与检索的版本列表。"""
        if selected:
            return [v for v in selected if v in self._index_cache]
        return list(self._index_cache.keys())

    def search_by_image(
        self,
        image: Union[str, Path, Image.Image],
        top_k: int = 5,
        versions: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        以图搜图：在指定版本的图像索引中检索，合并排序返回全局 Top-K。

        Args:
            image: 查询图像（路径或 PIL Image）
            top_k: 返回的 Top-K 数量
            versions: 要检索的版本列表，None 表示全部版本
        """
        query_embedding = self.vision_encoder.encode_single(image)
        active = self.get_active_versions(versions)

        all_results: List[SearchResult] = []
        for v in active:
            img_index, _ = self._index_cache[v]
            results = img_index.search(query_embedding, top_k=top_k)
            # 给结果附加版本标记（追加到 source 字段）
            for r in results:
                r.source = f"[{v}] {r.source}" if len(active) > 1 else r.source
            all_results.extend(results)

        # 全局按分数降序排序，取 Top-K
        all_results.sort(key=lambda r: r.score, reverse=True)
        all_results = all_results[:top_k]
        # 重新编排 rank
        for i, r in enumerate(all_results):
            r.rank = i + 1
        return all_results

    def search_by_text(
        self,
        query_text: str,
        top_k: int = 5,
        versions: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        以文搜文：在指定版本的文本索引中检索，合并排序返回全局 Top-K。

        Args:
            query_text: 查询文本
            top_k: 返回的 Top-K 数量
            versions: 要检索的版本列表，None 表示全部版本
        """
        query_embedding = self.text_encoder.encode_single(query_text)
        active = self.get_active_versions(versions)

        all_results: List[SearchResult] = []
        for v in active:
            _, txt_index = self._index_cache[v]
            if txt_index is None:
                continue
            results = txt_index.search(query_embedding, top_k=top_k)
            for r in results:
                r.source = f"[{v}] {r.source}" if len(active) > 1 else r.source
            all_results.extend(results)

        all_results.sort(key=lambda r: r.score, reverse=True)
        all_results = all_results[:top_k]
        for i, r in enumerate(all_results):
            r.rank = i + 1
        return all_results
