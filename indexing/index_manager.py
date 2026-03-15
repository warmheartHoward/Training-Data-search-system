"""
FAISS 向量索引管理器。
负责索引的创建、添加、保存、加载和检索操作。
"""
import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SearchResult:
    """单条检索结果"""
    rank: int              # 排名（从 1 开始）
    score: float           # 相似度分数（内积 / 余弦相似度）
    image_path: str        # 训练集图像路径（普通路径或 tar URI）
    text: str              # 训练集实体名称
    annotation: str        # 训练集打标结果（assistant 的完整回答）
    source: str            # 数据来源
    index_id: int          # 在索引中的 ID


class FaissIndexManager:
    """
    FAISS 索引管理器。
    - 支持 IndexFlatIP（精确内积检索）和 IndexHNSWFlat（近似检索）。
    - 管理 metadata 的对齐存储与加载。
    """

    def __init__(self, embedding_dim: int, index_type: str = "flat"):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self._build_index()

    def _build_index(self) -> None:
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 128
        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")

    def add(self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]]) -> None:
        assert embeddings.shape[0] == len(metadata_list)
        assert embeddings.shape[1] == self.embedding_dim
        embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)
        self.metadata.extend(metadata_list)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype(np.float32)

        actual_k = min(top_k, self.index.ntotal)
        if actual_k == 0:
            return []

        scores, indices = self.index.search(query_embedding, actual_k)

        results: List[SearchResult] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            results.append(SearchResult(
                rank=rank + 1,
                score=float(score),
                image_path=meta.get("image_path", ""),
                text=meta.get("text", ""),
                annotation=meta.get("annotation", ""),
                source=meta.get("source", ""),
                index_id=int(idx),
            ))
        return results

    def save(self, index_path: str, metadata_path: str) -> None:
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] 索引已保存: {index_path} (共 {self.index.ntotal} 条向量)")

    @classmethod
    def load(cls, index_path: str, metadata_path: str) -> "FaissIndexManager":
        index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        instance = cls.__new__(cls)
        instance.embedding_dim = index.d
        instance.index = index
        instance.metadata = metadata
        instance.index_type = "loaded"
        print(f"[INFO] 索引已加载: {index_path} (共 {index.ntotal} 条向量)")
        return instance
