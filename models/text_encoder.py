"""
BGE-M3 文本编码器封装。
提取文本的 dense embedding，用于语义级以文搜文。
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List
import numpy as np


class TextEncoder:
    """
    BGE-M3 文本编码器。
    - 使用 [CLS] token 的 dense embedding 作为文本全局特征。
    - 支持批量编码，适配多卡并行流水线。
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cuda:0"):
        """
        Args:
            model_name: HuggingFace 模型名称或本地路径
            device: 推理设备，如 "cuda:0"
        """
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.embedding_dim: int = self.model.config.hidden_size  # 1024 for bge-m3

    @torch.no_grad()
    def encode_single(self, text: str, max_length: int = 256) -> np.ndarray:
        """
        编码单条文本，返回 L2 归一化后的特征向量。

        Args:
            text: 输入文本（中文或英文）
            max_length: 最大 token 长度

        Returns:
            形状为 (embedding_dim,) 的 numpy 数组
        """
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs)
        # 取 [CLS] token 作为句子级表示
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = F.normalize(cls_embedding, p=2, dim=-1)
        return cls_embedding.cpu().numpy().squeeze(0)

    @torch.no_grad()
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 128,
        max_length: int = 256,
    ) -> np.ndarray:
        """
        批量编码文本列表，返回特征矩阵。

        Args:
            texts: 文本列表
            batch_size: 每个 batch 的文本数量（4090 24GB 建议 128）
            max_length: 最大 token 长度

        Returns:
            形状为 (N, embedding_dim) 的 numpy 数组
        """
        all_embeddings: List[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            cls_embedding = F.normalize(cls_embedding, p=2, dim=-1)
            all_embeddings.append(cls_embedding.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)
