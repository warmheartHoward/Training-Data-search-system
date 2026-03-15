"""
DINOv2-Large 视觉编码器封装。
提取图像的 CLS token 作为全局特征向量，用于实例级以图搜图。
支持从普通路径和 tar URI 两种方式读取图像。
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from typing import List, Union
import numpy as np
from pathlib import Path

from data.tar_reader import load_image, is_tar_uri


class ImagePathDataset(Dataset):
    """
    图像路径数据集，支持普通路径和 tar URI。
    对于无法打开的图像，返回 None 并在 collate 时跳过。
    """
    def __init__(self, image_paths: List[str], processor):
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        try:
            # 统一使用 load_image，同时支持普通路径和 tar URI
            image = load_image(path)
            if image is None:
                return idx, None
            inputs = self.processor(images=image, return_tensors="pt")
            return idx, inputs["pixel_values"].squeeze(0)
        except Exception as e:
            print(f"[WARNING] 无法加载图像 {path}: {e}")
            return idx, None


def _collate_fn(batch):
    """自定义 collate：过滤掉加载失败（None）的样本。"""
    valid = [(idx, pv) for idx, pv in batch if pv is not None]
    if not valid:
        return [], None
    indices, pixel_values = zip(*valid)
    return list(indices), torch.stack(pixel_values)


class VisionEncoder:
    """
    DINOv2-Large 视觉编码器。
    - 使用 CLS token 输出作为图像全局特征。
    - 支持批量提取，适配多卡并行流水线。
    """

    def __init__(self, model_name: str = "facebook/dinov2-large", device: str = "cuda:0"):
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.embedding_dim: int = self.model.config.hidden_size

    @torch.no_grad()
    def encode_single(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """
        编码单张图像，返回 L2 归一化后的特征向量。

        Args:
            image: 图像路径（普通路径或 tar URI）或 PIL Image 对象
        """
        if isinstance(image, (str, Path)):
            image = load_image(str(image))
            if image is None:
                raise ValueError(f"无法加载图像")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = F.normalize(cls_embedding, p=2, dim=-1)
        return cls_embedding.cpu().numpy().squeeze(0)

    @torch.no_grad()
    def encode_batch(
        self,
        image_paths: List[str],
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> tuple[np.ndarray, List[int]]:
        """
        批量编码图像列表，返回特征矩阵和有效样本的原始索引。

        注意：当图像来自 tar 包时，num_workers 必须为 0（主进程读取），
        因为 tarfile 句柄不能跨进程共享。

        Args:
            image_paths: 图像路径列表（支持普通路径和 tar URI 混合）
            batch_size: 每个 batch 的图像数量（4090 24GB 建议 32）
            num_workers: DataLoader 的工作进程数（tar 模式下应为 0）
        """
        # 如果路径中包含 tar URI，强制 num_workers=0
        has_tar = any(is_tar_uri(p) for p in image_paths[:10])  # 抽样检查
        if has_tar:
            num_workers = 0

        dataset = ImagePathDataset(image_paths, self.processor)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate_fn,
            pin_memory=(num_workers > 0),
        )

        all_embeddings: List[np.ndarray] = []
        all_indices: List[int] = []

        for indices, pixel_values in dataloader:
            if pixel_values is None:
                continue
            pixel_values = pixel_values.to(self.device)
            outputs = self.model(pixel_values=pixel_values)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            cls_embedding = F.normalize(cls_embedding, p=2, dim=-1)
            all_embeddings.append(cls_embedding.cpu().numpy())
            all_indices.extend(indices)

        if not all_embeddings:
            return np.empty((0, self.embedding_dim), dtype=np.float32), []

        return np.concatenate(all_embeddings, axis=0), all_indices
