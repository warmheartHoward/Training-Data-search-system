"""
离线建库脚本：多 GPU 并行提取训练集的图像特征和文本特征，构建 FAISS 索引。

数据格式：
    root_dir/
    ├── dataset_A/
    │   ├── images/    ← tar 包（data_000000.tar, ...）
    │   └── jsonl/     ← 对应 JSONL（data_000000.jsonl, ...）
    └── dataset_B/
        └── ...

    使用 entity_name 作为以文搜文的检索文本。
    使用 tar URI 格式存储图像路径，避免解压 tar 到磁盘。

使用方法：
    python offline_build_index.py --data_dir /path/to/root --num_gpus 8
"""
import os
import sys
import argparse
import time
from typing import List, Dict, Any
from dataclasses import dataclass

import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.vision_encoder import VisionEncoder
from models.text_encoder import TextEncoder
from indexing.index_manager import FaissIndexManager
from data.data_scanner import AnnotationData, scan_dataset
from data.quality_checker import check_dataset, format_report_text
from configs.config import ModelConfig, IndexConfig


# ============================================================================
#  数据准备
# ============================================================================

@dataclass
class TrainSample:
    """传入 worker 的轻量样本"""
    global_id: int
    image_path: str    # tar URI 或普通路径
    text: str          # entity_name
    annotation: str    # assistant 打标结果（存入 metadata 供可视化展示）
    source: str        # 数据来源


def annotations_to_samples(annotations: List[AnnotationData]) -> List[TrainSample]:
    """将扫描结果转为建索引所需的轻量样本列表。"""
    samples: List[TrainSample] = []
    for idx, ann in enumerate(annotations):
        samples.append(TrainSample(
            global_id=idx,
            image_path=ann.image_path,
            text=ann.entity_name,
            annotation=ann.annotation,
            source=ann.source_name,
        ))
    return samples


def shard_data(samples: List[TrainSample], num_shards: int) -> List[List[TrainSample]]:
    """将数据集均匀分成 num_shards 份。"""
    shards: List[List[TrainSample]] = [[] for _ in range(num_shards)]
    for i, sample in enumerate(samples):
        shards[i % num_shards].append(sample)
    for i, shard in enumerate(shards):
        print(f"  [SHARD {i}] {len(shard)} 条样本")
    return shards


# ============================================================================
#  单 GPU Worker
# ============================================================================

def worker_extract_features(
    gpu_id: int,
    shard: List[TrainSample],
    model_config: ModelConfig,
    result_dict: dict,
) -> None:
    """单 GPU worker：加载模型 → 提取图像/文本特征 → 写入共享字典。"""
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] 启动，{len(shard)} 条样本")

    vision_encoder = VisionEncoder(model_name=model_config.vision_model_name, device=device)
    text_encoder = TextEncoder(model_name=model_config.text_model_name, device=device)
    print(f"[GPU {gpu_id}] 模型加载完成")

    # ---- 图像特征 ----
    image_paths = [s.image_path for s in shard]
    print(f"[GPU {gpu_id}] 提取图像特征 (batch_size={model_config.vision_batch_size}) ...")
    image_embeddings, valid_indices = vision_encoder.encode_batch(
        image_paths,
        batch_size=model_config.vision_batch_size,
        num_workers=0,  # tar 模式必须单进程读取
    )
    print(f"[GPU {gpu_id}] 图像特征: {image_embeddings.shape[0]}/{len(shard)} 成功")

    # ---- 文本特征（只对有 entity_name 的样本编码） ----
    text_samples = [(i, s) for i, s in enumerate(shard) if s.text.strip()]
    texts = [s.text for _, s in text_samples]
    print(f"[GPU {gpu_id}] 提取文本特征 ({len(texts)} 条) ...")
    text_embeddings = text_encoder.encode_batch(
        texts,
        batch_size=model_config.text_batch_size,
        max_length=model_config.max_text_length,
    ) if texts else np.empty((0, text_encoder.embedding_dim), dtype=np.float32)
    print(f"[GPU {gpu_id}] 文本特征: {text_embeddings.shape}")

    # ---- 构建元信息 ----
    def _make_meta(sample: TrainSample) -> Dict[str, Any]:
        return {
            "global_id": sample.global_id,
            "image_path": sample.image_path,
            "text": sample.text,
            "annotation": sample.annotation,
            "source": sample.source,
        }

    image_metadata = [_make_meta(shard[i]) for i in valid_indices]
    text_metadata = [_make_meta(shard[i]) for i, _ in text_samples]

    result_dict[gpu_id] = {
        "image_embeddings": image_embeddings,
        "image_metadata": image_metadata,
        "text_embeddings": text_embeddings,
        "text_metadata": text_metadata,
    }

    del vision_encoder, text_encoder
    torch.cuda.empty_cache()
    print(f"[GPU {gpu_id}] 完成")


# ============================================================================
#  主流程
# ============================================================================

def build_index(
    data_dir: str,
    num_gpus: int = 8,
    index_type: str = "flat",
    index_dir: str = "indexes",
    run_qc: bool = True,
) -> None:
    total_start = time.time()

    # ---- 1. 扫描 ----
    print("=" * 60)
    print("[STEP 1] 扫描训练数据目录")
    print("=" * 60)
    annotations = scan_dataset(data_dir, show_progress=True)
    if not annotations:
        print("[ERROR] 未找到有效数据")
        sys.exit(1)

    # ---- 1.5 质检 ----
    if run_qc:
        print("\n" + "=" * 60)
        print("[STEP 1.5] 数据质检")
        print("=" * 60)
        qc_report = check_dataset(annotations, check_image_readable=False)
        print(format_report_text(qc_report))

    # ---- 2. 转换样本 ----
    samples = annotations_to_samples(annotations)
    print(f"\n[INFO] 建索引样本: {len(samples)} 条")

    # ---- 3. 多 GPU 并行 ----
    available_gpus = torch.cuda.device_count()
    actual_gpus = min(num_gpus, max(available_gpus, 1))
    if available_gpus == 0:
        print("[ERROR] 未检测到 GPU")
        sys.exit(1)

    shards = shard_data(samples, actual_gpus)

    print(f"\n{'=' * 60}")
    print(f"[STEP 2] {actual_gpus} GPU 并行提取特征")
    print("=" * 60)

    model_config = ModelConfig()
    manager = mp.Manager()
    result_dict = manager.dict()

    processes: List[mp.Process] = []
    for gpu_id in range(actual_gpus):
        p = mp.Process(
            target=worker_extract_features,
            args=(gpu_id, shards[gpu_id], model_config, result_dict),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    if len(result_dict) != actual_gpus:
        failed = set(range(actual_gpus)) - set(result_dict.keys())
        print(f"[ERROR] Worker 失败: {failed}")
        sys.exit(1)

    # ---- 4. 合并 + 建索引 ----
    print(f"\n{'=' * 60}")
    print("[STEP 3] 合并特征，构建 FAISS 索引")
    print("=" * 60)

    all_img_emb, all_img_meta = [], []
    all_txt_emb, all_txt_meta = [], []

    for gpu_id in range(actual_gpus):
        r = result_dict[gpu_id]
        all_img_emb.append(r["image_embeddings"])
        all_img_meta.extend(r["image_metadata"])
        all_txt_emb.append(r["text_embeddings"])
        all_txt_meta.extend(r["text_metadata"])

    img_emb = np.concatenate(all_img_emb, axis=0)
    txt_emb = np.concatenate(all_txt_emb, axis=0) if all_txt_emb else np.empty((0,), dtype=np.float32)

    print(f"  图像: {img_emb.shape}  |  文本: {txt_emb.shape}")

    index_config = IndexConfig(index_dir=index_dir)

    # 图像索引
    img_mgr = FaissIndexManager(embedding_dim=img_emb.shape[1], index_type=index_type)
    img_mgr.add(img_emb, all_img_meta)
    img_mgr.save(
        os.path.join(index_dir, index_config.image_index_file),
        os.path.join(index_dir, "image_metadata.pkl"),
    )

    # 文本索引
    if txt_emb.ndim == 2 and txt_emb.shape[0] > 0:
        txt_mgr = FaissIndexManager(embedding_dim=txt_emb.shape[1], index_type=index_type)
        txt_mgr.add(txt_emb, all_txt_meta)
        txt_mgr.save(
            os.path.join(index_dir, index_config.text_index_file),
            os.path.join(index_dir, "text_metadata.pkl"),
        )

    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"[DONE] 建库完成！{total_time:.1f}s")
    print(f"  图像索引: {img_emb.shape[0]} 条  |  文本索引: {txt_emb.shape[0] if txt_emb.ndim == 2 else 0} 条")
    print(f"  目录: {os.path.abspath(index_dir)}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多 GPU 离线建库")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="训练数据根目录（含多个数据源子文件夹）")
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--index_type", type=str, default="flat", choices=["flat", "hnsw"])
    parser.add_argument("--index_dir", type=str, default="indexes")
    parser.add_argument("--skip_qc", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    build_index(
        data_dir=args.data_dir,
        num_gpus=args.num_gpus,
        index_type=args.index_type,
        index_dir=args.index_dir,
        run_qc=not args.skip_qc,
    )
