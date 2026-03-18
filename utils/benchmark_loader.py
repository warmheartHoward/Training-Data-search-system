"""
Benchmark 评测文件夹解析工具。

支持的文件夹结构：
    benchmark_dir/
    ├── image_001.jpg
    ├── image_001.json    ← 与图像同名
    ├── image_002.png
    ├── image_002.json
    └── ...

JSON 中包含一个或多个模型的 QA 结果，格式如：
    {
        "ModelName_V1": [{"question": "...", "answer": "文物名称"}],
        "ModelName_V2": [{"question": "...", "answer": "文物名称"}]
    }
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# 支持的图像扩展名
IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def scan_benchmark_folder(folder: str) -> List[Dict[str, Any]]:
    """
    扫描评测文件夹，返回所有 图像+JSON 配对的样本列表。

    Args:
        folder: 评测数据文件夹路径

    Returns:
        样本列表，每个元素:
        {
            "image_path": str,   # 图像绝对路径
            "json_path": str,    # JSON 绝对路径
            "json_data": dict,   # 解析后的 JSON 内容
            "stem": str,         # 文件名（不含扩展名）
        }
    """
    folder_path = Path(os.path.normpath(folder))
    if not folder_path.is_dir():
        return []

    samples: List[Dict[str, Any]] = []

    # 收集所有图像文件
    image_files = sorted(
        f for f in folder_path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    for img_file in image_files:
        json_file = img_file.with_suffix(".json")
        if not json_file.exists():
            continue

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                json_data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        samples.append({
            "image_path": str(img_file.resolve()),
            "json_path": str(json_file.resolve()),
            "json_data": json_data,
            "stem": img_file.stem,
        })

    return samples


def extract_model_keys(samples: List[Dict[str, Any]]) -> List[str]:
    """
    从所有样本的 JSON 中提取可用的模型 key 列表。

    遍历所有 JSON，收集 top-level keys 的并集，
    筛选出值为 list 类型的 key（即 QA 结果列表）。

    Args:
        samples: scan_benchmark_folder 的返回值

    Returns:
        排序后的模型 key 列表
    """
    all_keys: Set[str] = set()
    for sample in samples:
        data = sample.get("json_data", {})
        for key, value in data.items():
            if isinstance(value, list):
                all_keys.add(key)
    return sorted(all_keys)


def get_entity_name(sample: Dict[str, Any], model_key: str) -> Optional[str]:
    """
    从样本 JSON 中提取指定模型的文物实体名称（answer 字段）。

    Args:
        sample: 单个样本字典
        model_key: 模型 key（JSON 中的 top-level key）

    Returns:
        实体名称字符串，缺失时返回 None
    """
    data = sample.get("json_data", {})
    qa_list = data.get(model_key)
    if not isinstance(qa_list, list) or len(qa_list) == 0:
        return None

    # 取第一条 QA 的 answer
    first_qa = qa_list[0]
    if isinstance(first_qa, dict):
        answer = first_qa.get("answer", "")
        return answer if answer else None

    return None
