"""
Benchmark 评测文件夹解析工具。

支持的文件夹结构：
    benchmark_dir/
    ├── image_001.jpg
    ├── image_001.json    ← 与图像同名
    ├── image_002.png
    ├── image_002.json
    └── ...

JSON 结构不限，系统会自动发现所有可达的字符串字段路径供用户选择。
例如：
    {"entity_name": "青铜鼎", "category": "青铜器"}
        → 字段路径: entity_name, category
    {"ModelA": [{"question": "...", "answer": "文物名称"}]}
        → 字段路径: ModelA[0].question, ModelA[0].answer
"""
import json
import os
import re
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


# ============================================================================
#  通用字段发现与提取
# ============================================================================

def extract_json_fields(samples: List[Dict[str, Any]], max_samples: int = 50) -> List[str]:
    """
    自动发现样本 JSON 中所有可达的字符串字段路径。

    递归遍历 JSON 结构，对列表只探查第一个元素。
    返回所有终点为非空字符串值的路径。

    路径格式示例：
        "entity_name"          → json["entity_name"]
        "ModelA[0].answer"     → json["ModelA"][0]["answer"]
        "meta.category"        → json["meta"]["category"]

    Args:
        samples: scan_benchmark_folder 的返回值
        max_samples: 最多探查多少个样本（避免大数据集耗时）

    Returns:
        排序后的字段路径列表
    """
    fields: Set[str] = set()

    def _walk(obj: Any, prefix: str) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                path = f"{prefix}.{key}" if prefix else key
                _walk(value, path)
        elif isinstance(obj, list) and len(obj) > 0:
            _walk(obj[0], f"{prefix}[0]")
        elif isinstance(obj, str) and obj.strip():
            if prefix:
                fields.add(prefix)

    for sample in samples[:max_samples]:
        _walk(sample.get("json_data", {}), "")

    return sorted(fields)


def _parse_field_path(path: str) -> list:
    """
    将字段路径字符串解析为访问序列。

    "ModelA[0].answer" → ["ModelA", 0, "answer"]
    "entity_name"      → ["entity_name"]
    "meta.category"    → ["meta", "category"]
    """
    parts: list = []
    for segment in path.split("."):
        match = re.match(r'^(.+?)\[(\d+)\]$', segment)
        if match:
            parts.append(match.group(1))
            parts.append(int(match.group(2)))
        else:
            parts.append(segment)
    return parts


def get_field_value(sample: Dict[str, Any], field_path: str) -> Optional[str]:
    """
    根据字段路径从样本 JSON 中提取字符串值。

    Args:
        sample: 单个样本字典
        field_path: 字段路径，如 "ModelA[0].answer" 或 "entity_name"

    Returns:
        字符串值，路径不存在或值非字符串时返回 None
    """
    obj: Any = sample.get("json_data", {})
    for part in _parse_field_path(field_path):
        if isinstance(part, int):
            if isinstance(obj, list) and len(obj) > part:
                obj = obj[part]
            else:
                return None
        elif isinstance(part, str):
            if isinstance(obj, dict):
                obj = obj.get(part)
            else:
                return None
        if obj is None:
            return None
    return str(obj) if isinstance(obj, str) else None


def preview_field_values(
    samples: List[Dict[str, Any]], field_path: str, max_preview: int = 5,
) -> List[str]:
    """
    预览某个字段路径在前几个样本中的实际值，帮助用户确认选择是否正确。

    Returns:
        非空字符串值列表（最多 max_preview 个）
    """
    values: List[str] = []
    for sample in samples:
        v = get_field_value(sample, field_path)
        if v:
            values.append(v)
            if len(values) >= max_preview:
                break
    return values
