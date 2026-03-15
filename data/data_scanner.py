"""
训练数据扫描器：适配 tar+JSONL 目录结构。

目录结构约定：
    root_dir/
    ├── dataset_source_A/              ← 数据源（不同命名的文件夹）
    │   ├── images/
    │   │   ├── data_000000.tar        ← 图像 tar 包
    │   │   ├── data_000001.tar
    │   │   └── ...
    │   └── jsonl/
    │       ├── data_000000.jsonl      ← 与 tar 同名的标注 JSONL
    │       ├── data_000001.jsonl
    │       └── ...
    ├── dataset_source_B/
    │   ├── images/
    │   └── jsonl/
    └── ...

JSONL 每行包含一条样本，关键字段提取规则：
- 图像路径：data[0].content[0].image.relative_path（tar 包内的相对路径）
- 实体名称：meta_info_image.knowledge_info.knowledge_entities[0].entity_name
- 打标结果：data[1].content[0].text.string（assistant 的回答）
- 数据来源：meta_info_image.source_info.source_name
- 打标模型：data_generated_info.task_oriented_data[0].model_name
"""
import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from data.tar_reader import make_tar_uri


@dataclass
class AnnotationData:
    """单条标注数据的完整解析结果"""
    # ---- 定位信息 ----
    image_path: str            # 图像访问路径（tar URI 格式：tar:///path.tar::inner/path.jpg）
    data_uuid: str = ""        # 数据唯一 ID
    source_dataset: str = ""   # 所属数据源文件夹名
    tar_name: str = ""         # 所属 tar 文件名（如 data_000000）

    # ---- 核心标注 ----
    entity_name: str = ""      # 实体名称（用于以文搜文）
    annotation: str = ""       # 打标结果（assistant 的完整回答，用于可视化展示）

    # ---- 辅助信息 ----
    source_name: str = ""      # 数据来源（如「上海博物馆(DPM)」）
    model_name: str = ""       # 打标模型名称
    image_relative_path: str = ""  # tar 包内的图像相对路径
    entity_tags: List[str] = field(default_factory=list)  # 实体标签

    # ---- 原始记录（用于质检和调试） ----
    raw_record: Dict[str, Any] = field(default_factory=dict)


def _safe_get(d: Any, *keys, default=""):
    """安全的嵌套字典取值，任一层级不存在时返回 default。"""
    current = d
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, None)
        elif isinstance(current, (list, tuple)) and isinstance(key, int):
            current = current[key] if 0 <= key < len(current) else None
        else:
            return default
        if current is None:
            return default
    return current if current is not None else default


def parse_jsonl_record(record: Dict[str, Any], tar_path: str, source_dataset: str, tar_name: str) -> Optional[AnnotationData]:
    """
    解析单条 JSONL 记录，提取关键字段。

    Args:
        record: JSONL 一行解析后的字典
        tar_path: 对应 tar 文件的绝对路径
        source_dataset: 数据源文件夹名
        tar_name: tar 文件的基本名（无扩展名）

    Returns:
        AnnotationData 对象，解析失败返回 None
    """
    try:
        data = record.get("data", [])
        if len(data) < 2:
            return None

        # ---- 图像相对路径 ----
        user_content = _safe_get(data, 0, "content")
        if not isinstance(user_content, list) or not user_content:
            return None

        # 从 user content 中找到 type=image 的项
        image_relative_path = ""
        for item in user_content:
            if isinstance(item, dict) and item.get("type") == "image":
                image_relative_path = _safe_get(item, "image", "relative_path", default="")
                break
        if not image_relative_path:
            return None

        # ---- assistant 的打标结果 ----
        assistant_content = _safe_get(data, 1, "content")
        annotation = ""
        if isinstance(assistant_content, list):
            for item in assistant_content:
                if isinstance(item, dict) and item.get("type") == "text":
                    annotation = _safe_get(item, "text", "string", default="")
                    break

        # ---- 实体名称 ----
        entities = _safe_get(record, "meta_info_image", "knowledge_info", "knowledge_entities", default=[])
        entity_name = ""
        entity_tags = []
        if isinstance(entities, list) and entities:
            entity_name = _safe_get(entities[0], "entity_name", default="")
            entity_tags = _safe_get(entities[0], "entity_tags", default=[])

        # ---- 数据来源 ----
        source_name = _safe_get(record, "meta_info_image", "source_info", "source_name", default="")

        # ---- 打标模型 ----
        task_data = _safe_get(record, "data_generated_info", "task_oriented_data", default=[])
        model_name = ""
        if isinstance(task_data, list) and task_data:
            model_name = _safe_get(task_data[0], "model_name", default="")

        # ---- 构建 tar URI ----
        image_uri = make_tar_uri(tar_path, image_relative_path)

        return AnnotationData(
            image_path=image_uri,
            data_uuid=record.get("data_uuid", ""),
            source_dataset=source_dataset,
            tar_name=tar_name,
            entity_name=entity_name,
            annotation=annotation,
            source_name=source_name,
            model_name=model_name,
            image_relative_path=image_relative_path,
            entity_tags=entity_tags if isinstance(entity_tags, list) else [],
            raw_record=record,
        )
    except Exception as e:
        print(f"[WARNING] 解析 JSONL 记录失败: {e}")
        return None


def scan_dataset(root_dir: str, show_progress: bool = True) -> List[AnnotationData]:
    """
    扫描训练数据根目录，遍历所有数据源的 jsonl/ 子目录，解析每条记录。

    Args:
        root_dir: 数据根目录
        show_progress: 是否显示进度条

    Returns:
        AnnotationData 列表
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"数据目录不存在: {root_dir}")

    # ---- 收集所有 JSONL 文件及其对应的 tar 路径 ----
    jsonl_tar_pairs: List[tuple[Path, Path, str]] = []  # (jsonl_path, tar_path, source_dataset)

    for source_dir in sorted(root.iterdir()):
        if not source_dir.is_dir():
            continue
        jsonl_dir = source_dir / "jsonl"
        images_dir = source_dir / "images"
        if not jsonl_dir.exists():
            continue

        source_name = source_dir.name

        for jsonl_file in sorted(jsonl_dir.glob("*.jsonl")):
            # data_000000.jsonl → data_000000.tar
            tar_name = jsonl_file.stem
            tar_file = images_dir / f"{tar_name}.tar"
            if not tar_file.exists():
                print(f"[WARNING] tar 文件不存在，跳过: {tar_file}")
                continue
            jsonl_tar_pairs.append((jsonl_file, tar_file, source_name))

    print(f"[INFO] 找到 {len(jsonl_tar_pairs)} 个 JSONL-TAR 配对")

    # ---- 逐个 JSONL 文件解析 ----
    samples: List[AnnotationData] = []
    total_lines = 0
    skipped = 0

    iterator = tqdm(jsonl_tar_pairs, desc="扫描 JSONL") if show_progress else jsonl_tar_pairs
    for jsonl_path, tar_path, source_dataset in iterator:
        tar_name = jsonl_path.stem
        tar_abs = str(tar_path.resolve())

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                total_lines += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                sample = parse_jsonl_record(record, tar_abs, source_dataset, tar_name)
                if sample is not None:
                    samples.append(sample)
                else:
                    skipped += 1

    print(f"[INFO] 扫描完成: {len(samples)} 条有效 | {skipped} 条跳过 | {total_lines} 条总计")
    return samples
