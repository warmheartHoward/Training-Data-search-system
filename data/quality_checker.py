"""
标注质量快速质检模块。
适配 tar+JSONL 数据格式，对每条标注进行多维度格式与内容检查。
"""
from dataclasses import dataclass, field
from typing import List, Dict
from enum import Enum
from collections import Counter

from data.data_scanner import AnnotationData
from data.tar_reader import load_image


class Severity(str, Enum):
    """质检问题严重等级"""
    ERROR = "ERROR"       # 严重：数据不可用
    WARNING = "WARNING"   # 警告：数据可用但有缺陷
    INFO = "INFO"         # 提示：建议优化


@dataclass
class QCIssue:
    """单条质检问题"""
    severity: Severity
    field: str
    message: str
    sample_name: str = ""


@dataclass
class QCSampleReport:
    """单条样本的质检报告"""
    image_path: str
    name: str
    issues: List[QCIssue] = field(default_factory=list)

    @property
    def has_error(self) -> bool:
        return any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def has_warning(self) -> bool:
        return any(i.severity == Severity.WARNING for i in self.issues)

    @property
    def is_clean(self) -> bool:
        return len(self.issues) == 0


@dataclass
class QCDatasetReport:
    """数据集质检汇总报告"""
    total_samples: int = 0
    clean_samples: int = 0
    error_samples: int = 0
    warning_samples: int = 0
    sample_reports: List[QCSampleReport] = field(default_factory=list)

    # 统计维度
    source_distribution: Dict[str, int] = field(default_factory=dict)
    model_distribution: Dict[str, int] = field(default_factory=dict)
    dataset_distribution: Dict[str, int] = field(default_factory=dict)

    # 按字段汇总的问题计数
    issue_field_counts: Dict[str, int] = field(default_factory=dict)


def check_single_sample(
    sample: AnnotationData,
    min_annotation_length: int = 50,
    max_annotation_length: int = 20000,
    check_image_readable: bool = False,
) -> QCSampleReport:
    """
    对单条标注数据执行全部质检规则。

    Args:
        sample: 待检查的标注数据
        min_annotation_length: 打标结果最小字符数
        max_annotation_length: 打标结果最大字符数
        check_image_readable: 是否从 tar 中实际读取图像验证（较慢）

    Returns:
        该样本的质检报告
    """
    label = sample.entity_name or sample.image_relative_path or sample.data_uuid
    report = QCSampleReport(image_path=sample.image_path, name=label)

    def _add(severity: Severity, fld: str, msg: str):
        report.issues.append(QCIssue(severity, fld, msg, label))

    # ---- 1. entity_name 检查 ----
    if not sample.entity_name.strip():
        _add(Severity.WARNING, "entity_name", "实体名称为空（将无法作为文本检索目标）")

    # ---- 2. annotation（打标结果）检查 ----
    ann = sample.annotation
    if not ann or not ann.strip():
        _add(Severity.ERROR, "annotation", "打标结果（assistant 回答）为空")
    else:
        ann_len = len(ann.strip())
        if ann_len < min_annotation_length:
            _add(Severity.WARNING, "annotation",
                 f"打标结果过短 ({ann_len} 字符 < {min_annotation_length})")
        if ann_len > max_annotation_length:
            _add(Severity.WARNING, "annotation",
                 f"打标结果过长 ({ann_len} 字符 > {max_annotation_length})")

        # 检查打标结果是否包含实体名称（一致性）
        if sample.entity_name.strip() and sample.entity_name.strip() not in ann:
            _add(Severity.INFO, "annotation",
                 f"打标结果中未出现实体名称「{sample.entity_name}」")

    # ---- 3. 图像路径检查 ----
    if not sample.image_relative_path:
        _add(Severity.ERROR, "image_relative_path", "图像相对路径为空")

    # ---- 4. 模型名称检查 ----
    if not sample.model_name.strip():
        _add(Severity.WARNING, "model_name", "打标模型名称为空")

    # ---- 5. 图像可读性检查（从 tar 中实际读取，可选） ----
    if check_image_readable:
        img = load_image(sample.image_path)
        if img is None:
            _add(Severity.ERROR, "image", "图像无法从 tar 中读取")

    return report


def check_dataset(
    samples: List[AnnotationData],
    min_annotation_length: int = 50,
    max_annotation_length: int = 20000,
    check_image_readable: bool = False,
) -> QCDatasetReport:
    """对整个数据集执行质检，生成汇总报告。"""
    report = QCDatasetReport(total_samples=len(samples))

    source_counter: Counter = Counter()
    model_counter: Counter = Counter()
    dataset_counter: Counter = Counter()
    issue_field_counter: Counter = Counter()

    for sample in samples:
        sample_report = check_single_sample(
            sample, min_annotation_length, max_annotation_length, check_image_readable,
        )
        report.sample_reports.append(sample_report)

        if sample_report.is_clean:
            report.clean_samples += 1
        elif sample_report.has_error:
            report.error_samples += 1
        else:
            report.warning_samples += 1

        source_counter[sample.source_name or "(空)"] += 1
        model_counter[sample.model_name or "(空)"] += 1
        dataset_counter[sample.source_dataset or "(空)"] += 1

        for issue in sample_report.issues:
            issue_field_counter[issue.field] += 1

    report.source_distribution = dict(source_counter.most_common())
    report.model_distribution = dict(model_counter.most_common())
    report.dataset_distribution = dict(dataset_counter.most_common())
    report.issue_field_counts = dict(issue_field_counter.most_common())

    return report


def format_report_text(report: QCDatasetReport) -> str:
    """将质检报告格式化为可打印的文本摘要。"""
    lines = [
        "=" * 60,
        "📋 数据集质检报告",
        "=" * 60,
        f"总样本数:     {report.total_samples}",
        f"通过 (无问题): {report.clean_samples}",
        f"警告:         {report.warning_samples}",
        f"错误 (不可用): {report.error_samples}",
        "",
        "--- 数据源分布 ---",
    ]
    for ds, cnt in report.dataset_distribution.items():
        lines.append(f"  {ds}: {cnt}")

    lines.append("\n--- 打标模型分布 ---")
    for model, cnt in report.model_distribution.items():
        lines.append(f"  {model}: {cnt}")

    lines.append("\n--- 来源分布 (Top 15) ---")
    for src, cnt in list(report.source_distribution.items())[:15]:
        lines.append(f"  {src}: {cnt}")

    lines.append("\n--- 高频问题字段 ---")
    for fld, cnt in list(report.issue_field_counts.items())[:10]:
        lines.append(f"  {fld}: {cnt} 条")

    lines.append("=" * 60)
    return "\n".join(lines)
