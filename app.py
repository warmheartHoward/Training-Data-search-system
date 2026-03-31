"""
Streamlit Dashboard：文物世界知识训练数据检索系统。

功能：
- 训练数据检索：上传评测文物图像 + 输入文物名称，检索训练集中的相似样本及其标注
- 批量评测：对整个 benchmark 文件夹批量检索，查看每个评测样本在训练集中的覆盖情况
- 数据质检：扫描 tar+JSONL 数据目录，多维度质检

启动方式：
    streamlit run app.py
"""
import os
import sys

import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.retrieval_service import RetrievalService, discover_versions
from data.data_scanner import scan_dataset
from data.tar_reader import load_image, load_image_bytes
from data.quality_checker import check_dataset, Severity
from configs.config import ModelConfig, IndexConfig, AppConfig
from utils.benchmark_loader import (
    scan_benchmark_folder, extract_model_keys, get_entity_name,
    extract_json_fields, get_field_value, preview_field_values,
)


# ============================================================================
#  全局样式
# ============================================================================

st.set_page_config(page_title="文物训练数据检索系统", page_icon="🏛️", layout="wide")

st.markdown("""
<style>
    .result-card {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        background: #fafafa;
    }
    .result-card.sim-high {
        border-left: 4px solid #ff4444;
        background: #fff5f5;
    }
    .result-card.sim-mid {
        border-left: 4px solid #ff8800;
        background: #fff8f0;
    }
    .result-card.sim-low {
        border-left: 4px solid #44bb44;
        background: #f5fff5;
    }
    .score-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 14px;
    }
    .score-high { background: #ff4444; color: white; }
    .score-mid  { background: #ff8800; color: white; }
    .score-low  { background: #44bb44; color: white; }
    .entity-name {
        font-size: 18px;
        font-weight: 600;
        color: #1a1a2e;
        margin: 8px 0 4px 0;
    }
    .source-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 8px;
        background: #e8eaf6;
        color: #3949ab;
        font-size: 12px;
    }
    .annotation-box {
        background: white;
        border: 1px solid #e8e8e8;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 8px;
        font-size: 14px;
        line-height: 1.7;
        max-height: 300px;
        overflow-y: auto;
        color: #333;
    }
    .bench-query-card {
        border: 2px solid #1976d2;
        border-radius: 12px;
        padding: 16px 20px;
        background: linear-gradient(135deg, #e3f2fd 0%, #f0f7ff 100%);
        margin-bottom: 16px;
    }
    .bench-query-card .entity-name {
        font-size: 22px;
        color: #0d47a1;
    }
    .bench-nav {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        padding: 8px 0;
    }
    .bench-stat-card {
        text-align: center;
        padding: 12px;
        border-radius: 10px;
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
    }
    .bench-stat-card .stat-value {
        font-size: 28px;
        font-weight: 700;
        color: #1a1a2e;
    }
    .bench-stat-card .stat-label {
        font-size: 13px;
        color: #666;
        margin-top: 4px;
    }
    .hidden-count {
        font-size: 13px;
        color: #999;
        font-style: italic;
        padding: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
#  辅助函数
# ============================================================================

def score_css_class(score: float) -> tuple[str, str]:
    """根据相似度分数返回卡片和徽章的 CSS 类名。"""
    if score > 0.9:
        return "sim-high", "score-high"
    elif score > 0.7:
        return "sim-mid", "score-mid"
    else:
        return "sim-low", "score-low"


@st.cache_resource(show_spinner="正在加载模型和索引 ...")
def get_service() -> RetrievalService:
    """全局单例：加载模型和 FAISS 索引（首次调用时执行，后续复用缓存）。"""
    return RetrievalService(ModelConfig(), IndexConfig(), AppConfig())


def render_result_card(result):
    """渲染单个检索结果卡片：图像 + 分数 + 实体名 + 来源 + 可展开的打标结果。"""
    card_cls, badge_cls = score_css_class(result.score)

    # 图片
    img_data = load_image_bytes(result.image_path)
    if img_data:
        st.image(img_data, use_container_width=True)
    else:
        st.warning(f"图像加载失败")

    # 分数 + 实体名 + 来源
    st.markdown(
        f'<div class="result-card {card_cls}">'
        f'  <span class="score-badge {badge_cls}">#{result.rank}&ensp;{result.score:.4f}</span>'
        f'  <div class="entity-name">{result.text or "(无实体名称)"}</div>'
        f'  <span class="source-tag">{result.source or "未知来源"}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 训练标注（可展开）
    if result.annotation:
        with st.expander("📖 查看训练标注", expanded=False):
            st.markdown(
                f'<div class="annotation-box">{result.annotation}</div>',
                unsafe_allow_html=True,
            )


# ============================================================================
#  Tab 导航
# ============================================================================

tab_retrieval, tab_bench, tab_qc = st.tabs([
    "🔍 训练数据检索", "📊 批量评测", "📋 数据质检"
])


# ============================================================================
#  Tab 1：训练数据检索
# ============================================================================

with tab_retrieval:
    st.header("文物训练数据检索")
    st.caption("上传评测文物图像 / 输入文物名称，检索训练集中的相似样本及其标注内容。"
               "相似度 >0.9 高度相似（红色），>0.7 中度相似（橙色），帮助分析模型知识来源。")

    # ---- 侧边栏 ----
    with st.sidebar:
        st.title("🏛️ 检索设置")

        # 数据版本选择
        _index_cfg = IndexConfig()
        _all_versions = discover_versions(_index_cfg.index_dir)
        if _all_versions:
            selected_versions = st.multiselect(
                "📦 数据版本",
                options=_all_versions,
                default=_all_versions,
                help="选择要检索的训练数据版本，可多选合并检索",
            )
        else:
            selected_versions = []
            st.warning("未发现索引版本，请先运行离线建库")

        st.markdown("---")
        uploaded_file = st.file_uploader(
            "📷 上传评测文物图像", type=["jpg", "jpeg", "png", "bmp", "webp"])
        query_text = st.text_input(
            "📝 输入文物名称",
            placeholder="例如：明代 掐丝珐琅胡人坐像")
        top_k = st.slider("🔢 Top-K 相似样本数", 1, 50, 5)
        st.markdown("---")
        search_clicked = st.button("🚀 开始检索", type="primary", use_container_width=True)

    # ---- Query 展示 ----
    query_image = None
    if uploaded_file or query_text:
        st.subheader("📌 当前评测样本")
        q_col1, q_col2 = st.columns([1, 2])
        if uploaded_file:
            query_image = Image.open(uploaded_file).convert("RGB")
            with q_col1:
                st.image(query_image, caption="评测文物图像", use_container_width=True)
        if query_text:
            with q_col2:
                st.markdown(f"### `{query_text}`")
        st.markdown("---")

    # ---- 检索执行 ----
    if search_clicked:
        if not uploaded_file and not query_text:
            st.warning("请上传文物图像或输入文物名称！")
            st.stop()

        service = get_service()
        col_img, col_txt = st.columns(2)

        # 以图搜图：视觉相似样本
        with col_img:
            st.subheader("🖼️ 视觉相似样本")
            if query_image:
                with st.spinner("图像检索中 ..."):
                    img_results = service.search_by_image(
                        query_image, top_k=top_k, versions=selected_versions)
                if not img_results:
                    st.info("训练集中未找到视觉相似的文物")
                else:
                    for r in img_results:
                        render_result_card(r)
                        st.markdown("")
            else:
                st.info("未上传图像，跳过视觉检索")

        # 以文搜文：名称相似样本
        with col_txt:
            st.subheader("📝 名称相似样本")
            if query_text:
                with st.spinner("文本检索中 ..."):
                    txt_results = service.search_by_text(
                        query_text, top_k=top_k, versions=selected_versions)
                if not txt_results:
                    st.info("训练集中未找到名称相似的文物")
                else:
                    for r in txt_results:
                        render_result_card(r)
                        st.markdown("")
            else:
                st.info("未输入名称，跳过文本检索")

    elif not uploaded_file and not query_text:
        st.info("👈 在左侧上传评测文物图像 / 输入文物名称，检索训练集中的相似样本")


# ============================================================================
#  Tab 2：数据质检
# ============================================================================

with tab_qc:
    st.header("📋 训练标注质检")
    st.caption("扫描训练数据目录（tar+JSONL），检查标注格式完整性和内容一致性。")

    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        qc_data_dir = st.text_input("数据根目录",
                                    placeholder="例如：/data/train_root")
    with col_c2:
        min_ann_len = st.number_input("打标结果最小长度", value=50, min_value=0)
    with col_c3:
        check_img = st.checkbox("验证图像可读性（慢）", value=False)

    qc_clicked = st.button("🔬 开始质检", type="primary")

    if qc_clicked:
        if not qc_data_dir or not os.path.isdir(qc_data_dir):
            st.error("请输入有效目录！")
            st.stop()

        with st.spinner("扫描中 ..."):
            annotations = scan_dataset(qc_data_dir, show_progress=False)
        if not annotations:
            st.error("未找到有效数据")
            st.stop()

        with st.spinner(f"质检 {len(annotations)} 条 ..."):
            qc_report = check_dataset(annotations, min_ann_len,
                                      check_image_readable=check_img)

        # ---- 概览 ----
        st.subheader("📊 概览")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("总样本", qc_report.total_samples)
        m2.metric("✅ 通过", qc_report.clean_samples)
        m3.metric("⚠️ 警告", qc_report.warning_samples)
        m4.metric("❌ 错误", qc_report.error_samples)
        st.markdown("---")

        # ---- 分布 ----
        st.subheader("📈 分布统计")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.markdown("**数据源**")
            for k, v in qc_report.dataset_distribution.items():
                st.write(f"- {k}: {v}")
        with d2:
            st.markdown("**打标模型**")
            for k, v in qc_report.model_distribution.items():
                st.write(f"- {k}: {v}")
        with d3:
            st.markdown("**来源 Top-10**")
            for k, v in list(qc_report.source_distribution.items())[:10]:
                st.write(f"- {k}: {v}")
        st.markdown("---")

        # ---- 高频问题 ----
        if qc_report.issue_field_counts:
            st.subheader("🔥 高频问题")
            for fld, cnt in list(qc_report.issue_field_counts.items())[:10]:
                pct = cnt / max(qc_report.total_samples, 1) * 100
                st.write(f"- `{fld}`: {cnt} 条 ({pct:.1f}%)")
            st.markdown("---")

        # ---- 问题明细 ----
        problems = [r for r in qc_report.sample_reports if not r.is_clean]
        if problems:
            st.subheader(f"🔎 问题样本 ({len(problems)} 条)")
            severity_filter = st.multiselect(
                "筛选", ["ERROR", "WARNING", "INFO"],
                default=["ERROR", "WARNING"])

            for rpt in problems[:200]:  # 限制渲染量
                filtered = [i for i in rpt.issues if i.severity.value in severity_filter]
                if not filtered:
                    continue
                has_err = any(i.severity == Severity.ERROR for i in filtered)
                icon = "❌" if has_err else "⚠️"
                with st.expander(f"{icon} {rpt.name}", expanded=False):
                    tc, dc = st.columns([1, 3])
                    with tc:
                        img = load_image(rpt.image_path)
                        if img:
                            st.image(img, width=150)
                        else:
                            st.caption("(无法加载)")
                    with dc:
                        st.caption(rpt.image_path)
                        for iss in filtered:
                            badge = {Severity.ERROR: "🔴", Severity.WARNING: "🟡",
                                     Severity.INFO: "🔵"}[iss.severity]
                            st.markdown(
                                f"{badge} **[{iss.severity.value}]** "
                                f"`{iss.field}` — {iss.message}")
        else:
            st.success("🎉 所有样本均通过质检！")


# ============================================================================
#  Tab 3：批量评测
# ============================================================================

with tab_bench:
    st.header("📊 Benchmark 批量评测")
    st.caption(
        "指定评测文件夹（图像 + 同名 JSON），批量检索每个评测样本在训练集中的覆盖情况。"
        "支持按相似度阈值过滤，逐样本浏览检索结果与训练标注。"
    )

    # ---- 控制栏 ----
    bench_c1, bench_c2 = st.columns([3, 1])
    with bench_c1:
        bench_folder = st.text_input(
            "📁 评测数据文件夹",
            placeholder="例如：/data/benchmark/cultural_relics",
            key="bench_folder_input",
        )
    with bench_c2:
        scan_clicked = st.button("🔍 扫描文件夹", use_container_width=True)

    # ---- 扫描文件夹 ----
    if scan_clicked and bench_folder:
        if not os.path.isdir(bench_folder):
            st.error("文件夹路径不存在！")
            st.stop()
        samples = scan_benchmark_folder(bench_folder)
        if not samples:
            st.warning("未找到有效的 图像+JSON 配对（要求同名的 .jpg/.png 和 .json）")
            st.stop()
        json_fields = extract_json_fields(samples)
        if not json_fields:
            st.warning("JSON 文件中未发现可用的字符串字段")
            st.stop()
        st.session_state["bench_samples"] = samples
        st.session_state["bench_json_fields"] = json_fields
        st.session_state.pop("bench_results", None)
        st.session_state["bench_current_idx"] = 0
        st.success(f"扫描完成：找到 {len(samples)} 个评测样本，{len(json_fields)} 个可选字段")

    # ---- 参数设置与批量检索 ----
    if "bench_samples" in st.session_state:
        samples = st.session_state["bench_samples"]
        json_fields = st.session_state["bench_json_fields"]

        # 第一行：字段选择 + 字段预览
        field_c1, field_c2 = st.columns([1, 2])
        with field_c1:
            selected_field = st.selectbox(
                "📝 文本检索字段",
                json_fields,
                help="选择 JSON 中哪个字段的内容用于语义文本检索",
                key="bench_field_select",
            )
        with field_c2:
            # 展示选中字段在前几个样本中的实际值，方便用户确认
            if selected_field:
                previews = preview_field_values(samples, selected_field, max_preview=3)
                if previews:
                    preview_str = " | ".join(f"`{v[:40]}`" for v in previews)
                    st.markdown(f"**字段预览：**{preview_str}", help="前几个样本中该字段的实际值")
                else:
                    st.caption("该字段在当前样本中无有效值")

        # 第二行：版本、Top-K、阈值、按钮
        p_c1, p_c2, p_c3, p_c4 = st.columns([2, 1, 1, 1])
        with p_c1:
            _bench_versions = discover_versions(IndexConfig().index_dir)
            bench_versions = st.multiselect(
                "📦 数据版本", options=_bench_versions,
                default=_bench_versions, key="bench_versions")
        with p_c2:
            bench_top_k = st.slider("Top-K", 1, 50, 5, key="bench_topk")
        with p_c3:
            bench_threshold = st.slider("相似度阈值", 0.0, 1.0, 0.5, 0.05,
                                        key="bench_threshold")
        with p_c4:
            st.markdown("<br>", unsafe_allow_html=True)
            run_clicked = st.button("🚀 开始批量检索", type="primary",
                                    use_container_width=True)

        # ---- 执行批量检索 ----
        if run_clicked:
            service = get_service()
            results = {}
            progress_bar = st.progress(0, text="批量检索中 ...")
            status_text = st.empty()

            for i, sample in enumerate(samples):
                entity_name = get_field_value(sample, selected_field)

                # 图像检索
                try:
                    img = Image.open(sample["image_path"]).convert("RGB")
                    img_results = service.search_by_image(
                        img, top_k=bench_top_k, versions=bench_versions)
                except Exception:
                    img_results = []

                # 文本检索
                txt_results = []
                if entity_name:
                    try:
                        txt_results = service.search_by_text(
                            entity_name, top_k=bench_top_k, versions=bench_versions)
                    except Exception:
                        txt_results = []

                results[sample["stem"]] = {
                    "entity_name": entity_name or "",
                    "image_results": img_results,
                    "text_results": txt_results,
                }

                progress_bar.progress(
                    (i + 1) / len(samples),
                    text=f"检索中 ... {i + 1}/{len(samples)} — {sample['stem']}"
                )

            progress_bar.empty()
            status_text.empty()
            st.session_state["bench_results"] = results
            st.session_state["bench_current_idx"] = 0

        # ---- 展示结果 ----
        if "bench_results" in st.session_state:
            results = st.session_state["bench_results"]
            threshold = st.session_state.get("bench_threshold", 0.5)

            # ---- 总览仪表盘 ----
            st.markdown("---")
            st.subheader("📈 覆盖度总览")

            total = len(results)
            img_hit = 0   # 图像最高分 >= 阈值
            txt_hit = 0   # 文本最高分 >= 阈值
            both_hit = 0  # 双匹配
            none_hit = 0  # 无匹配

            for stem, r in results.items():
                best_img = r["image_results"][0].score if r["image_results"] else 0.0
                best_txt = r["text_results"][0].score if r["text_results"] else 0.0
                i_hit = best_img >= threshold
                t_hit = best_txt >= threshold
                if i_hit:
                    img_hit += 1
                if t_hit:
                    txt_hit += 1
                if i_hit and t_hit:
                    both_hit += 1
                if not i_hit and not t_hit:
                    none_hit += 1

            s1, s2, s3, s4, s5 = st.columns(5)
            s1.markdown(
                f'<div class="bench-stat-card">'
                f'<div class="stat-value">{total}</div>'
                f'<div class="stat-label">总样本数</div></div>',
                unsafe_allow_html=True)
            s2.markdown(
                f'<div class="bench-stat-card">'
                f'<div class="stat-value" style="color:#ff4444">{img_hit}</div>'
                f'<div class="stat-label">图像匹配 ({img_hit*100//max(total,1)}%)</div></div>',
                unsafe_allow_html=True)
            s3.markdown(
                f'<div class="bench-stat-card">'
                f'<div class="stat-value" style="color:#ff8800">{txt_hit}</div>'
                f'<div class="stat-label">文本匹配 ({txt_hit*100//max(total,1)}%)</div></div>',
                unsafe_allow_html=True)
            s4.markdown(
                f'<div class="bench-stat-card">'
                f'<div class="stat-value" style="color:#d32f2f">{both_hit}</div>'
                f'<div class="stat-label">双重匹配 ({both_hit*100//max(total,1)}%)</div></div>',
                unsafe_allow_html=True)
            s5.markdown(
                f'<div class="bench-stat-card">'
                f'<div class="stat-value" style="color:#44bb44">{none_hit}</div>'
                f'<div class="stat-label">无匹配 ({none_hit*100//max(total,1)}%)</div></div>',
                unsafe_allow_html=True)

            # ---- 总览表格 ----
            st.markdown("")
            import pandas as pd
            table_rows = []
            for stem, r in results.items():
                best_img = r["image_results"][0].score if r["image_results"] else 0.0
                best_txt = r["text_results"][0].score if r["text_results"] else 0.0
                i_hit = best_img >= threshold
                t_hit = best_txt >= threshold
                status = "🔴 双匹配" if (i_hit and t_hit) else \
                         "🟠 图像匹配" if i_hit else \
                         "🟡 文本匹配" if t_hit else "🟢 无匹配"
                table_rows.append({
                    "样本": stem,
                    "文物名称": r["entity_name"] or "-",
                    "最高图像相似度": round(best_img, 4),
                    "最高文本相似度": round(best_txt, 4),
                    "覆盖状态": status,
                })

            df = pd.DataFrame(table_rows)
            df = df.sort_values("最高图像相似度", ascending=False).reset_index(drop=True)

            with st.expander("📋 查看全部样本总览表", expanded=False):
                st.dataframe(
                    df,
                    use_container_width=True,
                    height=min(400, 35 * len(df) + 38),
                )

            # ---- 逐样本浏览器 ----
            st.markdown("---")
            st.subheader("🔎 逐样本浏览")

            sample_stems = list(results.keys())
            num_samples = len(sample_stems)

            # 导航
            nav_c1, nav_c2, nav_c3, nav_c4 = st.columns([1, 1, 4, 1])
            with nav_c1:
                if st.button("⬅️ 上一个", use_container_width=True, key="bench_prev"):
                    idx = st.session_state.get("bench_current_idx", 0)
                    st.session_state["bench_current_idx"] = max(0, idx - 1)
            with nav_c2:
                if st.button("下一个 ➡️", use_container_width=True, key="bench_next"):
                    idx = st.session_state.get("bench_current_idx", 0)
                    st.session_state["bench_current_idx"] = min(num_samples - 1, idx + 1)
            with nav_c3:
                jump_idx = st.number_input(
                    "跳转到", min_value=1, max_value=num_samples,
                    value=st.session_state.get("bench_current_idx", 0) + 1,
                    key="bench_jump",
                )
                st.session_state["bench_current_idx"] = jump_idx - 1
            with nav_c4:
                st.markdown(
                    f"<div style='text-align:center;padding-top:28px;color:#666;'>"
                    f"共 {num_samples} 个样本</div>",
                    unsafe_allow_html=True)

            current_idx = st.session_state.get("bench_current_idx", 0)
            current_stem = sample_stems[current_idx]
            current_result = results[current_stem]

            # 找到对应的样本信息
            current_sample = next(
                (s for s in samples if s["stem"] == current_stem), None
            )

            # ---- 当前样本卡片 ----
            st.markdown(f"### 样本 {current_idx + 1}/{num_samples}：`{current_stem}`")

            query_col1, query_col2 = st.columns([1, 2])
            with query_col1:
                if current_sample:
                    try:
                        bench_img = Image.open(current_sample["image_path"]).convert("RGB")
                        st.image(bench_img, caption="评测文物图像", use_container_width=True)
                    except Exception:
                        st.warning("图像加载失败")
            with query_col2:
                entity = current_result["entity_name"]
                best_img_score = (current_result["image_results"][0].score
                                  if current_result["image_results"] else 0.0)
                best_txt_score = (current_result["text_results"][0].score
                                  if current_result["text_results"] else 0.0)

                st.markdown(
                    f'<div class="bench-query-card">'
                    f'  <div class="entity-name">{entity or "(无实体名称)"}</div>'
                    f'  <div style="margin-top:12px;font-size:14px;color:#555;">'
                    f'    最高图像相似度：<b>{best_img_score:.4f}</b> &emsp; '
                    f'    最高文本相似度：<b>{best_txt_score:.4f}</b>'
                    f'  </div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # ---- 检索结果：左右两列 ----
            res_col_img, res_col_txt = st.columns(2)

            with res_col_img:
                st.markdown("#### 🖼️ 视觉相似样本")
                img_results = current_result["image_results"]
                shown = [r for r in img_results if r.score >= threshold]
                hidden = len(img_results) - len(shown)
                if shown:
                    for r in shown:
                        render_result_card(r)
                        st.markdown("")
                else:
                    st.info("无超过阈值的视觉相似结果")
                if hidden > 0:
                    st.markdown(
                        f'<div class="hidden-count">'
                        f'已隐藏 {hidden} 条低于阈值 ({threshold:.2f}) 的结果</div>',
                        unsafe_allow_html=True)

            with res_col_txt:
                st.markdown("#### 📝 名称相似样本")
                txt_results = current_result["text_results"]
                shown = [r for r in txt_results if r.score >= threshold]
                hidden = len(txt_results) - len(shown)
                if shown:
                    for r in shown:
                        render_result_card(r)
                        st.markdown("")
                elif not entity:
                    st.info("该样本无实体名称，跳过文本检索")
                else:
                    st.info("无超过阈值的文本相似结果")
                if hidden > 0:
                    st.markdown(
                        f'<div class="hidden-count">'
                        f'已隐藏 {hidden} 条低于阈值 ({threshold:.2f}) 的结果</div>',
                        unsafe_allow_html=True)
