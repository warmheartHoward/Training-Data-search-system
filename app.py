"""
Streamlit Dashboard：文物世界知识训练数据检索系统。

功能：
- 训练数据检索：上传评测文物图像 + 输入文物名称，检索训练集中的相似样本及其标注
- 数据质检：扫描 tar+JSONL 数据目录，多维度质检

启动方式：
    streamlit run app.py
"""
import os
import sys

import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.retrieval_service import RetrievalService
from data.data_scanner import scan_dataset
from data.tar_reader import load_image, load_image_bytes
from data.quality_checker import check_dataset, Severity
from configs.config import ModelConfig, IndexConfig, AppConfig


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

tab_retrieval, tab_qc = st.tabs(["🔍 训练数据检索", "📋 数据质检"])


# ============================================================================
#  Tab 1：泄露检索
# ============================================================================

with tab_retrieval:
    st.header("文物训练数据检索")
    st.caption("上传评测文物图像 / 输入文物名称，检索训练集中的相似样本及其标注内容。"
               "相似度 >0.9 高度相似（红色），>0.7 中度相似（橙色），帮助分析模型知识来源。")

    @st.cache_resource(show_spinner="正在加载模型和索引 ...")
    def get_service() -> RetrievalService:
        return RetrievalService(ModelConfig(), IndexConfig(), AppConfig())

    # ---- 侧边栏 ----
    with st.sidebar:
        st.title("🏛️ 检索设置")
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
                    img_results = service.search_by_image(query_image, top_k=top_k)
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
                    txt_results = service.search_by_text(query_text, top_k=top_k)
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
