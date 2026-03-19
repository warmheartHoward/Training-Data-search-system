# 文物世界知识训练数据检索系统

评测大模型文物世界知识能力时的辅助工具。支持单张交互式检索和整个 Benchmark 批量检索两种模式，快速查看评测文物在训练集中是否有相同或相似的样本，并直接对比训练标注内容，帮助分析模型输出的知识来源。

## 核心用途

- **训练覆盖度分析**：评测某文物时，检索训练集中有没有见过同一件文物
- **相似样本溯源**：查找视觉上相似的文物（如同类器型、同一窑口），分析它们的训练标注是否会影响模型输出
- **标注内容对比**：直接查看检索命中样本的完整打标结果，判断模型可能从哪些训练数据中获取了知识
- **Benchmark 批量评测**：对整个评测集一键批量检索，生成覆盖度统计报告，逐样本浏览检索结果

## 系统架构

```
┌──────────────────────────────────────────────────────────────────┐
│                       离线建库 (Offline)                          │
│                                                                  │
│  tar+JSONL 目录 ──► 数据扫描器 ──► 质检 ──► 多GPU特征提取 ──► FAISS索引  │
│  (多数据源)       tar_reader    QC      8×4090并行       (图像/文本) │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│                       在线检索 (Online)                           │
│                                                                  │
│  单张检索：评测图像/名称 ──► 特征提取 ──► Top-K ──► 可视化对比          │
│  批量评测：Benchmark 文件夹 ──► 逐样本检索 ──► 覆盖度报告 + 逐样本浏览  │
└──────────────────────────────────────────────────────────────────┘
```

## 目录结构

```
ImageSearchSystem/
├── configs/
│   └── config.py              # 全局配置
├── models/
│   ├── vision_encoder.py      # DINOv2-Large 视觉编码器（实例级文物特征）
│   └── text_encoder.py        # BGE-M3 文本编码器（文物名称语义匹配）
├── indexing/
│   └── index_manager.py       # FAISS 索引管理
├── data/
│   ├── tar_reader.py          # tar 包图像读取工具（tar URI 协议）
│   ├── data_scanner.py        # tar+JSONL 数据扫描器
│   └── quality_checker.py     # 标注质检模块
├── services/
│   └── retrieval_service.py   # 在线检索服务
├── utils/
│   └── benchmark_loader.py    # Benchmark 评测文件夹解析工具
├── indexes/                   # 离线建库产出
├── offline_build_index.py     # 多GPU离线建库脚本
├── app.py                     # Streamlit Web UI（三个功能页）
├── requirements.txt
└── README.md
```

## 数据格式

### 训练数据（离线建库用）

```
root_dir/
├── dataset_source_A/              ← 不同来源的数据集
│   ├── images/
│   │   ├── data_000000.tar        ← 图像 tar 包
│   │   └── ...
│   └── jsonl/
│       ├── data_000000.jsonl      ← 与 tar 同名的标注 JSONL
│       └── ...
├── dataset_source_B/
│   ├── images/
│   └── jsonl/
└── ...
```

每条 JSONL 记录中提取的关键字段：

| 字段路径 | 用途 |
|---------|------|
| `data[0].content[*].image.relative_path` | tar 包内的图像相对路径 |
| `meta_info_image.knowledge_info.knowledge_entities[0].entity_name` | 文物实体名称（文本检索键） |
| `data[1].content[0].text.string` | 训练标注内容（检索命中后展示） |
| `meta_info_image.source_info.source_name` | 数据来源 |
| `data_generated_info.task_oriented_data[0].model_name` | 打标模型 |

系统使用自定义 tar URI 直接从 tar 包中读取图像，**无需解压到磁盘**：

```
tar:///absolute/path/to/data_000000.tar::珐琅/明代 掐丝珐琅胡人坐像.jpg
```

### Benchmark 评测数据（批量评测用）

```
benchmark_dir/
├── image_001.jpg
├── image_001.json       ← 与图像同名的 JSON
├── image_002.png
├── image_002.json
└── ...
```

JSON 格式：包含一个或多个模型的 QA 结果，系统从中提取 `answer` 作为文物名称进行文本检索。

```json
{
  "Qwen3-VL-235B_V2": [
    {"question": "图片中的这件衣服叫什么名字？", "answer": "午后系领狐毛饰边外套"}
  ],
  "GPT-4o": [
    {"question": "...", "answer": "..."}
  ]
}
```

## 技术栈

| 组件 | 选型 | 用途 |
|------|------|------|
| Vision Model | `facebook/dinov2-large` (1024d) | 实例级文物图像特征，判断是否为同一件文物 |
| Text Model | `BAAI/bge-m3` (1024d) | 中英文语义检索，按文物名称匹配 |
| Vector DB | FAISS (IndexFlatIP / IndexHNSW) | 高性能向量相似度检索 |
| Web UI | Streamlit | 交互式检索与标注可视化 |
| 并行框架 | torch.multiprocessing | 8 卡并行特征提取 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 离线建库

```bash
# 基础用法：8 卡并行建库
python offline_build_index.py \
    --data_dir /path/to/root_dir \
    --num_gpus 8

# 按版本建库（推荐）：不同数据集各自生成独立索引
python offline_build_index.py --data_dir /data/museum_v1   --version museum_v1
python offline_build_index.py --data_dir /data/auction_v2  --version auction_v2
python offline_build_index.py --data_dir /data/wiki_v3     --version wiki_v3

# 跳过质检
python offline_build_index.py --data_dir /path/to/root_dir --version v1 --skip_qc

# 大数据量用 HNSW 近似索引
python offline_build_index.py --data_dir /path/to/root_dir --version v1 --index_type hnsw
```

建库产出（多版本目录结构）：
```
indexes/
├── museum_v1/
│   ├── image_index.faiss
│   ├── text_index.faiss
│   ├── image_metadata.pkl
│   └── text_metadata.pkl
├── auction_v2/
│   ├── image_index.faiss
│   └── ...
└── wiki_v3/
    └── ...
```

> 不指定 `--version` 时索引直接存到 `indexes/` 根目录（兼容旧格式，版本名显示为 `default`）。

### 3. 启动 Web UI

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

远程访问可用 SSH 端口转发：

```bash
ssh -L 8501:localhost:8501 user@server
# 本地浏览器访问 http://localhost:8501
```

## 功能说明

### Tab 1：训练数据检索

单张交互式检索，适合逐个排查评测样本。

1. 侧边栏选择数据版本（可多选合并检索）
2. 上传评测文物图像、输入文物名称
3. 点击「开始检索」
4. 左列：视觉相似样本 Top-K（以图搜图）
5. 右列：名称相似样本 Top-K（以文搜文）
6. 每张结果卡片显示相似度（红 >0.9 / 橙 >0.7 / 绿）、文物名称、数据来源（多版本时前缀显示版本名）
7. 展开「查看训练标注」可查看该样本的完整打标内容

### Tab 2：批量评测

对整个 Benchmark 文件夹一键批量检索，适合系统性评估训练集覆盖度。

1. 输入 Benchmark 文件夹路径 → 扫描 → 自动发现可用模型
2. 选择模型（如 `Qwen3-VL-235B_V2`）和数据版本，设置 Top-K 和相似度阈值
3. 点击「开始批量检索」，自动遍历所有评测样本（带进度条）
4. **覆盖度总览**：5 个统计指标 — 总样本数 / 图像匹配数 / 文本匹配数 / 双重匹配数 / 无匹配数
5. **总览表格**：每个样本的最高相似度和覆盖状态，可排序
6. **逐样本浏览器**：Prev/Next 导航或跳转，查看每个评测样本的 Top-K 检索结果
7. **阈值实时过滤**：调整阈值滑块即时生效，不需要重新检索

### Tab 3：数据质检

扫描训练数据目录，检查标注格式和内容质量。

1. 输入训练数据根目录
2. 自动扫描所有 tar+JSONL 配对
3. 展示数据分布（数据源、打标模型、来源）和标注质量问题明细

## 数据版本管理

系统支持多版本索引的独立构建和合并检索：

- **建库时** 通过 `--version` 指定版本名，索引存入独立子目录
- **检索时** UI 自动扫描 `indexes/` 下所有版本，支持多选合并检索
- 多版本合并检索时，每个版本各自返回 Top-K，系统按分数全局排序后取最终 Top-K
- 结果卡片的数据来源字段会自动标注版本名（如 `[museum_v1] 上海博物馆`）

典型使用场景：
```
# 博物馆数据更新了新一批，单独建库
python offline_build_index.py --data_dir /data/museum_2026Q1 --version museum_2026Q1

# UI 中同时勾选旧版和新版，合并检索
```

## 配置调整

编辑 `configs/config.py`：

```python
# 小显存 GPU 降低 batch_size
ModelConfig.vision_batch_size = 16
ModelConfig.text_batch_size = 64

# 在线检索使用其他 GPU
AppConfig.retrieval_device = "cuda:1"

# 离线环境使用本地模型
ModelConfig.vision_model_name = "/models/dinov2-large"
ModelConfig.text_model_name = "/models/bge-m3"
```
