# 文物世界知识训练数据检索系统

评测大模型文物世界知识能力时的辅助工具。上传评测文物图像或输入文物名称，快速检索训练集中是否包含相同或相似的文物样本，查看对应的训练标注内容，帮助分析模型输出的知识来源和潜在干扰项。

## 核心用途

- **训练覆盖度分析**：评测某文物时，检索训练集中有没有见过同一件文物
- **相似样本溯源**：查找视觉上相似的文物（如同类器型、同一窑口），分析它们的训练标注是否会影响模型对当前测试样本的输出
- **标注内容对比**：直接查看检索命中样本的完整打标结果，判断模型可能从哪些训练数据中获取了知识

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
│  评测文物图像/名称 ──► 特征提取 ──► FAISS Top-K ──► Streamlit 可视化    │
│  (用户上传)         DINOv2/BGE    (单卡)       (相似样本+标注对比)  │
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
├── indexes/                   # 离线建库产出
├── offline_build_index.py     # 多GPU离线建库脚本
├── app.py                     # Streamlit Web UI
├── requirements.txt
└── README.md
```

## 训练数据格式

```
root_dir/
├── dataset_source_A/              ← 不同来源的数据集
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
```

每条 JSONL 记录中提取的关键字段：

| 字段路径 | 用途 |
|---------|------|
| `data[0].content[*].image.relative_path` | tar 包内的图像相对路径 |
| `meta_info_image.knowledge_info.knowledge_entities[0].entity_name` | 文物实体名称（以文搜文的检索键） |
| `data[1].content[0].text.string` | 训练标注内容（检索命中后展示） |
| `meta_info_image.source_info.source_name` | 数据来源 |
| `data_generated_info.task_oriented_data[0].model_name` | 打标模型 |

### tar URI 协议

系统使用自定义 tar URI 直接从 tar 包中读取图像，**无需解压到磁盘**：

```
tar:///absolute/path/to/data_000000.tar::珐琅/明代 掐丝珐琅胡人坐像.jpg
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
# 8 卡并行建库
python offline_build_index.py \
    --data_dir /path/to/root_dir \
    --num_gpus 8 \
    --index_dir indexes

# 跳过质检
python offline_build_index.py --data_dir /path/to/root_dir --skip_qc

# 大数据量用 HNSW 近似索引
python offline_build_index.py --data_dir /path/to/root_dir --index_type hnsw
```

建库产出：
```
indexes/
├── image_index.faiss      # 图像向量索引
├── text_index.faiss       # 文本向量索引
├── image_metadata.pkl     # 图像元信息（含完整训练标注）
└── text_metadata.pkl      # 文本元信息
```

### 3. 启动 Web UI

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

远程访问可用 SSH 端口转发：

```bash
ssh -L 8501:localhost:8501 user@server
# 本地浏览器访问 http://localhost:8501
```

### 功能说明

**训练数据检索页**：
1. 上传评测文物图像、输入文物名称
2. 点击「开始检索」
3. 左列：以图搜图 Top-K（视觉相似的训练样本）
4. 右列：以文搜文 Top-K（名称语义相似的训练样本）
5. 每张结果卡片显示相似度（红/橙/绿）、文物名称、数据来源
6. 展开「查看训练标注」可查看该样本的完整打标内容

**数据质检页**：
1. 输入训练数据根目录
2. 自动扫描所有 tar+JSONL 配对
3. 展示数据分布和标注质量问题

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
