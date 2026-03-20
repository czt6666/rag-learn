# 多模态 Graph-RAG 系统 - 安装和使用指南

## 📦 文件说明

下载的压缩包包含完整的项目代码。
 
## 🚀 快速安装（3 步）

### 步骤 1: 解压文件

```bash
tar -xzf multimodal_graph_rag.tar.gz
cd multimodal_graph_rag
```

### 步骤 2: 安装依赖

```bash
# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 安装 CLIP（可选，用于真实的多模态嵌入）
pip install git+https://github.com/openai/CLIP.git
```

### 步骤 3: 配置环境

```bash
# 复制配置文件
cp .env.example .env

# 编辑配置文件
nano .env  # 或使用其他编辑器
```

**必须配置的项目：**

```env
# Neo4j 配置（如果使用 Docker）
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# API Key（至少配置一个）
OPENAI_API_KEY=sk-your-key-here
# 或
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

## 🎯 快速测试（无需真实数据）

### 生成测试数据

```bash
python generate_sample_data.py
```

这会在 `./data/mramg_bench/` 创建示例数据：

- `docs.jsonl` - 3个示例文档
- `qa.jsonl` - 2个问答对
- `images/` - 图像目录（空）

### 启动 Neo4j（如果本地没有）

**使用 Docker（推荐）：**

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:latest
```

**或安装 Neo4j Desktop：**
访问 https://neo4j.com/download/

### 运行系统

```bash
# 完整流程（构建 + 查询）
python main.py --mode full --llm openai

# 系统会：
# 1. 加载数据
# 2. 解析文档
# 3. 构建向量索引
# 4. 构建知识图谱
# 5. 进入交互式查询模式
```

**交互示例：**

```
请输入问题: 什么是人工智能？

==================================================
问题: 什么是人工智能？
==================================================

答案:
人工智能（AI）是计算机科学的一个分支，致力于创建能够
模拟人类智能的系统...

检索统计: 向量 3 个, 图谱 2 个
==================================================
```

## 📚 项目结构

```
multimodal_graph_rag/
├── config.py                           # 配置管理
├── module_0_data_loader.py             # 数据加载
├── module_1_parser.py                  # 文档解析
├── module_2_vector_index.py            # 向量索引（CLIP + ChromaDB）
├── module_3_knowledge_graph.py         # 知识图谱（LLM + Neo4j）
├── module_4_7_retrieval_generation.py  # 检索融合与生成
├── module_8_evaluation.py              # 评估系统
├── main.py                             # 主程序
├── generate_sample_data.py             # 测试数据生成器
├── requirements.txt                    # Python 依赖
├── .env.example                        # 配置模板
└── README.md                           # 说明文档
```

## 🎮 使用模式

### 模式 1: 完整流程（推荐首次使用）

```bash
python main.py --mode full --llm openai
```

自动完成：数据加载 → 解析 → 索引构建 → 图谱构建 → 交互查询

### 模式 2: 仅构建索引

```bash
python main.py --mode build --llm openai --reset
```

适用于：更新数据后重建索引

### 模式 3: 仅查询（使用已有索引）

```bash
python main.py --mode query --llm openai
```

适用于：索引已存在，直接查询

### 模式 4: 系统评估

```bash
python main.py --mode evaluate --llm openai --eval-sample 10
```

适用于：评估系统性能

## 🔧 Python API 使用

```python
from module_1_parser import DocumentParser
from module_2_vector_index import VectorIndexBuilder
from module_3_knowledge_graph import GraphBuilder
from module_4_7_retrieval_generation import MultimodalRAG

# 1. 准备数据
docs = [
    {
        "doc_id": "doc_001",
        "text": "人工智能是计算机科学的分支。",
        "images": []
    }
]

# 2. 解析文档
parser = DocumentParser()
chunks = parser.parse_all_documents(docs)

# 3. 构建向量索引
vector_builder = VectorIndexBuilder()
vector_builder.build_index(chunks)

# 4. 构建知识图谱
graph_builder = GraphBuilder(llm_provider="openai")
graph_builder.build_graph(chunks)

# 5. 创建 RAG 系统
rag = MultimodalRAG(
    vector_builder=vector_builder,
    knowledge_graph=graph_builder.kg,
    llm_provider="openai"
)

# 6. 查询
result = rag.query("什么是人工智能？")
print(result['answer'])
```

## 📊 使用真实数据集

如果你有 MRAMG-Bench 或其他数据集：

1. 将数据放到 `./data/mramg_bench/` 目录：
   ```
   data/mramg_bench/
   ├── docs.jsonl          # 文档数据
   ├── qa.jsonl            # 问答对
   └── images/             # 图像文件
       ├── img_001.jpg
       └── ...
   ```

2. 运行系统：
   ```bash
   python main.py --mode full --llm openai --reset
   ```

## ⚠️ 常见问题

### Q: 没有 CLIP 可以运行吗？

**A:** 可以！系统会自动使用虚拟嵌入（随机向量）用于测试。如果要获得真实效果，需要安装 CLIP：

```bash
pip install git+https://github.com/openai/CLIP.git
```

### Q: Neo4j 连接失败？

**A:**

1. 确保 Neo4j 正在运行
2. 检查 `.env` 中的配置
3. 使用 Docker 快速启动：
   ```bash
   docker run -d -p 7474:7474 -p 7687:7687 \
     -e NEO4J_AUTH=neo4j/password neo4j
   ```

### Q: API 费用担心？

**A:**

- 使用示例数据测试（仅 3 个文档）成本极低
- 使用 `--eval-sample 10` 限制评估数量
- 优先使用 `gpt-4o-mini`（已默认）

### Q: 内存不够？

**A:**

- 在 `module_2_vector_index.py` 中设置 `device = "cpu"`
- 减少批处理大小
- 使用更少的数据测试

## 🎓 学习路径

1. **第 1 天：** 运行示例数据，理解整体流程
2. **第 2 天：** 阅读各模块代码，理解实现细节
3. **第 3 天：** 使用自己的数据，调整参数
4. **第 4 天：** 扩展功能，如添加新的检索策略

## 📞 获取帮助

```bash
# 查看命令行帮助
python main.py --help

# 查看详细日志
python main.py --log-level DEBUG --mode full
```

## 🎉 开始使用

```bash
# 一键开始
python generate_sample_data.py && python main.py --mode full --llm openai
```

祝使用愉快！
