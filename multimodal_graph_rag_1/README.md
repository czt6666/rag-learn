# 多模态 Graph-RAG 系统

完整的多模态知识图谱增强检索生成系统。

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

### 2. 配置环境

```bash
cp .env.example .env
# 编辑 .env 填入 API keys 和 Neo4j 配置
```

### 3. 生成测试数据

```bash
python generate_sample_data.py
```

### 4. 运行系统

```bash
# 完整流程（构建索引 + 查询）
python main.py --mode full --llm openai

# 仅构建索引
python main.py --mode build --reset

# 仅查询
python main.py --mode query

# 评估
python main.py --mode evaluate --eval-sample 10
```

## 📦 模块说明

- **module_0_data_loader.py** - 数据加载
- **module_1_parser.py** - 文档解析
- **module_2_vector_index.py** - CLIP + ChromaDB
- **module_3_knowledge_graph.py** - LLM + Neo4j
- **module_4_7_retrieval_generation.py** - 检索融合与生成
- **module_8_evaluation.py** - 评估系统

## 🔧 使用方式

### Python API

```python
from module_1_parser import DocumentParser
from module_2_vector_index import VectorIndexBuilder
from module_3_knowledge_graph import GraphBuilder
from module_4_7_retrieval_generation import MultimodalRAG

# 解析文档
parser = DocumentParser()
chunks = parser.parse_all_documents(docs)

# 构建索引
vector_builder = VectorIndexBuilder()
vector_builder.build_index(chunks)

# 构建图谱
graph_builder = GraphBuilder(llm_provider="openai")
graph_builder.build_graph(chunks)

# 创建 RAG
rag = MultimodalRAG(vector_builder, graph_builder.kg)

# 查询
result = rag.query("什么是人工智能？")
print(result['answer'])
```

## ⚙️ 配置项

- `NEO4J_URI` - Neo4j 地址
- `NEO4J_PASSWORD` - Neo4j 密码
- `OPENAI_API_KEY` - OpenAI API Key
- `VECTOR_TOP_K` - 检索数量 (默认 5)
- `GRAPH_MAX_HOPS` - 图遍历跳数 (默认 2)

## 📊 评估指标

- Exact Match - 精确匹配
- Contains Match - 包含匹配
- Token F1 - 词汇重叠 F1

## 🛠️ 技术栈

| 组件 | 技术 |
|------|------|
| 多模态嵌入 | CLIP |
| 向量数据库 | ChromaDB |
| 图数据库 | Neo4j |
| LLM | OpenAI/Anthropic |

## 📄 许可

MIT License
