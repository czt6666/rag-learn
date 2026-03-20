# 🏗️ Multi-modal Graph-RAG 系统工程模块清单

### **模块 0：数据拉取**

* 下载 MRAMG‑Bench 数据集（文档 JSONL、QA JSONL、图像 metadata）
* 保存至本地目录，形成统一路径索引

---

### **模块 1：数据解析**

* 读取文档 JSONL
* 按段落或句子分割文本
* 检测 `<PIC>` 占位符，关联对应 image_id
* 提取 image_caption 字段作为图像文本描述
* 组合每个段落文本 + 图像 caption 成 **multimodal chunk**
* 输出 JSONL 或 Python dict：

```python
{
    "doc_id": ...,
    "chunk_id": ...,
    "text": "...",
    "image_ids": [...],
    "image_captions": [...],
    "joined_text": "文本 + 图像描述"
}
```

---

### **模块 2：多模态嵌入与向量索引**

* 使用 **CLIP** 生成每个 chunk 的文本 + 图像 caption embedding
* 存入 **ChromaDB**，包括 metadata（doc_id, chunk_id, image_ids）

---

### **模块 3：知识图构建**

* 使用 LLM + Prompt 处理每个 chunk 的 `joined_text`
* 提取实体列表和实体间关系
* 将实体节点和关系边写入 **Neo4j**
* 每个节点包含 metadata（来源 doc_id, chunk_id, image_ids）

---

### **模块 4：向量检索**

* 接收用户 query
* 使用 CLIP 编码 query，检索 **ChromaDB** top-K chunk
* 返回每个 chunk 的文本、图像 caption、metadata

---

### **模块 5：图检索**

* 从向量检索结果中提取实体
* 在 **Neo4j** 中查找实体多跳路径
* 返回路径节点和关系边

---

### **模块 6：检索融合**

* 将向量检索结果和图检索结果合并成统一候选集
* 输出结构：

```python
{
    "vector_hits": [...],
    "graph_hits": [...],
    "merged_hits": [...],
}
```

---

### **模块 7：自然语言整合与生成**

* 组织融合后的 hits 为 LLM 输入
* Prompt 示例：

```
Query: {user_query}

Vector hits:
{vector_hits_text}

Graph paths:
{graph_hits_text}

Instruction:
Use the above information to generate a complete answer in natural language. Reference images if relevant.
```

* 调用 **LangChain + LLM** 输出自然语言回答

---

### **模块 8：评估**

* 根据 MRAMG‑Bench ground truth 对答案进行评分
* 可记录上下文覆盖率、图像引用准确性等



