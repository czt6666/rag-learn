# 归一化

## 为什么需要 L2 归一化



1. 消除模长的影响（这是核心）

可能长句子语义不相关，但是模长长，导致相似度高

不归一化会导致 “这个向量大” ≠ “这个向量语义更相关”

2. 点击 = 相似度

在工程上非常重要，方便向量数据库查询

3. 让查询结果稳定
4. CLIP本质就是比较“球面几何”



## 其他归一化方法

### L1 归一化

$$
\|\mathbf{x}\|_1 = |x_1| + |x_2| + \dots
$$

特点：

- 所有分量绝对值和 = 1
- 常见于 **概率分布 / 稀疏表示**
- ❌ 不适合语义 embedding

------

### Max 归一化

$$
x' = \frac{x - \min}{\max - \min}
$$

- 常见于传统 ML / 特征工程
- 对 embedding **基本没用**
- 对异常值极其敏感

------

### Z-score 标准化

$$
x' = \frac{x - \mu}{\sigma}
$$

- 常见于统计建模
- Transformer 内部靠 **LayerNorm**
- ❌ embedding 检索几乎不用

------

### BatchNorm / LayerNorm

这类：

- 是 **模型训练稳定手段**
- ❗ **不是 embedding 后处理**
- 不解决向量相似度问题

------

### Whitening / PCA Whitening

用于：

- 消除 embedding 各维度相关性
- 学术论文 / 高端搜索系统

缺点：

- 复杂
- 对 CLIP / SBERT **收益有限**
- 工程中很少做





# Neo4j

## 节点

**Chunk**

```
Chunk = 一段文本 + 对应的图像描述
```

原始文本或图片描述，直接为给LLM的东西，经过 **Spliter** 划分之后的内容

是文档内容的切片

👉 就像一本书里的段落。



**Entity**

> 实体
>
> (:Entity {name: "Transformer", type: "Model"})

是图的核心，是图的节点，可以重复利用

**颜色 “beige” 也值得是 Entity**（因为 QA 会问），不要只抽“名词”，要抽“可被问到的属性”

👉 就像你看了很多书后，脑子里形成的概念。



**Image**

> 图片本身

从属于 Chunk

👉 就像书里的插图。





🧠 推理路径（给 LLM 用）

```
Entity → RELATED_TO → Entity → RELATED_TO → Entity
```

📄 证据回溯路径（给 RAG 用）

```
Entity → MENTIONED_IN → Chunk → HAS_IMAGE → Image
```

✅ 正确形态

```
Entity —— Entity —— Entity
   |        |
 Chunk    Chunk
   |
 Image
```





## 关系

**RELATED_TO**

> 有方向的动词
>
> (a)-[:RELATED_TO {confidence: 0.8}]->(b)

a RELATED_TO b

关系也可以有属性



## 匹配

**MATCH**

> 

```
MATCH (e:Entity {name:"Transformer"})
RETURN e

MATCH (e:Entity {name:$entity})-[:RELATED_TO*1..3]-(n)
RETURN n
```







1. 根据 chunk_id 找 entities
2. 根据 entity 找相关 entities（多跳）
3. 根据 entity 找来源 chunk
4. 执行 LLM 生成的 Cypher















