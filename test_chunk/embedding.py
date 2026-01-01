"""
Embedding模块
负责将文本转换为向量表示
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union


class EmbeddingModel:
    """文本向量化模型"""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        初始化embedding模型

        Args:
            model_name: 预训练模型名称
                推荐模型:
                - paraphrase-multilingual-MiniLM-L12-v2: 多语言支持，384维
                - all-MiniLM-L6-v2: 英文模型，384维
                - paraphrase-multilingual-mpnet-base-v2: 更大的多语言模型，768维
        """
        self.model_name = model_name
        # print(f"正在加载embedding模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        # print(f"模型加载完成！向量维度: {self.dimension}")

    def embed_single(self, text: str) -> np.ndarray:
        """
        将单个文本转换为向量

        Args:
            text: 输入文本

        Returns:
            numpy数组形式的向量
        """
        if not text or not text.strip():
            raise ValueError("输入文本不能为空")

        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def embed_batch(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        批量将文本转换为向量

        Args:
            texts: 文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度条

        Returns:
            numpy数组，shape为(n, dimension)
        """
        if not texts:
            raise ValueError("文本列表不能为空")

        # 过滤空文本
        valid_texts = [text for text in texts if text and text.strip()]

        if len(valid_texts) != len(texts):
            print(f"警告: 过滤了 {len(texts) - len(valid_texts)} 个空文本")

        if not valid_texts:
            raise ValueError("没有有效的文本可以处理")

        embeddings = self.model.encode(
            valid_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

        return embeddings

    def get_dimension(self) -> int:
        """获取向量维度"""
        return self.dimension

    def get_model_name(self) -> str:
        """获取模型名称"""
        return self.model_name


class EmbeddingCache:
    """向量缓存（可选功能，提高重复文本的处理速度）"""

    def __init__(self, model: EmbeddingModel):
        """
        Args:
            model: EmbeddingModel实例
        """
        self.model = model
        self.cache = {}

    def embed_single(self, text: str) -> np.ndarray:
        """
        带缓存的单文本向量化

        Args:
            text: 输入文本

        Returns:
            numpy数组形式的向量
        """
        if text in self.cache:
            return self.cache[text]

        embedding = self.model.embed_single(text)
        self.cache[text] = embedding
        return embedding

    def embed_batch(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        带缓存的批量文本向量化

        Args:
            texts: 文本列表
            **kwargs: 传递给embed_batch的其他参数

        Returns:
            numpy数组
        """
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []

        # 检查缓存
        for i, text in enumerate(texts):
            if text in self.cache:
                embeddings.append(self.cache[text])
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
                embeddings.append(None)  # 占位

        # 处理未缓存的文本
        if texts_to_embed:
            new_embeddings = self.model.embed_batch(texts_to_embed, **kwargs)

            # 更新缓存和结果
            for idx, text, emb in zip(indices_to_embed, texts_to_embed, new_embeddings):
                self.cache[text] = emb
                embeddings[idx] = emb

        return np.array(embeddings)

    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        print("缓存已清空")

    def get_cache_size(self) -> int:
        """获取缓存大小"""
        return len(self.cache)


# ============= 使用示例 =============

if __name__ == "__main__":
    # 创建embedding模型
    embedder = EmbeddingModel()

    print("\n" + "=" * 50)
    print("测试1: 单个文本向量化")
    print("=" * 50)

    text = "人工智能正在改变世界"
    vector = embedder.embed_single(text)
    print(f"文本: {text}")
    print(f"向量维度: {vector.shape}")
    print(f"向量前10个值: {vector[:10]}")

    print("\n" + "=" * 50)
    print("测试2: 批量文本向量化")
    print("=" * 50)

    texts = [
        "机器学习是AI的核心技术",
        "深度学习在图像识别中表现出色",
        "自然语言处理让机器理解人类语言",
        "今天天气很好"
    ]

    vectors = embedder.embed_batch(texts, show_progress=False)
    print(f"处理了 {len(texts)} 个文本")
    print(f"结果shape: {vectors.shape}")

    for i, (text, vec) in enumerate(zip(texts, vectors)):
        print(f"\n文本{i + 1}: {text}")
        print(f"向量范数: {np.linalg.norm(vec):.4f}")

    print("\n" + "=" * 50)
    print("测试3: 带缓存的向量化")
    print("=" * 50)

    cached_embedder = EmbeddingCache(embedder)

    # 第一次处理
    test_texts = ["测试文本1", "测试文本2", "测试文本1"]  # 注意有重复
    result1 = cached_embedder.embed_batch(test_texts, show_progress=False)
    print(f"首次处理完成，缓存大小: {cached_embedder.get_cache_size()}")

    # 第二次处理（应该使用缓存）
    result2 = cached_embedder.embed_batch(test_texts, show_progress=False)
    print(f"再次处理完成，缓存大小: {cached_embedder.get_cache_size()}")
    print(f"两次结果是否相同: {np.allclose(result1, result2)}")