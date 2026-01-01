"""
向量存储模块
基于ChromaDB实现向量的持久化存储和相似度检索
"""

import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
import os
import uuid


class VectorDatabase:
    """向量数据库类"""

    def __init__(self, persist_directory: str = "./vector_db"):
        """
        初始化向量数据库

        Args:
            persist_directory: 数据库持久化目录路径
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        # 初始化ChromaDB持久化客户端
        self.client = chromadb.PersistentClient(path=persist_directory)
        # print(f"向量数据库初始化完成")
        # print(f"存储路径: {os.path.abspath(persist_directory)}")

    def create_collection(self, collection_name: str, overwrite: bool = False) -> chromadb.Collection:
        """
        创建新的集合（数据库表）

        Args:
            collection_name: 集合名称
            overwrite: 如果集合已存在，是否覆盖

        Returns:
            ChromaDB集合对象
        """
        if overwrite:
            try:
                self.client.delete_collection(name=collection_name)
                print(f"已删除旧集合: {collection_name}")
            except:
                pass

        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # 使用余弦距离
        )
        # print(f"集合 '{collection_name}' 已就绪")
        return collection

    def get_collection(self, collection_name: str) -> Optional[chromadb.Collection]:
        """
        获取已存在的集合

        Args:
            collection_name: 集合名称

        Returns:
            集合对象，如果不存在则返回None
        """
        try:
            return self.client.get_collection(name=collection_name)
        except:
            print(f"集合 '{collection_name}' 不存在")
            return None

    def add_vectors(
            self,
            collection_name: str,
            vectors: Union[List[List[float]], np.ndarray],
            texts: List[str],
            ids: Optional[List[str]] = None,
            metadatas: Optional[List[Dict]] = None
    ) -> List[str]:
        """
        向集合添加向量

        Args:
            collection_name: 集合名称
            vectors: 向量列表或numpy数组
            texts: 对应的原始文本列表
            ids: 文档ID列表（可选，不提供则自动生成UUID）
            metadatas: 元数据列表（可选）

        Returns:
            实际使用的ID列表
        """
        collection = self.create_collection(collection_name)

        # 转换向量格式
        if isinstance(vectors, np.ndarray):
            vectors = vectors.tolist()

        # 生成ID
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]

        # 确保ID唯一
        if len(ids) != len(set(ids)):
            print("警告: 检测到重复ID，将自动生成新ID")
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]

        # 添加到数据库
        collection.add(
            embeddings=vectors,
            documents=texts,
            ids=ids,
            metadatas=metadatas
        )

        # print(f"成功添加 {len(vectors)} 条记录到集合 '{collection_name}'")
        return ids

    def search_similar(
            self,
            collection_name: str,
            query_vector: Union[List[float], np.ndarray],
            top_k: int = 5,
            filter_metadata: Optional[Dict] = None
    ) -> Dict[str, List]:
        """
        相似度检索，返回最相似的前k个结果

        Args:
            collection_name: 集合名称
            query_vector: 查询向量
            top_k: 返回结果数量
            filter_metadata: 元数据过滤条件（可选）

        Returns:
            字典，包含:
                - ids: 文档ID列表
                - documents: 原始文本列表
                - vectors: 向量列表
                - distances: 距离列表（越小越相似）
                - metadatas: 元数据列表
        """
        collection = self.get_collection(collection_name)
        if collection is None:
            raise ValueError(f"集合 '{collection_name}' 不存在")

        # 转换向量格式
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        # 执行查询
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=filter_metadata,
            include=['documents', 'embeddings', 'distances', 'metadatas']
        )

        return {
            'ids': results['ids'][0] if results['ids'] else [],
            'documents': results['documents'][0] if results['documents'] else [],
            'vectors': results['embeddings'][0] if results['embeddings'] else [],
            'distances': results['distances'][0] if results['distances'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else []
        }

    def get_by_ids(self, collection_name: str, ids: List[str]) -> Dict[str, List]:
        """
        根据ID获取记录

        Args:
            collection_name: 集合名称
            ids: 文档ID列表

        Returns:
            字典，包含documents, vectors, metadatas
        """
        collection = self.get_collection(collection_name)
        if collection is None:
            raise ValueError(f"集合 '{collection_name}' 不存在")

        results = collection.get(
            ids=ids,
            include=['documents', 'embeddings', 'metadatas']
        )

        return {
            'ids': results['ids'],
            'documents': results['documents'],
            'vectors': results['embeddings'],
            'metadatas': results['metadatas']
        }

    def delete_by_ids(self, collection_name: str, ids: List[str]) -> None:
        """
        根据ID删除记录

        Args:
            collection_name: 集合名称
            ids: 要删除的文档ID列表
        """
        collection = self.get_collection(collection_name)
        if collection is None:
            raise ValueError(f"集合 '{collection_name}' 不存在")

        collection.delete(ids=ids)
        print(f"已从集合 '{collection_name}' 删除 {len(ids)} 条记录")

    def get_collection_count(self, collection_name: str) -> int:
        """获取集合中的记录数量"""
        collection = self.get_collection(collection_name)
        if collection is None:
            return 0
        return collection.count()

    def list_collections(self) -> List[str]:
        """列出所有集合名称"""
        collections = self.client.list_collections()
        return [col.name for col in collections]

    def delete_collection(self, collection_name: str) -> None:
        """删除整个集合"""
        try:
            self.client.delete_collection(name=collection_name)
            print(f"集合 '{collection_name}' 已删除")
        except Exception as e:
            print(f"删除集合失败: {e}")

    def get_all_documents(self, collection_name: str) -> Dict[str, List]:
        """
        获取集合中的所有文档

        Args:
            collection_name: 集合名称

        Returns:
            包含所有文档的字典
        """
        collection = self.get_collection(collection_name)
        if collection is None:
            raise ValueError(f"集合 '{collection_name}' 不存在")

        # ChromaDB的get方法不带参数时返回所有文档
        results = collection.get(include=['documents', 'embeddings', 'metadatas'])

        return {
            'ids': results['ids'],
            'documents': results['documents'],
            'vectors': results['embeddings'],
            'metadatas': results['metadatas']
        }


# ============= 使用示例 =============

if __name__ == "__main__":
    # 创建数据库
    db = VectorDatabase(persist_directory="./test_vector_db")

    print("\n" + "=" * 50)
    print("测试1: 添加向量")
    print("=" * 50)

    # 模拟一些向量和文本
    test_vectors = np.random.rand(5, 384).tolist()
    test_texts = [
        "人工智能正在改变世界",
        "机器学习是AI的核心",
        "深度学习效果显著",
        "今天天气很好",
        "明天会下雨吗"
    ]
    test_metadatas = [
        {"category": "AI", "source": "doc1"},
        {"category": "AI", "source": "doc2"},
        {"category": "AI", "source": "doc3"},
        {"category": "weather", "source": "doc4"},
        {"category": "weather", "source": "doc5"}
    ]

    ids = db.add_vectors(
        collection_name="test_collection",
        vectors=test_vectors,
        texts=test_texts,
        metadatas=test_metadatas
    )

    print(f"集合记录数: {db.get_collection_count('test_collection')}")

    print("\n" + "=" * 50)
    print("测试2: 相似度检索")
    print("=" * 50)

    query_vector = np.random.rand(384)
    results = db.search_similar(
        collection_name="test_collection",
        query_vector=query_vector,
        top_k=3
    )

    print(f"查询返回 {len(results['documents'])} 个结果:")
    for i, (doc, dist, meta) in enumerate(zip(
            results['documents'],
            results['distances'],
            results['metadatas']
    ), 1):
        print(f"{i}. 文本: {doc}")
        print(f"   距离: {dist:.4f}")
        print(f"   元数据: {meta}\n")

    print("\n" + "=" * 50)
    print("测试3: 按ID获取")
    print("=" * 50)

    selected_ids = ids[:2]
    retrieved = db.get_by_ids("test_collection", selected_ids)
    print(f"获取到 {len(retrieved['documents'])} 条记录:")
    for doc, meta in zip(retrieved['documents'], retrieved['metadatas']):
        print(f"- {doc} (类别: {meta['category']})")

    print("\n" + "=" * 50)
    print("测试4: 列出所有集合")
    print("=" * 50)

    collections = db.list_collections()
    print(f"当前集合: {collections}")