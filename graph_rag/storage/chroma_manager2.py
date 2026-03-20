"""
Chroma 向量数据库管理模块
负责实体和文本块的向量存储、检索
"""

import logging
from typing import List, Dict, Optional, Union
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingProvider:
    """Embedding 接口（方便以后替换）"""
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class BGEEmbedder(EmbeddingProvider):
    """BGE Embedding 模型封装"""

    def __init__(self, model_name: str = "BAAI/bge-base-zh-v1.5", device: str = "cpu"):
        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"✓ BGE模型加载成功: {model_name}, 维度: {self.dimension}")
        except Exception as e:
            raise RuntimeError(f"BGE模型加载失败: {str(e)}")

    def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """生成文本向量"""
        return self.model.encode(texts, normalize_embeddings=True).tolist()



class ChromaManager:
    def __init__(
        self,
        db_path: str = "../chroma_db",
        collection_name: str = "default",
        embedder: Optional[EmbeddingProvider] = None,
    ):
        """
        :param db_path: Chroma 持久化目录（可认为是数据库名）
        :param collection_name: collection 名
        :param embedder: embedder 实现（可替换）
        """
        self.embedder = embedder or BGEEmbedder()

        self.client = chromadb.Client(
            Settings(
                persist_directory=db_path,
                anonymized_telemetry=False,
            )
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_function,
        )

    def _embedding_function(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.embed(texts)

    def add(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
    ):
        """
        添加文档
        """
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    # ================== 删 ==================
    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict] = None,
    ):
        """
        删除文档
        """
        self.collection.delete(
            ids=ids,
            where=where,
        )

    # ================== 查 ==================
    def search(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict] = None,
    ):
        """
        相似度查询
        """
        query_embedding = self.embedder.embed(query_text)

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
        )

    # ================== 其他 ==================
    def count(self) -> int:
        return self.collection.count()

    def persist(self):
        self.client.persist()

# 使用示例
if __name__ == "__main__":
    # 初始化
    manager = ChromaManager(persist_directory="../chroma_db")

    # 添加实体
    entities = [
        {"name": "阿里巴巴", "type": "组织", "description": "中国最大的电子商务公司"},
        {"name": "马云", "type": "人物", "description": "阿里巴巴创始人"},
        {"name": "杭州", "type": "地点", "description": "浙江省省会"}
    ]

    manager.add(entities)
