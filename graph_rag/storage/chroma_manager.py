"""
Chroma 向量数据库管理模块
负责实体和文本块的向量存储、检索
"""

import logging
from typing import List, Dict, Optional, Union
import chromadb
from chromadb.config import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BGEEmbedder:
    """BGE Embedding 模型封装"""

    def __init__(self, model_name: str = "BAAI/bge-base-zh-v1.5", device: str = "cpu"):
        """
        初始化 BGE 模型

        Args:
            model_name: 模型名称
            device: 设备 "cpu" 或 "cuda"
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=device)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"✓ BGE模型加载成功: {model_name}, 维度: {self.dimension}")
        except ImportError:
            raise ImportError("需要安装: pip install sentence-transformers")
        except Exception as e:
            raise RuntimeError(f"BGE模型加载失败: {str(e)}")

    def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        生成文本向量

        Args:
            texts: 单个文本或文本列表

        Returns:
            向量或向量列表
        """
        if isinstance(texts, str):
            return self.model.encode(texts, normalize_embeddings=True).tolist()
        else:
            return self.model.encode(texts, normalize_embeddings=True).tolist()


class ChromaManager:
    """Chroma 向量数据库管理器"""

    def __init__(self,
                 persist_directory: str = "../chroma_db",
                 embedder: Optional[BGEEmbedder] = None,
                 entity_collection_name: str = "entities",
                 chunk_collection_name: str = "chunks"):
        """
        初始化 Chroma 数据库

        Args:
            persist_directory: 持久化存储目录
            embedder: 自定义 Embedding 模型，None则使用默认BGE
            entity_collection_name: 实体集合名称，默认为 "entities"
            chunk_collection_name: 文本块集合名称，默认为 "chunks"
        """
        self.persist_directory = persist_directory
        self.entity_collection_name = entity_collection_name
        self.chunk_collection_name = chunk_collection_name

        # 初始化 Chroma 客户端
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # 初始化 Embedding 模型
        if embedder is None:
            self.embedder = BGEEmbedder()
        else:
            self.embedder = embedder

        # 创建集合
        self.entity_collection = self._get_or_create_collection(entity_collection_name)
        self.chunk_collection = self._get_or_create_collection(chunk_collection_name)

        logger.info(f"✓ Chroma 数据库初始化成功: {persist_directory}")
        logger.info(f"  实体集合: {entity_collection_name}, 文本块集合: {chunk_collection_name}")

    def _get_or_create_collection(self, name: str):
        """获取或创建集合"""
        try:
            collection = self.client.get_collection(name)
            logger.info(f"✓ 加载已存在的集合: {name}")
        except:
            collection = self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
            )
            logger.info(f"✓ 创建新集合: {name}")
        return collection

    def add_entity(self,
                   name: str,
                   entity_type: str,
                   description: str = "",
                   metadata: Optional[Dict] = None) -> bool:
        """
        添加实体向量

        Args:
            name: 实体名称
            entity_type: 实体类型
            description: 实体描述
            metadata: 额外元数据

        Returns:
            是否成功
        """
        try:
            # 生成向量（使用名称+描述）
            text = f"{name} {description}".strip()
            embedding = self.embedder.embed(text)

            # 准备元数据
            meta = {
                "name": name,
                "type": entity_type,
                "description": description
            }
            if metadata:
                meta.update(metadata)

            # 存储
            self.entity_collection.add(
                ids=[f"entity_{name}"],
                embeddings=[embedding],
                documents=[text],
                metadatas=[meta]
            )
            return True

        except Exception as e:
            logger.error(f"添加实体向量失败 {name}: {str(e)}")
            return False

    def add_entities(self, entities: List[Dict]) -> int:
        """
        批量添加实体向量

        Args:
            entities: 实体列表 [{"name": "", "type": "", "description": ""}]

        Returns:
            成功添加的数量
        """
        if not entities:
            return 0

        try:
            ids = []
            embeddings = []
            documents = []
            metadatas = []

            for entity in entities:
                name = entity.get('name', '')
                entity_type = entity.get('type', '其他')
                description = entity.get('description', '')

                text = f"{name} {description}".strip()
                embedding = self.embedder.embed(text)

                ids.append(f"entity_{name}")
                embeddings.append(embedding)
                documents.append(text)
                metadatas.append({
                    "name": name,
                    "type": entity_type,
                    "description": description
                })

            self.entity_collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

            logger.info(f"✓ 批量添加实体向量: {len(entities)}")
            return len(entities)

        except Exception as e:
            logger.error(f"批量添加实体向量失败: {str(e)}")
            return 0

    def add_chunk(self,
                  chunk_id: str,
                  text: str,
                  source_file: str = "",
                  entity_names: Optional[List[str]] = None,
                  metadata: Optional[Dict] = None) -> bool:
        """
        添加文本块向量

        Args:
            chunk_id: 文本块ID
            text: 文本内容
            source_file: 来源文件
            entity_names: 包含的实体名称列表
            metadata: 额外元数据

        Returns:
            是否成功
        """
        try:
            # 生成向量
            embedding = self.embedder.embed(text)

            # 准备元数据
            meta = {
                "chunk_id": chunk_id,
                "source_file": source_file,
                "entity_names": ",".join(map(str, entity_names))  or [],
                "length": len(text)
            }
            if metadata:
                meta.update(metadata)

            # 存储
            self.chunk_collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[meta]
            )
            return True

        except Exception as e:
            logger.error(f"添加文本块向量失败 {chunk_id}: {str(e)}")
            return False

    def add_chunks(self, chunks: List[Dict]) -> int:
        """
        批量添加文本块向量

        Args:
            chunks: 文本块列表 [{"chunk_id": "", "text": "", "source_file": "", "entity_names": []}]

        Returns:
            成功添加的数量
        """
        if not chunks:
            return 0

        try:
            ids = []
            embeddings = []
            documents = []
            metadatas = []

            for chunk in chunks:
                chunk_id = chunk.get('chunk_id', '')
                text = chunk.get('text', '')

                embedding = self.embedder.embed(text)

                ids.append(chunk_id)
                embeddings.append(embedding)
                documents.append(text)
                metadatas.append({
                    "chunk_id": chunk_id,
                    "source_file": chunk.get('source_file', ''),
                    "entity_names": chunk.get('entity_names', []),
                    "length": len(text)
                })

            self.chunk_collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

            logger.info(f"✓ 批量添加文本块向量: {len(chunks)}")
            return len(chunks)

        except Exception as e:
            logger.error(f"批量添加文本块向量失败: {str(e)}")
            return 0

    def search_entities(self,
                        query: str,
                        top_k: int = 5,
                        entity_type: Optional[str] = None) -> List[Dict]:
        """
        搜索相似实体

        Args:
            query: 查询文本
            top_k: 返回数量
            entity_type: 实体类型过滤

        Returns:
            相似实体列表
        """
        try:
            # 生成查询向量
            query_embedding = self.embedder.embed(query)

            # 构建过滤条件
            where = {"type": entity_type} if entity_type else None

            # 查询
            results = self.entity_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where
            )

            # 格式化结果
            entities = []
            if results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    entities.append({
                        "name": results['metadatas'][0][i]['name'],
                        "type": results['metadatas'][0][i]['type'],
                        "description": results['metadatas'][0][i]['description'],
                        "distance": results['distances'][0][i] if 'distances' in results else None
                    })

            logger.info(f"✓ 实体检索: 找到 {len(entities)} 个相关实体")
            return entities

        except Exception as e:
            logger.error(f"实体检索失败: {str(e)}")
            return []

    def search_chunks(self,
                      query: str,
                      top_k: int = 5,
                      source_file: Optional[str] = None) -> List[Dict]:
        """
        搜索相似文本块

        Args:
            query: 查询文本
            top_k: 返回数量
            source_file: 来源文件过滤

        Returns:
            相似文本块列表
        """
        try:
            # 生成查询向量
            query_embedding = self.embedder.embed(query)

            # 构建过滤条件
            where = {"source_file": source_file} if source_file else None

            # 查询
            results = self.chunk_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where
            )

            # 格式化结果
            chunks = []
            if results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    chunks.append({
                        "chunk_id": results['metadatas'][0][i]['chunk_id'],
                        "text": results['documents'][0][i],
                        "source_file": results['metadatas'][0][i]['source_file'],
                        "entity_names": results['metadatas'][0][i].get('entity_names', []),
                        "distance": results['distances'][0][i] if 'distances' in results else None
                    })

            logger.info(f"✓ 文本块检索: 找到 {len(chunks)} 个相关块")
            return chunks

        except Exception as e:
            logger.error(f"文本块检索失败: {str(e)}")
            return []

    def hybrid_search(self, query: str, top_k: int = 5) -> Dict:
        """
        混合检索（同时检索实体和文本块）

        Args:
            query: 查询文本
            top_k: 每个类型返回数量

        Returns:
            {
                "entities": [...],
                "chunks": [...]
            }
        """
        entities = self.search_entities(query, top_k=top_k)
        chunks = self.search_chunks(query, top_k=top_k)

        return {
            "entities": entities,
            "chunks": chunks
        }

    def get_entity_chunks(self, entity_name: str, top_k: int = 5) -> List[Dict]:
        """
        获取包含特定实体的文本块

        Args:
            entity_name: 实体名称
            top_k: 返回数量

        Returns:
            文本块列表
        """
        try:
            results = self.chunk_collection.get(
                where={"entity_names": {"$contains": entity_name}},
                limit=top_k
            )

            chunks = []
            if results['ids']:
                for i in range(len(results['ids'])):
                    chunks.append({
                        "chunk_id": results['metadatas'][i]['chunk_id'],
                        "text": results['documents'][i],
                        "source_file": results['metadatas'][i]['source_file'],
                        "entity_names": results['metadatas'][i].get('entity_names', [])
                    })

            return chunks

        except Exception as e:
            logger.error(f"获取实体文本块失败: {str(e)}")
            return []

    def delete_entity(self, name: str) -> bool:
        """删除实体"""
        try:
            self.entity_collection.delete(ids=[f"entity_{name}"])
            return True
        except Exception as e:
            logger.error(f"删除实体失败 {name}: {str(e)}")
            return False

    def delete_chunk(self, chunk_id: str) -> bool:
        """删除文本块"""
        try:
            self.chunk_collection.delete(ids=[chunk_id])
            return True
        except Exception as e:
            logger.error(f"删除文本块失败 {chunk_id}: {str(e)}")
            return False

    def clear_all(self):
        """清空所有数据"""
        try:
            self.client.delete_collection(self.entity_collection_name)
            self.client.delete_collection(self.chunk_collection_name)
            self.entity_collection = self._get_or_create_collection(self.entity_collection_name)
            self.chunk_collection = self._get_or_create_collection(self.chunk_collection_name)
            logger.info("✓ 已清空所有向量数据")
        except Exception as e:
            logger.error(f"清空数据失败: {str(e)}")

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        try:
            entity_count = self.entity_collection.count()
            chunk_count = self.chunk_collection.count()

            return {
                "entity_count": entity_count,
                "chunk_count": chunk_count,
                "embedding_dimension": self.embedder.dimension
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            return {}


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
    manager.add_entities(entities)

    # 添加文本块
    chunks = [
        {
            "chunk_id": "chunk_001",
            "text": "马云于1999年在杭州创立了阿里巴巴集团",
            "source_file": "doc1.txt",
            "entity_names": ["马云", "杭州", "阿里巴巴"]
        }
    ]
    manager.add_chunks(chunks)

    # 搜索实体
    results = manager.search_entities("电商公司", top_k=3)
    print(f"\n实体检索结果:")
    for r in results:
        print(f"  {r['name']} ({r['type']}): {r['description']}")

    # 搜索文本块
    chunks = manager.search_chunks("阿里巴巴的创始人", top_k=3)
    print(f"\n文本块检索结果:")
    for c in chunks:
        print(f"  {c['chunk_id']}: {c['text'][:50]}...")

    # 混合检索
    hybrid = manager.hybrid_search("马云创业", top_k=3)
    print(f"\n混合检索:")
    print(f"  实体: {len(hybrid['entities'])} 个")
    print(f"  文本块: {len(hybrid['chunks'])} 个")

    # 统计信息
    stats = manager.get_statistics()
    print(f"\n统计信息: {stats}")