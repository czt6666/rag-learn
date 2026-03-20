"""模块 2：多模态嵌入与向量索引"""
import logging
import torch
import numpy as np
from typing import List, Dict, Optional
from PIL import Image
from tqdm import tqdm
import chromadb
from config import get_settings
from module_1_parser import MultimodalChunk

logger = logging.getLogger(__name__)

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger.warning("CLIP not available, using dummy embeddings")


class MultimodalEmbedder:
    """多模态嵌入生成器"""
    
    def __init__(self, model_name: str = "ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {self.device}")
        
        if CLIP_AVAILABLE:
            logger.info(f"加载 CLIP 模型: {model_name}")
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            self.model.eval()
        else:
            self.model = None
            logger.warning("使用虚拟嵌入（用于测试）")
    
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """编码文本"""
        if not CLIP_AVAILABLE or self.model is None:
            # 虚拟嵌入用于测试
            return np.random.randn(len(texts), 512).astype(np.float32)
        
        with torch.no_grad():
            tokens = clip.tokenize(texts, truncate=True).to(self.device)
            embeddings = self.model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy()
    
    def encode_chunk(self, chunk: MultimodalChunk) -> List[float]:
        """为多模态 chunk 生成嵌入"""
        embedding = self.encode_text([chunk.joined_text])
        return embedding[0].tolist()


class VectorStore:
    """ChromaDB 向量存储"""
    
    def __init__(self, persist_directory: Optional[str] = None):
        self.settings = get_settings()
        if persist_directory is None:
            persist_directory = self.settings.chromadb_path
        
        logger.info(f"初始化 ChromaDB: {persist_directory}")
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="multimodal_chunks",
            metadata={"description": "多模态文档块"}
        )
        logger.info(f"集合文档数: {self.collection.count()}")
    
    def add_chunks(self, chunks: List[MultimodalChunk], embeddings: List[List[float]]):
        """添加 chunks"""
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.joined_text for chunk in chunks]
        metadatas = [
            {
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "image_ids": ",".join(chunk.image_ids),
                "image_captions": ",".join(chunk.image_captions),
                "has_images": str(len(chunk.image_ids) > 0)
            }
            for chunk in chunks
        ]
        
        batch_size = 100
        for i in tqdm(range(0, len(chunks), batch_size), desc="添加到 ChromaDB"):
            end_idx = min(i + batch_size, len(chunks))
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
        
        logger.info(f"添加完成，总文档数: {self.collection.count()}")
    
    def query(self, query_embedding: List[float], n_results: int = 5) -> Dict:
        """查询相似文档"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results
    
    def reset(self):
        """清空集合"""
        self.client.delete_collection("multimodal_chunks")
        self.collection = self.client.get_or_create_collection(
            name="multimodal_chunks"
        )


class VectorIndexBuilder:
    """向量索引构建器"""
    
    def __init__(self, model_name: Optional[str] = None):
        self.settings = get_settings()
        if model_name is None:
            model_name = self.settings.clip_model_name
        
        self.embedder = MultimodalEmbedder(model_name)
        self.vector_store = VectorStore()
    
    def build_index(self, chunks: List[MultimodalChunk], reset: bool = False):
        """构建向量索引"""
        if reset:
            self.vector_store.reset()
        
        logger.info(f"开始构建向量索引: {len(chunks)} 个 chunks")
        
        embeddings = []
        for chunk in tqdm(chunks, desc="生成嵌入"):
            embedding = self.embedder.encode_chunk(chunk)
            embeddings.append(embedding)
        
        self.vector_store.add_chunks(chunks, embeddings)
        logger.info("向量索引构建完成")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索相似文档"""
        query_embedding = self.embedder.encode_text([query])[0].tolist()
        results = self.vector_store.query(query_embedding, n_results=top_k)
        
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'chunk_id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return formatted_results
