import chromadb
from chromadb.config import Settings
import os
from typing import List, Optional, Any

from embedder import Embedder


class ChromaStore:
    """
    极简 ChromaDB 封装
    默认存储位置: ./chroma_db 
    """

    def __init__(self, collection_name: str, persist_directory: str = "./chroma_db"):
        # 1. 确保目录存在
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)

        # 2. 初始化持久化客户端 (数据会写到磁盘)
        self.client = chromadb.PersistentClient(path=persist_directory)

        # 3. 获取或创建集合 (类似数据库中的表)
        # 显式约定: 使用余弦相似度 (cosine)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_data(self, ids: List[str], embeddings: Any, metadatas: Optional[List[dict]] = None,
                 documents: Optional[List[str]] = None):
        """
        存入数据
        embeddings: 可以是 torch.Tensor 或 list
        """
        # 将 Tensor 转换为 list (Chroma 只接受 Python 原生列表)
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.cpu().detach().tolist()

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

    def search(self, query_embeddings: Any, n_results: int = 3):
        """
        向量检索
        """
        if hasattr(query_embeddings, "tolist"):
            query_embeddings = query_embeddings.cpu().detach().tolist()

        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results
        )

    def count(self):
        return self.collection.count()


# -----------------------------
# 测试代码 (main)
# -----------------------------
if __name__ == "__main__":
    # 1. 初始化
    db = ChromaStore("test_1")
    embedder = Embedder()

    DOC_ID = "test_doc"

    # 2. 准备数据
    img_path = "./test_data/MRAMG-Bench/IMAGE/IMAGE/images/WEB/30000132.jpg"
    candidate_texts = [
        "The National Museum of the American Indian is geographically located in Washington, D.C. and specifically situated on the Mall.",
        "The National Museum of the American Indian officially opened in the year 2004.",
        "The National Museum of the American Indian embodies both cultural respect and architectural beauty through its design and institutional identity.",
        "The National Museum of the American Indian has architectural features including a curvilinear structure, a smooth beige-colored exterior, and a flowing design.",
        "The museum's name reflects terminology that American Indians are comfortable with, particularly the term American Indian."
    ]

    # 3. 编码 (利用 GPU)
    # 获取图片向量用于后续搜索
    # img_vector = embedder.encode_image(img_path)
    # 获取所有文本向量用于入库
    text_vectors = embedder.encode_text(candidate_texts)

    # for index, text in enumerate(candidate_texts):
    #     # 按照你的要求构建 ID: {doc_id}_{chunk_id}
    #     unique_id = f"{DOC_ID}_{index}"
    #
    #     db.add_data(
    #         ids=[unique_id],
    #         embeddings=text_vectors[index:index + 1],  # 取出对应行的向量 [1, 512]
    #         metadatas=[{
    #             "source": "manual_label",
    #             "category": "biology" if "butterfly" in text.lower() or "Papilio" in text else "other",
    #             "length": len(text)
    #         }],
    #         documents=[f"Category: Insect | Label: {text} | Description: A candidate tag for visual matching."]
    #     )

    print(f"✅ 成功入库，当前数据库条数: {db.count()}")

    query = "Are both the National Museum of the American Indian in Washington, D.C. and the Xanadu House in Kissimmee, Florida the same color?"
    query = "color"
    query_vectors = embedder.encode_text(query)
    # search 内部需要支持 where 参数，这里先演示标准的向量搜索
    results = db.search(query_vectors, n_results=3)

    # 6. 深度解析搜索结果
    print("\n" + "=" * 60)
    for i in range(len(results['ids'][0])):
        res_id = results['ids'][0][i]
        res_doc = results['documents'][0][i]
        res_meta = results['metadatas'][0][i]
        # Chroma 默认 cosine 空间下 distance = 1 - similarity
        score = 1 - results['distances'][0][i]

        print(f"排名 #{i + 1}")
        print(f" 🆔 ID: {res_id}")
        print(f" 🎯 相似度: {score:.4f}")
        print(f" 📝 文档内容: {res_doc}")
        print(f" 🏷️ 元数据: {res_meta}")
        print("-" * 30)
