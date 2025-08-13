import datetime
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("/Users/v_changzhitao/Desktop/project/kg/Models/BAAI/bge-large-zh-v1___5")
# client = chromadb.Client() # Memory
client = chromadb.PersistentClient("./chroma_v3") # File
collection = client.get_or_create_collection(
    name="czt666",
    metadata={
        "description": "随机文本文件的向量数据库",
        "creat_time": str(datetime.datetime.now()),
        "hnsw:space": "cosine" # 余弦相似度
    }
)

def txt_2db():
    # 1. 加载所有的文件
    path_list = list(Path("../my_knowledge").glob("*.txt"))
    text_list = [] #文本内容

    for path in path_list:#A.复杂度
        text = path.read_text(encoding="utf-8") #读取内容
        text_list.append(text)

    # 2. 进行向量嵌入
    embeddings = model.encode(text_list) #列表知识库

    # 3. 存入数据库
    collection.add(
        embeddings=embeddings.tolist(),  # 向量
        documents= text_list,  # 文本
        metadatas=[{"id":i} for i in text_list],  # 元数据
        ids=[f"doc_{i}" for i,_ in enumerate(text_list)] #ID
    )

    print(f'数据库中的数据量：{collection.count()}')


if __name__ == '__main__':
    txt_2db()
    query = ["工作内容是什么"]
    query_embeddings = model.encode(query)
    data = collection.query(query_embeddings.tolist(), n_results=2)
    print(data)