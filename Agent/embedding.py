from langchain.embeddings import init_embeddings

embeddings = init_embeddings("openai:text-embedding-3-small")

vec = embeddings.embed_query("什么是向量检索")
print(len(vec), vec[:5])
