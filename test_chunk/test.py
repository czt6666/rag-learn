from embedding import EmbeddingModel, EmbeddingCache
from similarity import SimilarityCalculator

embedder = EmbeddingModel()
embedder = EmbeddingCache(embedder)
similarities = SimilarityCalculator()
vecs = embedder.embed_batch(["苹果是一种水果", "1. 苹果是一种水果; 2. 苹果是一个科技公司"])
print(similarities.cosine_similarity(vecs[0], vecs[1])) # 余弦相似度
print(similarities.euclidean_distance(vecs[0], vecs[1])) # 欧氏距离
print(similarities.manhattan_distance(vecs[0], vecs[1])) # 曼哈顿
print(similarities.dot_product(vecs[0], vecs[1])) # 点积
