from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLm-L6-v2')

sentences = [
    "The weather is lovely today.",
    "It's so sunny outside.",
    "He drove to the stadium."
]

embeddings = model.encode(sentences)
print(embeddings.shape)

# for i in embeddings:
#     print(i)
#     print("===========")

word_embeddings = model.encode(["love","sun","I"])
similarities = model.similarity(word_embeddings, embeddings)
print(similarities)