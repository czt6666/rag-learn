from main import RAGPipeline

if __name__ == '__main__':
    dir_path = "./files"
    stor_version = 1
    collection_name = f"crab_test1_{stor_version}"

    pipeline = RAGPipeline(
        embedding_model_name="paraphrase-multilingual-MiniLM-L12-v2",
        db_path="./vector_db",
        use_cache=True
    )
    # files = pipeline.read_directory(dir_path)
    # for file_name, text in files.items():
    #     result = pipeline.embed_and_store(
    #         texts=text,
    #         collection_name=collection_name,
    #         split_strategy='fixed',
    #         split_params={'chunk_size': 100, 'overlap': 20},
    #         metadatas={'source': file_name}
    #     )
    #     print(f"  [{file_name}] {result['num_texts']} 块")

    query = "我感到愤怒怎么办"

    print(query)

    results = pipeline.search(query, collection_name, top_k=5, similarity_threshold=0.5)
    pipeline.print_results(results, query, show_metadata=False)