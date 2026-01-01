"""
RAG基础Pipeline - 灵活的功能点封装
"""
from sympy import limit

from reader import FileReader
from chunk import get_splitter
from embedding import EmbeddingModel, EmbeddingCache
from storage import VectorDatabase
from similarity import SimilarityCalculator
import numpy as np
from typing import List, Dict, Optional, Union
import os


class RAGPipeline:
    """RAG基础功能Pipeline"""

    def __init__(
            self,
            embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
            db_path: str = "./vector_db",
            use_cache: bool = True
    ):
        """
        初始化RAG Pipeline

        Args:
            embedding_model_name: embedding模型名称
            db_path: 数据库存储路径
            use_cache: 是否使用缓存
        """
        self.reader = FileReader()
        self.embedder = EmbeddingModel(model_name=embedding_model_name)
        self.db = VectorDatabase(persist_directory=db_path)

        if use_cache:
            self.embedder = EmbeddingCache(self.embedder)

    # ============= 1. 读取文件功能 =============

    def read_file(self, file_path: str) -> str:
        """
        读取单个文件

        Args:
            file_path: 文件路径

        Returns:
            文件内容文本
        """
        return self.reader.read_file(file_path)

    def read_files(
            self,
            file_paths: List[str],
            return_dict: bool = True
    ) -> Union[Dict[str, str], List[str]]:
        """
        读取多个文件

        Args:
            file_paths: 文件路径列表
            return_dict: True返回{文件名:内容}，False返回内容列表

        Returns:
            字典或列表格式的文件内容
        """
        return self.reader.read_files(file_paths, return_dict=return_dict)

    def read_directory(
            self,
            dir_path: str,
            extensions: List[str] = None,
            recursive: bool = False,
            return_dict: bool = True
    ) -> Union[Dict[str, str], str]:
        """
        读取目录下的文件

        Args:
            dir_path: 目录路径
            extensions: 文件扩展名过滤，如['.txt', '.pdf']
            recursive: 是否递归读取子目录
            return_dict: True返回{文件名:内容}，False返回拼接字符串

        Returns:
            文件内容
        """
        return self.reader.read_directory(
            dir_path=dir_path,
            extensions=extensions,
            recursive=recursive,
            return_dict=return_dict
        )

    # ============= 2. 文本处理与存储功能 =============

    def embed_and_store(
            self,
            texts: Union[str, List[str]],
            collection_name: str,
            split_strategy: str = None,
            split_params: Dict = None,
            metadatas: Union[Dict, List[Dict]] = None,
            chunk_before_embed: bool = True
    ) -> Dict:
        """
        将文本embedding并存储到向量数据库

        Args:
            texts: 单个文本或文本列表
            collection_name: 集合名称
            split_strategy: 切分策略，None表示不切分
            split_params: 切分参数
            metadatas: 元数据，单个dict或list of dict
            chunk_before_embed: 是否在embedding前先切分

        Returns:
            存储结果统计
        """
        # 统一处理为列表
        if isinstance(texts, str):
            texts = [texts]

        # 文本切分（如果需要）
        if chunk_before_embed and split_strategy:
            if split_params is None:
                split_params = {}

            splitter = get_splitter(split_strategy, **split_params)

            all_chunks = []
            all_metadatas = []

            for idx, text in enumerate(texts):
                chunks = splitter.split(text)
                all_chunks.extend(chunks)

                # 处理元数据
                for chunk_idx, chunk in enumerate(chunks):
                    meta = {
                        'text_id': idx,
                        'chunk_id': chunk_idx,
                        'chunk_length': len(chunk),
                        'strategy': split_strategy
                    }

                    # 合并用户提供的元数据
                    if metadatas:
                        if isinstance(metadatas, dict):
                            meta.update(metadatas)
                        elif isinstance(metadatas, list) and idx < len(metadatas):
                            meta.update(metadatas[idx])

                    all_metadatas.append(meta)

            texts_to_embed = all_chunks
            metadatas_to_store = all_metadatas
        else:
            texts_to_embed = texts
            metadatas_to_store = []

            # 处理元数据
            for idx in range(len(texts)):
                meta = {'text_id': idx}
                if metadatas:
                    if isinstance(metadatas, dict):
                        meta.update(metadatas)
                    elif isinstance(metadatas, list) and idx < len(metadatas):
                        meta.update(metadatas[idx])
                metadatas_to_store.append(meta)

        # Embedding
        vectors = self.embedder.embed_batch(texts_to_embed, show_progress=False)

        # 存储
        ids = self.db.add_vectors(
            collection_name=collection_name,
            vectors=vectors,
            texts=texts_to_embed,
            metadatas=metadatas_to_store
        )

        return {
            'collection_name': collection_name,
            'num_texts': len(texts_to_embed),
            'vector_dim': vectors.shape[1],
            'ids': ids,
            'strategy': split_strategy,
            'params': split_params
        }

    # ============= 3. 搜索功能 =============

    def search(
            self,
            query: Union[str, np.ndarray],
            collection_name: str,
            top_k: int = 5,
            similarity_threshold: float = None
    ) -> List[Dict]:
        """
        搜索相似文本

        Args:
            query: 查询文本或查询向量
            collection_name: 集合名称
            top_k: 返回结果数
            similarity_threshold: 相似度阈值过滤

        Returns:
            搜索结果列表，每个结果包含: text, similarity, distance, metadata
        """
        # 处理查询
        if isinstance(query, str):
            query_vector = self.embedder.embed_single(query)
        else:
            query_vector = query

        # 搜索
        results = self.db.search_similar(
            collection_name=collection_name,
            query_vector=query_vector,
            top_k=top_k
        )

        # 格式化结果
        formatted_results = []
        for doc, vec, dist, meta in zip(
                results['documents'],
                results['vectors'],
                results['distances'],
                results['metadatas']
        ):
            similarity = 1 - dist

            # 阈值过滤
            if similarity_threshold and similarity < similarity_threshold:
                continue

            formatted_results.append({
                'text': doc,
                'similarity': similarity,
                'distance': dist,
                'metadata': meta,
                'vector': np.array(vec)
            })

        return formatted_results

    # ============= 4. 结果输出功能 =============

    def print_results(
            self,
            results: List[Dict],
            query: str = None,
            show_metadata: bool = False,
            show_vector: bool = False,
            max_text_length: int = None
    ):
        """
        打印搜索结果

        Args:
            results: 搜索结果列表
            query: 查询文本（可选）
            show_metadata: 是否显示元数据
            show_vector: 是否显示向量
            max_text_length: 文本显示最大长度，None为不限制
        """
        if query:
            print(f"\n查询: {query}")

        print(f"\n找到 {len(results)} 个结果")
        print("=" * 70)

        for i, result in enumerate(results, 1):
            print(f"\n排名 {i} | 相似度: {result['similarity']:.4f}")

            # 文本内容
            text = result['text']
            if max_text_length and len(text) > max_text_length:
                text = text[:max_text_length] + "..."
            print(f"文本: {text}")

            # 元数据
            if show_metadata and result['metadata']:
                print(f"元数据: {result['metadata']}")

            # 向量
            if show_vector:
                print(f"向量: {result['vector'][:5]}... (dim={len(result['vector'])})")

            print("-" * 70)

    def export_results(
            self,
            results: List[Dict],
            format: str = 'dict',
            include_vector: bool = False
    ) -> Union[List[Dict], str]:
        """
        导出搜索结果

        Args:
            results: 搜索结果列表
            format: 导出格式 'dict', 'json', 'text'
            include_vector: 是否包含向量

        Returns:
            导出的结果
        """
        if format == 'dict':
            if not include_vector:
                return [{k: v for k, v in r.items() if k != 'vector'} for r in results]
            return results

        elif format == 'json':
            import json
            export_data = []
            for r in results:
                data = {k: v for k, v in r.items() if k != 'vector'}
                if include_vector:
                    data['vector'] = r['vector'].tolist()
                export_data.append(data)
            return json.dumps(export_data, ensure_ascii=False, indent=2)

        elif format == 'text':
            lines = []
            for i, r in enumerate(results, 1):
                lines.append(f"[{i}] 相似度: {r['similarity']:.4f}")
                lines.append(f"    {r['text']}")
                if r['metadata']:
                    lines.append(f"    元数据: {r['metadata']}")
                lines.append("")
            return "\n".join(lines)

        else:
            raise ValueError(f"不支持的格式: {format}")

    # ============= 辅助功能 =============

    def get_collection_info(self, collection_name: str = None) -> Dict:
        """
        获取集合信息

        Args:
            collection_name: 集合名称，None则返回所有集合

        Returns:
            集合信息
        """
        if collection_name:
            count = self.db.get_collection_count(collection_name)
            return {
                'name': collection_name,
                'count': count
            }
        else:
            collections = self.db.list_collections()
            return {
                'collections': collections,
                'counts': {col: self.db.get_collection_count(col) for col in collections}
            }

    def delete_collection(self, collection_name: str):
        """删除集合"""
        self.db.delete_collection(collection_name)


# ============= 使用示例 =============

def example_basic_usage():
    """基础使用示例"""
    pipeline = RAGPipeline()

    # 1. 读取文件
    text = pipeline.read_file("document.txt")

    # 2. 处理并存储
    result = pipeline.embed_and_store(
        texts=text,
        collection_name="my_docs",
        split_strategy='sentence',
        metadatas={'source': 'document.txt'}
    )
    print(f"存储了 {result['num_texts']} 个文本块")

    # 3. 搜索
    results = pipeline.search(
        query="如何使用这个系统",
        collection_name="my_docs",
        top_k=5
    )

    # 4. 输出结果
    pipeline.print_results(results, query="如何使用这个系统")


def example_batch_files():
    """批量文件处理示例"""
    pipeline = RAGPipeline()

    # 读取目录下所有文档
    docs = pipeline.read_directory(
        dir_path="./documents",
        extensions=['.txt', '.pdf', '.docx'],
        recursive=True,
        return_dict=True
    )

    # 批量处理
    for filename, content in docs.items():
        pipeline.embed_and_store(
            texts=content,
            collection_name="all_docs",
            split_strategy='n_sentence',
            split_params={'n': 3},
            metadatas={'filename': filename}
        )

    print(f"处理完成，集合信息: {pipeline.get_collection_info('all_docs')}")


def example_strategy_comparison():
    """策略对比测试示例"""
    pipeline = RAGPipeline()

    # 读取文档
    text = pipeline.read_file("test.txt")

    # 测试不同策略
    strategies = [
        ('sentence', {}),
        ('n_sentence_2', {'n': 2}),
        ('n_sentence_3', {'n': 3}),
        ('fixed_200', {'chunk_size': 200, 'overlap': 50})
    ]

    query = "测试查询"

    print("\n策略对比测试")
    print("=" * 70)

    for name, params in strategies:
        # 确定策略类型
        strategy = params.pop('strategy', name.split('_')[0] if '_' in name else name)

        # 存储
        collection_name = f"test_{name}"
        result = pipeline.embed_and_store(
            texts=text,
            collection_name=collection_name,
            split_strategy=strategy,
            split_params=params
        )

        # 搜索
        results = pipeline.search(query, collection_name, top_k=3)

        # 输出
        print(f"\n[{name}] 块数: {result['num_texts']}, Top-1相似度: {results[0]['similarity']:.4f}")
        print(f"  {results[0]['text'][:60]}...")


def main():
    """主函数"""
    pipeline = RAGPipeline(
        embedding_model_name="paraphrase-multilingual-MiniLM-L12-v2",
        db_path="./vector_db",
        use_cache=True
    )

    # 1. 读取文件
    print("步骤1: 读取文件")
    text = pipeline.read_file("files/蟹堡王餐厅·员工财富密码手册（工作规范终极版）.docx")
    stor_version = 3

    # 2. 处理并存储 - 测试不同策略
    strategies = [
        ('1j', 'n_sentence', {'n': 1}),
        ('2j', 'n_sentence', {'n': 2}),
        ('4j', 'n_sentence', {'n': 4}),
        ('6j', 'n_sentence', {'n': 6}),
        ('fixed', 'fixed', {'chunk_size': 50, 'overlap': 10}),
        ('paragraph', 'paragraph', {'min_length': 10}),
        ('semantic', 'semantic', {'max_length': 100}),
    ]

    print("步骤2: 存储文档（不同切分策略）")
    for name, strategy, params in strategies:
        collection_name = f"crab_{name}_{stor_version}"
        result = pipeline.embed_and_store(
            texts=text,
            collection_name=collection_name,
            split_strategy=strategy,
            split_params=params,
            metadatas={'source': '蟹堡王手册'}
        )
        print(f"  [{name}] {result['num_texts']} 块")

    # 3. 搜索对比
    query = "如何接待一只沙丁鱼"

    print("\n步骤3: 搜索对比")

    for name, strategy, _ in strategies:
        collection_name = f"crab_{name}_{stor_version}"
        print(f"\n\n{collection_name}")
        results = pipeline.search(query, collection_name, top_k=15, similarity_threshold=0.5)
        pipeline.print_results(results, query, show_metadata=False)


if __name__ == "__main__":
    main()
