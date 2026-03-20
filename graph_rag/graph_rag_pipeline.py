"""
Graph RAG 完整流程
整合所有模块，提供端到端的知识图谱构建和问答功能
"""

import logging
import time
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# 导入所有模块
from graph_rag.scriptes.file_reader import FileReader
from graph_rag.scriptes.text_splitter import get_splitter
from graph_rag.scriptes.entity_extractor import EntityExtractor
from storage.neo4j_manager import Neo4jManager
from storage.chroma_manager import ChromaManager, BGEEmbedder
from generation.context_builder import ContextBuilder
from generation.answer_generator import AnswerGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GraphRAGPipeline:
    """Graph RAG 完整流程管理器"""

    def __init__(self,
                 # 数据库配置
                 neo4j_entity_label: str = "Entity",  # 实体标签，用于区分不同的数据集
                 chroma_entity_collection: str = "entities",  # 实体集合名称
                 chroma_chunk_collection: str = "chunk",  # 文本块集合名称

                 # 文本分块配置
                 chunk_strategy: str = "semantic",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50, ):
        """
        初始化 Graph RAG 流程

        Args:
            neo4j_entity_label: Neo4j 实体标签（可选，默认从配置文件读取，用于区分不同的数据集）
            chroma_entity_collection: Chroma 实体集合名称（可选，默认从配置文件读取）
            chroma_chunk_collection: Chroma 文本块集合名称（可选，默认从配置文件读取）
            chunk_strategy: 文本分块策略（sentence/fixed/paragraph/semantic/custom/n_sentence）
            chunk_size: 分块大小
            chunk_overlap: 分块重叠
        """
        logger.info("=" * 80)
        logger.info("初始化 Graph RAG 流程...")
        logger.info("=" * 80)

        self.max_workers = 3
        self.lock = Lock()

        # 1. 文档读取器
        self.file_reader = FileReader()
        logger.info("✓ 文档读取器已加载")

        # 2. 文本分块器（使用 script 中的 get_splitter）
        # 根据策略构建参数
        splitter_params = {}
        if chunk_strategy == "semantic":
            splitter_params = {"max_length": chunk_size}
        elif chunk_strategy == "fixed":
            splitter_params = {"chunk_size": chunk_size, "overlap": chunk_overlap}
        elif chunk_strategy == "paragraph":
            splitter_params = {"min_length": 10}
        elif chunk_strategy == "n_sentence":
            splitter_params = {"n": chunk_size // 100}  # 根据 chunk_size 估算句子数

        self.text_splitter = get_splitter(chunk_strategy, **splitter_params)
        logger.info(f"✓ 文本分块器已加载: {chunk_strategy}")

        # 3. 实体抽取器（使用 script 中的 EntityExtractor，支持从配置文件读取）
        self.entity_extractor = EntityExtractor()
        logger.info(f"✓ 实体抽取器已加载: {self.entity_extractor.client.llm}")

        # 4. Neo4j 管理器（支持从配置文件读取表名）
        self.neo4j_manager = Neo4jManager(
            entity_label=neo4j_entity_label
        )
        self.neo4j_manager.create_indexes()
        logger.info(f"✓ Neo4j 已连接: {self.neo4j_manager.uri}")
        logger.info(f"  实体标签: {self.neo4j_manager.entity_label}")

        # 5. Embedding 模型
        self.embedder = BGEEmbedder()
        logger.info(f"✓ Embedding 模型已加载")

        # 6. Chroma 管理器（支持从配置文件读取集合名称）
        self.chroma_manager = ChromaManager(
            embedder=self.embedder,
            entity_collection_name=chroma_entity_collection,
            chunk_collection_name=chroma_chunk_collection
        )
        logger.info(f"✓ Chroma 已初始化")
        logger.info(
            f"  实体集合: {self.chroma_manager.entity_collection_name}, 文本块集合: {self.chroma_manager.chunk_collection_name}")

        # 7. 上下文构建器
        self.context_builder = ContextBuilder(
            max_entities=20,
            max_relations=30,
            max_chunks=5
        )
        logger.info("✓ 上下文构建器已加载")

        # 8. 答案生成器（支持从配置文件读取）
        self.answer_generator = AnswerGenerator()
        logger.info("✓ 答案生成器已加载")

        logger.info("=" * 80)
        logger.info("Graph RAG 流程初始化完成！")
        logger.info("=" * 80)

    def _extract_entities(self, text: str, chunk_id: str) -> Dict:
        """
        带重试机制的实体抽取

        Args:
            text: 输入文本
            chunk_id: 文本块ID

        Returns:
            抽取结果
        """
        # 调用抽取API（会等待返回）
        result = self.entity_extractor.extract(text)

        if result and (result.get('entities') or result.get('relations')):
            return {
                'chunk_id': chunk_id,
                'entities': result.get('entities', []),
                'relations': result.get('relations', [])
            }

        # 返回空结果但不重试
        return {
            'chunk_id': chunk_id,
            'entities': [],
            'relations': []
        }

    def _process_chunk(self, chunk: str, chunk_id: str, filename: str) -> Dict:
        """
        处理单个文本块（用于并发）

        Args:
            chunk: 文本内容
            chunk_id: 文本块ID
            filename: 源文件名

        Returns:
            处理结果
        """
        # Step 1: 实体关系抽取（带重试）
        extraction_result = self._extract_entities(chunk, chunk_id)

        entities = extraction_result.get('entities', [])
        relations = extraction_result.get('relations', [])

        # Step 2: 存储到 Neo4j（需要加锁，避免并发冲突）
        with self.lock:
            if entities:
                self.neo4j_manager.add_entities(entities, source_chunk=chunk_id)
            if relations:
                self.neo4j_manager.add_relations(relations, source_chunk=chunk_id)

        # Step 3: 存储到 Chroma
        entity_names = [e['name'] for e in entities]

        # 存储实体向量
        if entities:
            with self.lock:
                self.chroma_manager.add_entities(entities)

        # 存储文本块向量
        with self.lock:
            self.chroma_manager.add_chunk(
                chunk_id=chunk_id,
                text=chunk,
                source_file=filename,
                entity_names=entity_names
            )

        return {
            'chunk_id': chunk_id,
            'entity_count': len(entities),
            'relation_count': len(relations),
            'success': 'error' not in extraction_result
        }

    def ingest_documents(self,
                         file_paths: Optional[List[str]] = None,
                         directory: Optional[str] = None,
                         recursive: bool = True,
                         use_parallel: bool = True) -> Dict:
        """
        文档摄入流程：读取 → 分块 → 抽取 → 存储
        支持并发处理以提高效率

        Args:
            file_paths: 文件路径列表
            directory: 目录路径
            recursive: 是否递归读取子目录
            use_parallel: 是否使用并发处理

        Returns:
            摄入统计信息
        """
        logger.info("\n" + "=" * 80)
        logger.info("开始文档摄入流程...")
        logger.info("=" * 80)

        # Step 1: 读取文档
        logger.info("\n[1/4] 读取文档...")
        if directory:
            documents = self.file_reader.read_directory(
                directory,
                recursive=recursive,
                return_dict=True
            )
            logger.info(f"✓ 从目录读取: {len(documents)} 个文件")
        elif file_paths:
            documents = self.file_reader.read_files(file_paths, return_dict=True)
            logger.info(f"✓ 读取文件: {len(documents)} 个")
        else:
            raise ValueError("必须提供 file_paths 或 directory")

        total_entities = 0
        total_relations = 0
        total_chunks = 0
        failed_chunks = 0

        # 准备所有处理任务
        tasks = []
        for filename, content in documents.items():
            if not content or content.startswith("[错误"):
                logger.warning(f"跳过文件: {filename}")
                continue

            logger.info(f"\n处理文件: {filename}")

            # Step 2: 文本分块
            logger.info("  [2/4] 文本分块...")
            chunks = self.text_splitter.split(content)
            logger.info(f"  ✓ 分块完成: {len(chunks)} 个块")
            total_chunks += len(chunks)

            # 收集任务
            for i, chunk in enumerate(chunks):
                chunk_id = f"{filename}_chunk_{i}"
                tasks.append((chunk, chunk_id, filename))

        # Step 3 & 4: 实体抽取 + 存储（并发或串行）
        logger.info(f"\n[3/4] 实体关系抽取 (共 {len(tasks)} 个块)...")
        logger.info(f"[4/4] 存储到数据库...")

        if use_parallel and len(tasks) > 1:
            logger.info(f"✓ 使用并发处理 (最大 {self.max_workers} 线程)")

            # 并发处理
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._process_chunk, chunk, chunk_id, filename): chunk_id
                    for chunk, chunk_id, filename in tasks
                }

                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    chunk_id = futures[future]

                    try:
                        result = future.result()
                        total_entities += result['entity_count']
                        total_relations += result['relation_count']

                        if not result['success']:
                            failed_chunks += 1

                        if completed % 10 == 0 or completed == len(tasks):
                            logger.info(f"  进度: {completed}/{len(tasks)}")

                    except Exception as e:
                        logger.error(f"  处理失败 {chunk_id}: {str(e)}")
                        failed_chunks += 1
        else:
            logger.info("✓ 使用串行处理")

            # 串行处理
            for i, (chunk, chunk_id, filename) in enumerate(tasks, 1):
                try:
                    result = self._process_chunk(chunk, chunk_id, filename)
                    total_entities += result['entity_count']
                    total_relations += result['relation_count']

                    if not result['success']:
                        failed_chunks += 1

                    if i % 10 == 0 or i == len(tasks):
                        logger.info(f"  进度: {i}/{len(tasks)}")

                except Exception as e:
                    logger.error(f"  处理失败 {chunk_id}: {str(e)}")
                    failed_chunks += 1

        # 统计信息
        logger.info("\n" + "=" * 80)
        logger.info("文档摄入完成！")
        logger.info("=" * 80)

        stats = {
            "documents_processed": len(documents),
            "total_chunks": total_chunks,
            "failed_chunks": failed_chunks,
            "total_entities_extracted": total_entities,
            "total_relations_extracted": total_relations,
            "neo4j_stats": self.neo4j_manager.get_statistics(),
            "chroma_stats": self.chroma_manager.get_statistics()
        }

        logger.info(f"\n摄入统计:")
        logger.info(f"  文档数: {stats['documents_processed']}")
        logger.info(f"  文本块: {stats['total_chunks']}")
        logger.info(f"  失败块: {stats['failed_chunks']}")
        logger.info(f"  图谱实体数: {stats['neo4j_stats']['entity_count']}")
        logger.info(f"  图谱关系数: {stats['neo4j_stats']['relation_count']}")

        return stats

    def query(self,
              question: str,
              top_k_entities: int = 5,
              top_k_chunks: int = 5,
              max_hops: int = 2,
              use_streaming: bool = False) -> Dict:
        """
        问答流程：检索 → 扩展 → 构建上下文 → 生成答案

        Args:
            question: 用户问题
            top_k_entities: 检索实体数量
            top_k_chunks: 检索文本块数量
            max_hops: 图扩展最大跳数
            use_streaming: 是否使用流式生成

        Returns:
            问答结果
        """
        logger.info("\n" + "=" * 80)
        logger.info(f"问题: {question}")
        logger.info("=" * 80)

        # Step 1: 向量检索
        logger.info("\n[1/4] 向量检索...")
        vector_results = self.chroma_manager.hybrid_search(
            question,
            top_k=max(top_k_entities, top_k_chunks)
        )

        entities = vector_results['entities'][:top_k_entities]
        chunks = vector_results['chunks'][:top_k_chunks]

        # Step 2: 图扩展
        logger.info("\n[2/4] 图扩展...")
        entity_names = [e['name'] for e in entities]

        if entity_names:
            subgraph = self.neo4j_manager.expand_subgraph(entity_names, max_hops=max_hops)
            expanded_entities = subgraph['entities']
            relations = subgraph['relations']

            logger.info("\n【RAG-图扩展】实体:")
            for e in expanded_entities:
                logger.info(f"  - {e.get('name')} ({e.get('type')})")

            logger.info("\n【RAG-图扩展】关系:")
            for r in relations:
                logger.info(
                    f"  - ({r.get('source')}) -[{r.get('relation')}]-> ({r.get('target')})"
                )
        else:
            expanded_entities = []
            relations = []
            logger.warning("未找到相关实体，跳过图扩展")

        # Step 3: 构建上下文
        logger.info("\n[3/4] 构建上下文...")
        context = self.context_builder.build_context(
            entities=expanded_entities,
            relations=relations,
            chunks=chunks,
            query=question
        )
        print(context)

        # Step 4: 生成答案
        logger.info("\n[4/4] 生成答案...")

        if use_streaming:
            logger.info("使用流式生成...")
            result = self.answer_generator.generate_streaming(
                context=context,
                query=question
            )
        else:
            result = self.answer_generator.generate(
                context=context,
                query=question
            )

        if result['success']:
            logger.info(f"✓ 答案生成成功 ({len(result['answer'])} 字符)")
        else:
            logger.error(f"✗ 答案生成失败: {result.get('error')}")

        # 组装完整结果
        full_result = {
            "question": question,
            "answer": result.get('answer', ''),
            "success": result['success'],
            "retrieval": {
                "entities": entities,
                "expanded_entities": expanded_entities,
                "relations": relations,
                "chunks": chunks
            },
            "context": context,
        }

        return full_result

    def batch_query(self, questions: List[str]) -> List[Dict]:
        """批量问答"""
        results = []
        for i, question in enumerate(questions, 1):
            logger.info(f"\n处理问题 {i}/{len(questions)}")
            result = self.query(question)
            results.append(result)
        return results

    def clear_all_data(self):
        """清空所有数据"""
        logger.warning("正在清空所有数据...")
        self.neo4j_manager.clear_database()
        self.chroma_manager.clear_all()
        logger.info("✓ 所有数据已清空")

    def get_statistics(self) -> Dict:
        """获取系统统计信息"""
        return {
            "neo4j": self.neo4j_manager.get_statistics(),
            "chroma": self.chroma_manager.get_statistics()
        }

    def close(self):
        """关闭所有连接"""
        self.neo4j_manager.close()
        logger.info("✓ 所有连接已关闭")


# 使用示例
if __name__ == "__main__":
    # 方式1: 从配置文件读取（推荐）
    pipeline = GraphRAGPipeline(
        neo4j_entity_label="Entity_project1",  # 指定 Neo4j 实体标签
        chroma_entity_collection="entities_project1",  # 指定 Chroma 实体集合
        chroma_chunk_collection="chunks_project1"  # 指定 Chroma 文本块集合
    )

    # ========== 摄入文档 ==========
    # stats = pipeline.ingest_documents(
    #     file_paths=["../test_files/思维模型.md"],
    #     recursive=True,
    #     use_parallel=True  # 启用并发
    # )
    #
    # print(f"\n处理文档: {stats['documents_processed']} 个")
    # print(f"生成文本块: {stats['total_chunks']} 个")
    # print(f"失败文本块: {stats['failed_chunks']} 个")
    # print(f"提取实体: {stats['neo4j_stats']['entity_count']} 个")
    # print(f"提取关系: {stats['neo4j_stats']['relation_count']} 个")

    # ========== 问答查询 ==========
    result = pipeline.query("""我从十一月初开始
断断续续的没了4首歌
不论是被抢还是觉得不合适换人
有的是试音直接被斩，有的是两三轮最后通知不合适有的甚至是录完了突然被抢
运营刚才告诉我上一首不合适立马给了我一首新的，但我真没信心不敢唱了""")
    print(f"\n问题: {result['question']}")
    print(f"答案: {result['answer']}")

    # 关闭连接
    pipeline.close()
