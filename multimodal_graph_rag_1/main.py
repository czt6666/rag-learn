"""主流程"""
import logging
import argparse
from pathlib import Path
from config import get_settings
from module_0_data_loader import DataDownloader
from module_1_parser import DocumentParser
from module_2_vector_index import VectorIndexBuilder
from module_3_knowledge_graph import GraphBuilder
from module_4_7_retrieval_generation import MultimodalRAG
from module_8_evaluation import Evaluator

logger = logging.getLogger(__name__)


class MultimodalGraphRAGPipeline:
    """完整流程"""
    
    def __init__(self, llm_provider: str = "openai", reset_index: bool = False):
        self.settings = get_settings()
        self.llm_provider = llm_provider
        self.reset_index = reset_index
        
        self.data_loader = DataDownloader()
        self.parser = DocumentParser()
        self.vector_builder = VectorIndexBuilder()
        self.graph_builder = GraphBuilder(llm_provider=llm_provider)
        self.evaluator = Evaluator()
        
        logger.info(f"初始化完成，LLM: {llm_provider}")
    
    def setup(self):
        """设置环境"""
        logger.info("设置环境...")
        self.data_loader.setup_directories()
        checks = self.data_loader.verify_dataset()
        if not all(checks.values()):
            logger.warning("数据集不完整")
    
    def load_data(self):
        """加载数据"""
        logger.info("加载数据...")
        self.docs = self.data_loader.load_docs()
        self.qa_pairs = self.data_loader.load_qa()
        logger.info(f"加载: {len(self.docs)} 文档, {len(self.qa_pairs)} QA")
    
    def parse_documents(self):
        """解析文档"""
        logger.info("解析文档...")
        self.chunks = self.parser.parse_all_documents(self.docs)
        output_path = Path(self.settings.output_path) / "chunks.jsonl"
        self.parser.save_chunks(self.chunks, str(output_path))
        logger.info(f"解析: {len(self.chunks)} chunks")
    
    def build_vector_index(self):
        """构建向量索引"""
        logger.info("构建向量索引...")
        self.vector_builder.build_index(self.chunks, reset=self.reset_index)
    
    def build_knowledge_graph(self):
        """构建知识图谱"""
        logger.info("构建知识图谱...")
        self.graph_builder.build_graph(self.chunks, reset=self.reset_index)
    
    def create_rag_system(self):
        """创建 RAG 系统"""
        logger.info("创建 RAG 系统...")
        self.rag = MultimodalRAG(
            vector_builder=self.vector_builder,
            knowledge_graph=self.graph_builder.kg,
            llm_provider=self.llm_provider
        )
    
    def interactive_query(self):
        """交互式查询"""
        print("\n" + "="*60)
        print("进入交互式查询模式")
        print("输入 'quit' 或 'exit' 退出")
        print("="*60 + "\n")
        
        while True:
            try:
                question = input("\n请输入问题: ").strip()
                
                if question.lower() in ['quit', 'exit', '退出']:
                    break
                
                if not question:
                    continue
                
                result = self.rag.query(question)
                
                print("\n" + "="*60)
                print(f"问题: {result['question']}")
                print("="*60)
                print(f"\n答案:\n{result['answer']}\n")
                
                metadata = result['metadata']
                print(f"检索统计: 向量 {metadata['vector_hits']} 个, "
                      f"图谱 {metadata['graph_hits']} 个")
                print("="*60)
                
            except KeyboardInterrupt:
                print("\n\n再见!")
                break
            except Exception as e:
                logger.error(f"查询出错: {e}")
    
    def evaluate(self, sample_size=None):
        """评估系统"""
        logger.info("开始评估...")
        
        qa_pairs = self.qa_pairs
        if sample_size and sample_size < len(qa_pairs):
            import random
            qa_pairs = random.sample(qa_pairs, sample_size)
        
        results = self.evaluator.evaluate_all(
            qa_pairs=qa_pairs,
            rag_system=self.rag,
            save_path=str(Path(self.settings.output_path) / "evaluation.json")
        )
        
        return results
    
    def run_full_pipeline(self, skip_build: bool = False):
        """运行完整流程"""
        self.setup()
        self.load_data()
        
        if not skip_build:
            self.parse_documents()
            self.build_vector_index()
            self.build_knowledge_graph()
        else:
            self.parse_documents()
        
        self.create_rag_system()
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'graph_builder'):
            self.graph_builder.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多模态 Graph-RAG")
    
    parser.add_argument('--mode', choices=['build', 'query', 'evaluate', 'full'],
                       default='full', help='运行模式')
    parser.add_argument('--llm', choices=['openai', 'anthropic'],
                       default='openai', help='LLM 提供商')
    parser.add_argument('--reset', action='store_true', help='重置索引')
    parser.add_argument('--eval-sample', type=int, default=None, help='评估样本数')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='日志级别')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    pipeline = MultimodalGraphRAGPipeline(
        llm_provider=args.llm,
        reset_index=args.reset
    )
    
    try:
        if args.mode == 'build':
            pipeline.setup()
            pipeline.load_data()
            pipeline.parse_documents()
            pipeline.build_vector_index()
            pipeline.build_knowledge_graph()
        
        elif args.mode == 'query':
            pipeline.run_full_pipeline(skip_build=True)
            pipeline.interactive_query()
        
        elif args.mode == 'evaluate':
            pipeline.run_full_pipeline(skip_build=not args.reset)
            pipeline.evaluate(sample_size=args.eval_sample)
        
        elif args.mode == 'full':
            pipeline.run_full_pipeline(skip_build=False)
            pipeline.interactive_query()
    
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()
