"""模块 4-7：检索、融合与生成"""
import logging
import re
from typing import List, Dict, Optional
from config import get_settings
from module_2_vector_index import VectorIndexBuilder
from module_3_knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


class VectorRetriever:
    """模块 4：向量检索"""
    
    def __init__(self, index_builder: VectorIndexBuilder):
        self.index_builder = index_builder
        self.settings = get_settings()
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        if top_k is None:
            top_k = self.settings.vector_top_k
        
        logger.info(f"向量检索: {query} (top_k={top_k})")
        results = self.index_builder.search(query, top_k=top_k)
        logger.info(f"检索到 {len(results)} 个结果")
        return results


class GraphRetriever:
    """模块 5：图检索"""
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        self.settings = get_settings()
    
    def extract_entities_from_query(self, query: str) -> List[str]:
        """从查询提取实体"""
        words = re.findall(r'\w+', query)
        stopwords = {'是', '的', '了', '在', '有', '和', '与', '或', '等', 
                    'what', 'is', 'are', 'the', 'a', 'an', 'of', 'to'}
        entities = [w for w in words if len(w) > 1 and w.lower() not in stopwords]
        return entities
    
    def retrieve(self, query: str, vector_results: List[Dict] = None) -> List[Dict]:
        logger.info(f"图检索: {query}")
        
        entities_from_query = self.extract_entities_from_query(query)
        entities_from_vector = []
        
        if vector_results:
            for result in vector_results:
                text = result.get('document', '')
                entities_from_vector.extend(self.extract_entities_from_query(text))
        
        all_entities = list(set(entities_from_query + entities_from_vector))
        
        if not all_entities:
            return []
        
        logger.info(f"提取的实体: {all_entities[:10]}")
        graph_data = self.kg.find_entities_in_text(all_entities)
        logger.info(f"图检索到 {len(graph_data)} 个节点")
        return graph_data


class RetrieverFusion:
    """模块 6：检索融合"""
    
    def fuse(self, vector_results: List[Dict], graph_results: List[Dict]) -> Dict:
        logger.info("融合检索结果")
        
        vector_docs = []
        for result in vector_results:
            vector_docs.append({
                'chunk_id': result.get('chunk_id'),
                'text': result.get('document'),
                'metadata': result.get('metadata'),
                'score': 1.0 / (1.0 + result.get('distance', 0))
            })
        
        graph_nodes = []
        graph_relations = []
        for result in graph_results:
            node = result.get('node', {})
            neighbors = result.get('neighbors', [])
            relations = result.get('relations', [])
            
            graph_nodes.append(node)
            graph_nodes.extend(neighbors)
            graph_relations.extend(relations)
        
        unique_nodes = {node.get('name'): node for node in graph_nodes if node.get('name')}
        
        merged = {
            'vector_hits': vector_docs,
            'graph_hits': {
                'nodes': list(unique_nodes.values()),
                'relations': list(set(graph_relations))
            },
            'summary': {
                'vector_count': len(vector_docs),
                'graph_nodes_count': len(unique_nodes),
                'graph_relations_count': len(set(graph_relations))
            }
        }
        
        return merged


class NLGGenerator:
    """模块 7：自然语言生成"""
    
    def __init__(self, api_key: Optional[str] = None, provider: str = "openai"):
        self.settings = get_settings()
        self.provider = provider
        
        if provider == "openai":
            from openai import OpenAI
            api_key = api_key or self.settings.openai_api_key
            self.client = OpenAI(api_key=api_key)
            self.model = "gpt-4o-mini"
        elif provider == "anthropic":
            from anthropic import Anthropic
            api_key = api_key or self.settings.anthropic_api_key
            self.client = Anthropic(api_key=api_key)
            self.model = "claude-3-5-sonnet-20241022"
    
    def generate_answer(self, query: str, fused_results: Dict) -> str:
        """生成答案"""
        context = self._build_context(fused_results)
        
        prompt = f"""基于以下信息回答问题。

用户问题: {query}

检索文档:
{context['vector_context']}

知识图谱:
{context['graph_context']}

要求:
1. 基于提供的信息生成完整答案
2. 如果信息中提到图像，可以引用
3. 保持答案自然流畅

答案:"""
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是专业的问答助手。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                answer = response.choices[0].message.content
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    temperature=0.7,
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = response.content[0].text
            
            return answer
        except Exception as e:
            logger.error(f"生成失败: {e}")
            return f"抱歉，无法生成答案: {e}"
    
    def _build_context(self, fused_results: Dict) -> Dict:
        """构建上下文"""
        vector_hits = fused_results.get('vector_hits', [])
        vector_context_parts = []
        
        for i, hit in enumerate(vector_hits, 1):
            text = hit.get('text', '')[:500]
            metadata = hit.get('metadata', {})
            image_ids = metadata.get('image_ids', '')
            
            context_part = f"{i}. {text}"
            if image_ids:
                context_part += f" [相关图像: {image_ids}]"
            
            vector_context_parts.append(context_part)
        
        vector_context = "\n".join(vector_context_parts) or "无相关文档"
        
        graph_hits = fused_results.get('graph_hits', {})
        nodes = graph_hits.get('nodes', [])
        relations = graph_hits.get('relations', [])
        
        node_names = [node.get('name') for node in nodes if node.get('name')]
        graph_context = f"相关实体: {', '.join(node_names[:20])}\n"
        graph_context += f"关系类型: {', '.join(set(relations[:10]))}"
        
        return {
            'vector_context': vector_context,
            'graph_context': graph_context or "无相关图谱信息"
        }


class MultimodalRAG:
    """完整的多模态 RAG 系统"""
    
    def __init__(
        self, 
        vector_builder: VectorIndexBuilder,
        knowledge_graph: KnowledgeGraph,
        llm_provider: str = "openai",
        api_key: Optional[str] = None
    ):
        self.vector_retriever = VectorRetriever(vector_builder)
        self.graph_retriever = GraphRetriever(knowledge_graph)
        self.fusion = RetrieverFusion()
        self.generator = NLGGenerator(api_key=api_key, provider=llm_provider)
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        """完整查询流程"""
        logger.info(f"\n{'='*50}")
        logger.info(f"查询: {question}")
        logger.info(f"{'='*50}")
        
        vector_results = self.vector_retriever.retrieve(question, top_k=top_k)
        graph_results = self.graph_retriever.retrieve(question, vector_results)
        fused_results = self.fusion.fuse(vector_results, graph_results)
        answer = self.generator.generate_answer(question, fused_results)
        
        return {
            'question': question,
            'answer': answer,
            'retrieval_results': fused_results,
            'metadata': {
                'vector_hits': len(vector_results),
                'graph_hits': len(graph_results)
            }
        }
