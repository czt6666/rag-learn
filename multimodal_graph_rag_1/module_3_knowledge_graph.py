"""模块 3：知识图构建"""
import logging
import json
from typing import List, Dict, Optional
from neo4j import GraphDatabase
from tqdm import tqdm
from config import get_settings
from module_1_parser import MultimodalChunk

logger = logging.getLogger(__name__)


class LLMEntityExtractor:
    """使用 LLM 提取实体和关系"""
    
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
    
    def extract_entities_and_relations(self, text: str) -> Dict:
        """从文本中提取实体和关系"""
        prompt = f"""从以下文本中提取实体和关系。

文本: {text}

返回 JSON 格式:
{{
  "entities": [{{"name": "实体名", "type": "类型"}}],
  "relations": [{{"source": "实体1", "relation": "关系", "target": "实体2"}}]
}}

只返回 JSON:"""
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是知识图谱构建助手。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                result = json.loads(response.choices[0].message.content)
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text
                start = content.find('{')
                end = content.rfind('}') + 1
                result = json.loads(content[start:end])
            
            return result
        except Exception as e:
            logger.error(f"提取失败: {e}")
            return {"entities": [], "relations": []}


class KnowledgeGraph:
    """Neo4j 知识图谱"""
    
    def __init__(self):
        self.settings = get_settings()
        logger.info(f"连接 Neo4j: {self.settings.neo4j_uri}")
        self.driver = GraphDatabase.driver(
            self.settings.neo4j_uri,
            auth=(self.settings.neo4j_user, self.settings.neo4j_password)
        )
    
    def create_entity(self, name: str, entity_type: str, metadata: Dict = None):
        """创建实体节点"""
        metadata = metadata or {}
        with self.driver.session() as session:
            query = f"""
            MERGE (e:{entity_type} {{name: $name}})
            SET e += $metadata
            RETURN e
            """
            session.run(query, name=name, metadata=metadata)
    
    def create_relation(self, source: str, relation: str, target: str, metadata: Dict = None):
        """创建关系"""
        metadata = metadata or {}
        with self.driver.session() as session:
            query = """
            MERGE (s {name: $source})
            MERGE (t {name: $target})
            MERGE (s)-[r:%s]->(t)
            SET r += $metadata
            """ % relation.replace(' ', '_').upper()
            session.run(query, source=source, target=target, metadata=metadata)
    
    def add_chunk_to_graph(self, chunk: MultimodalChunk, entities: List[Dict], relations: List[Dict]):
        """将 chunk 添加到图谱"""
        metadata = {
            'doc_id': chunk.doc_id,
            'chunk_id': chunk.chunk_id,
            'image_ids': ','.join(chunk.image_ids)
        }
        
        for entity in entities:
            self.create_entity(
                name=entity['name'],
                entity_type=entity.get('type', 'ENTITY'),
                metadata=metadata
            )
        
        for relation in relations:
            self.create_relation(
                source=relation['source'],
                relation=relation['relation'],
                target=relation['target'],
                metadata=metadata
            )
    
    def find_entities_in_text(self, entities: List[str]) -> List[Dict]:
        """查找实体"""
        if not entities:
            return []
        
        with self.driver.session() as session:
            query = """
            MATCH (n)
            WHERE n.name IN $entities
            OPTIONAL MATCH path = (n)-[r]-(m)
            RETURN n, collect(DISTINCT m) as neighbors, collect(DISTINCT r) as relations
            """
            result = session.run(query, entities=entities)
            
            graph_data = []
            for record in result:
                node = dict(record['n'])
                neighbors = [dict(n) for n in record['neighbors'] if n]
                relations = [r.type for r in record['relations'] if r]
                
                graph_data.append({
                    'node': node,
                    'neighbors': neighbors,
                    'relations': relations
                })
            
            return graph_data
    
    def clear_graph(self):
        """清空图谱"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    
    def close(self):
        """关闭连接"""
        self.driver.close()


class GraphBuilder:
    """知识图谱构建器"""
    
    def __init__(self, llm_provider: str = "openai", api_key: Optional[str] = None):
        self.extractor = LLMEntityExtractor(api_key=api_key, provider=llm_provider)
        self.kg = KnowledgeGraph()
    
    def build_graph(self, chunks: List[MultimodalChunk], reset: bool = False):
        """构建知识图谱"""
        if reset:
            self.kg.clear_graph()
        
        logger.info(f"开始构建知识图谱: {len(chunks)} 个 chunks")
        
        for chunk in tqdm(chunks, desc="构建知识图谱"):
            result = self.extractor.extract_entities_and_relations(chunk.joined_text)
            entities = result.get('entities', [])
            relations = result.get('relations', [])
            
            if entities or relations:
                self.kg.add_chunk_to_graph(chunk, entities, relations)
        
        logger.info("知识图谱构建完成")
    
    def close(self):
        self.kg.close()
