"""
Neo4j 图数据库管理模块
负责实体和关系的存储、查询、更新
"""

import logging
import os
from typing import List, Dict, Optional, Set
from neo4j import GraphDatabase

# 尝试从配置文件导入，如果失败则使用环境变量或默认值
try:
    from graph_rag.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
except ImportError:
    # 如果无法导入配置文件，则从环境变量读取
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jManager:
    """Neo4j 数据库管理器"""

    def __init__(self, uri: str = None, username: str = None, password: str = None,
                 entity_label: str = "Entity"):
        """
        初始化 Neo4j 连接

        Args:
            uri: Neo4j 数据库地址，如 "bolt://localhost:7687"
                 如果为 None，则从配置文件或环境变量读取
            username: 用户名，如果为 None，则从配置文件或环境变量读取
            password: 密码，如果为 None，则从配置文件或环境变量读取
            entity_label: 实体标签名称，默认为 "Entity"，可用于区分不同的数据集
                         例如："Entity_default", "Entity_project1" 等
        """
        # 如果参数未提供，则从配置文件或环境变量读取
        self.uri = uri if uri is not None else NEO4J_URI
        self.username = username if username is not None else NEO4J_USERNAME
        self.password = password if password is not None else NEO4J_PASSWORD
        self.entity_label = entity_label
        
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        logger.info(f"✓ 已连接到 Neo4j: {self.uri}")
        logger.info(f"  使用实体标签: {entity_label}")

    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            logger.info("✓ Neo4j 连接已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def clear_database(self):
        """清空当前标签的所有数据（谨慎使用）"""
        with self.driver.session() as session:
            session.run(f"MATCH (n:{self.entity_label}) DETACH DELETE n")
        logger.info(f"✓ 已清空标签 {self.entity_label} 的所有数据")

    def create_indexes(self):
        """创建索引以提升查询性能"""
        with self.driver.session() as session:
            # 为实体名称创建索引（使用动态标签）
            session.run(f"CREATE INDEX entity_name IF NOT EXISTS FOR (e:{self.entity_label}) ON (e.name)")
            # 为实体类型创建索引
            session.run(f"CREATE INDEX entity_type IF NOT EXISTS FOR (e:{self.entity_label}) ON (e.type)")
        logger.info("✓ 索引已创建")

    def add_entity(self, name: str, entity_type: str, description: str = "",
                   source_chunk: str = "") -> bool:
        """
        添加单个实体

        Args:
            name: 实体名称
            entity_type: 实体类型
            description: 实体描述
            source_chunk: 来源文本块ID

        Returns:
            是否成功
        """
        query = f"""
        MERGE (e:{self.entity_label} {{name: $name}})
        ON CREATE SET 
            e.type = $type,
            e.description = $description,
            e.source_chunks = [$source_chunk],
            e.created_at = datetime()
        ON MATCH SET
            e.description = CASE 
                WHEN e.description = '' THEN $description 
                ELSE e.description 
            END,
            e.source_chunks = CASE
                WHEN NOT $source_chunk IN e.source_chunks 
                THEN e.source_chunks + $source_chunk
                ELSE e.source_chunks
            END
        RETURN e
        """

        try:
            with self.driver.session() as session:
                session.run(query,
                            name=name,
                            type=entity_type,
                            description=description,
                            source_chunk=source_chunk)
            return True
        except Exception as e:
            logger.error(f"添加实体失败 {name}: {str(e)}")
            return False

    def add_entities(self, entities: List[Dict], source_chunk: str = "") -> int:
        """
        批量添加实体

        Args:
            entities: 实体列表 [{"name": "", "type": "", "description": ""}]
            source_chunk: 来源文本块ID

        Returns:
            成功添加的数量
        """
        count = 0
        for entity in entities:
            if self.add_entity(
                    name=entity.get('name', ''),
                    entity_type=entity.get('type', '其他'),
                    description=entity.get('description', ''),
                    source_chunk=source_chunk
            ):
                count += 1

        logger.info(f"✓ 批量添加实体: {count}/{len(entities)}")
        return count

    def add_relation(self, source: str, target: str, relation: str,
                     source_chunk: str = "", weight: float = 1.0) -> bool:
        """
        添加关系

        Args:
            source: 源实体名称
            target: 目标实体名称
            relation: 关系类型
            source_chunk: 来源文本块ID
            weight: 关系权重

        Returns:
            是否成功
        """
        # 将关系类型转换为合法的Neo4j关系名（去除特殊字符）
        safe_relation = relation.replace(' ', '_').replace('-', '_')

        query = f"""
        MATCH (s:{self.entity_label} {{name: $source}})
        MATCH (t:{self.entity_label} {{name: $target}})
        MERGE (s)-[r:{safe_relation}]->(t)
        ON CREATE SET
            r.relation_type = $relation,
            r.weight = $weight,
            r.source_chunks = [$source_chunk],
            r.created_at = datetime()
        ON MATCH SET
            r.weight = r.weight + $weight,
            r.source_chunks = CASE
                WHEN NOT $source_chunk IN r.source_chunks
                THEN r.source_chunks + $source_chunk
                ELSE r.source_chunks
            END
        RETURN r
        """

        try:
            with self.driver.session() as session:
                result = session.run(query,
                                     source=source,
                                     target=target,
                                     relation=relation,
                                     source_chunk=source_chunk,
                                     weight=weight)
                return result.single() is not None
        except Exception as e:
            logger.error(f"添加关系失败 {source}-[{relation}]->{target}: {str(e)}")
            return False

    def add_relations(self, relations: List[Dict], source_chunk: str = "") -> int:
        """
        批量添加关系

        Args:
            relations: 关系列表 [{"source": "", "target": "", "relation": ""}]
            source_chunk: 来源文本块ID

        Returns:
            成功添加的数量
        """
        count = 0
        for relation in relations:
            if self.add_relation(
                    source=relation.get('source', ''),
                    target=relation.get('target', ''),
                    relation=relation.get('relation', ''),
                    source_chunk=source_chunk
            ):
                count += 1

        logger.info(f"✓ 批量添加关系: {count}/{len(relations)}")
        return count

    def get_entity(self, name: str) -> Optional[Dict]:
        """
        获取实体信息

        Args:
            name: 实体名称

        Returns:
            实体信息字典
        """
        query = f"""
        MATCH (e:{self.entity_label} {{name: $name}})
        RETURN e.name as name, e.type as type, e.description as description
        """

        with self.driver.session() as session:
            result = session.run(query, name=name)
            record = result.single()
            if record:
                return dict(record)
        return None

    def search_entities(self, keyword: str, entity_type: str = None, limit: int = 10) -> List[Dict]:
        """
        搜索实体

        Args:
            keyword: 关键词（模糊匹配）
            entity_type: 实体类型过滤
            limit: 返回数量限制

        Returns:
            实体列表
        """
        if entity_type:
            query = f"""
            MATCH (e:{self.entity_label})
            WHERE e.name CONTAINS $keyword AND e.type = $type
            RETURN e.name as name, e.type as type, e.description as description
            LIMIT $limit
            """
            params = {"keyword": keyword, "type": entity_type, "limit": limit}
        else:
            query = f"""
            MATCH (e:{self.entity_label})
            WHERE e.name CONTAINS $keyword
            RETURN e.name as name, e.type as type, e.description as description
            LIMIT $limit
            """
            params = {"keyword": keyword, "limit": limit}

        with self.driver.session() as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]

    def get_entity_relations(self, name: str, direction: str = "both",
                             max_hops: int = 1) -> Dict:
        """
        获取实体的关系子图

        Args:
            name: 实体名称
            direction: 关系方向 "out"(出边), "in"(入边), "both"(双向)
            max_hops: 最大跳数

        Returns:
            {
                "center_entity": {...},
                "related_entities": [...],
                "relations": [...]
            }
        """
        if direction == "out":
            direction_clause = "-[r]->"
        elif direction == "in":
            direction_clause = "<-[r]-"
        else:
            direction_clause = "-[r]-"

        query = f"""
        MATCH path = (center:{self.entity_label} {{name: $name}}){direction_clause}(related:{self.entity_label})
        WHERE length(path) <= $max_hops
        RETURN 
            center.name as center_name,
            center.type as center_type,
            center.description as center_description,
            related.name as related_name,
            related.type as related_type,
            related.description as related_description,
            type(r) as relation_type,
            r.weight as weight
        """

        with self.driver.session() as session:
            result = session.run(query, name=name, max_hops=max_hops)
            records = list(result)

            if not records:
                return {
                    "center_entity": None,
                    "related_entities": [],
                    "relations": []
                }

            # 中心实体
            first = records[0]
            center_entity = {
                "name": first["center_name"],
                "type": first["center_type"],
                "description": first["center_description"]
            }

            # 相关实体和关系
            related_entities = []
            relations = []
            seen_entities = set()

            for record in records:
                related_name = record["related_name"]
                if related_name not in seen_entities:
                    seen_entities.add(related_name)
                    related_entities.append({
                        "name": related_name,
                        "type": record["related_type"],
                        "description": record["related_description"]
                    })

                relations.append({
                    "source": first["center_name"],
                    "target": related_name,
                    "relation": record["relation_type"],
                    "weight": record["weight"]
                })

            return {
                "center_entity": center_entity,
                "related_entities": related_entities,
                "relations": relations
            }

    def expand_subgraph(self, entity_names: List[str], max_hops: int = 2) -> Dict:
        """
        从多个实体扩展子图

        Args:
            entity_names: 实体名称列表
            max_hops: 最大跳数

        Returns:
            {
                "entities": [...],
                "relations": [...]
            }
        """
        query = f"""
        MATCH path = (e1:{self.entity_label})-[r*1..2]-(e2:{self.entity_label})
        WHERE e1.name IN $names
        WITH e1, e2, relationships(path) as rels
        UNWIND rels as rel
        RETURN DISTINCT
            startNode(rel).name as source,
            startNode(rel).type as source_type,
            startNode(rel).description as source_desc,
            endNode(rel).name as target,
            endNode(rel).type as target_type,
            endNode(rel).description as target_desc,
            type(rel) as relation_type,
            rel.weight as weight
        """

        with self.driver.session() as session:
            result = session.run(query, names=entity_names)
            records = list(result)

            entities_dict = {}
            relations = []

            for record in records:
                # 收集实体
                source_name = record["source"]
                target_name = record["target"]

                if source_name not in entities_dict:
                    entities_dict[source_name] = {
                        "name": source_name,
                        "type": record["source_type"],
                        "description": record["source_desc"]
                    }

                if target_name not in entities_dict:
                    entities_dict[target_name] = {
                        "name": target_name,
                        "type": record["target_type"],
                        "description": record["target_desc"]
                    }

                # 收集关系
                relations.append({
                    "source": source_name,
                    "target": target_name,
                    "relation": record["relation_type"],
                    "weight": record["weight"]
                })

            return {
                "entities": list(entities_dict.values()),
                "relations": relations
            }

    def get_statistics(self) -> Dict:
        """获取图数据库统计信息"""
        with self.driver.session() as session:
            # 实体数量
            entity_count = session.run(f"MATCH (e:{self.entity_label}) RETURN count(e) as count").single()["count"]

            # 关系数量（只统计当前标签实体的关系）
            relation_count = session.run(f"MATCH (:{self.entity_label})-[r]->(:{self.entity_label}) RETURN count(r) as count").single()["count"]

            # 实体类型分布
            type_dist = session.run(f"""
                MATCH (e:{self.entity_label})
                RETURN e.type as type, count(e) as count
                ORDER BY count DESC
            """)
            type_distribution = {record["type"]: record["count"] for record in type_dist}

            # 关系类型分布
            rel_dist = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as relation, count(r) as count
                ORDER BY count DESC
                LIMIT 20
            """)
            relation_distribution = {record["relation"]: record["count"] for record in rel_dist}

        return {
            "entity_count": entity_count,
            "relation_count": relation_count,
            "type_distribution": type_distribution,
            "relation_distribution": relation_distribution
        }


# 使用示例
if __name__ == "__main__":
    # 方式1: 从配置文件读取（推荐）
    manager = Neo4jManager()
    
    # 方式2: 手动指定连接信息（会覆盖配置文件）
    # manager = Neo4jManager(
    #     uri="bolt://localhost:7687",
    #     username="neo4j",
    #     password="12345678"
    # )

    # 创建索引
    manager.create_indexes()

    # 添加实体
    entities = [
        {"name": "阿里巴巴", "type": "组织", "description": "中国最大的电子商务公司"},
        {"name": "马云", "type": "人物", "description": "阿里巴巴创始人"},
        {"name": "杭州", "type": "地点", "description": "浙江省省会"}
    ]
    manager.add_entities(entities, source_chunk="chunk_001")

    # 添加关系
    relations = [
        {"source": "马云", "target": "阿里巴巴", "relation": "创立"},
        {"source": "阿里巴巴", "target": "杭州", "relation": "位于"}
    ]
    manager.add_relations(relations, source_chunk="chunk_001")

    # 查询实体
    entity = manager.get_entity("阿里巴巴")
    print(f"实体: {entity}")

    # 获取关系子图
    subgraph = manager.get_entity_relations("马云", direction="both", max_hops=2)
    print(f"子图: {subgraph}")

    # 统计信息
    stats = manager.get_statistics()
    print(f"统计: {stats}")

    # 关闭连接
    manager.close()