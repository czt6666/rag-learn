"""
Graph RAG 配置文件
支持从环境变量读取配置，如果没有则使用默认值
"""

import os

# LLM 配置
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "sk-3d9e9d96bf9c48ce887ecaaa4659152e")

# Neo4j 配置
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")
NEO4J_ENTITY_LABEL = os.getenv("NEO4J_ENTITY_LABEL", "Entity")  # 实体标签，可用于区分不同的数据集

# Chroma 配置
CHROMA_ENTITY_COLLECTION = os.getenv("CHROMA_ENTITY_COLLECTION", "entities")  # 实体集合名称
CHROMA_CHUNK_COLLECTION = os.getenv("CHROMA_CHUNK_COLLECTION", "chunks")  # 文本块集合名称