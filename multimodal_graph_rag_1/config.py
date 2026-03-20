"""配置管理模块"""
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """系统配置"""
    
    # Neo4j 配置
    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", env="NEO4J_USER")
    neo4j_password: str = Field(default="password", env="NEO4J_PASSWORD")
    
    # ChromaDB 配置
    chromadb_path: str = Field(default="./data/chromadb", env="CHROMADB_PATH")
    
    # 数据集路径
    dataset_path: str = Field(default="./data/mramg_bench", env="DATASET_PATH")
    docs_jsonl: str = Field(default="./data/mramg_bench/docs.jsonl", env="DOCS_JSONL")
    qa_jsonl: str = Field(default="./data/mramg_bench/qa.jsonl", env="QA_JSONL")
    images_path: str = Field(default="./data/mramg_bench/images", env="IMAGES_PATH")
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    # 检索配置
    vector_top_k: int = Field(default=5, env="VECTOR_TOP_K")
    graph_max_hops: int = Field(default=2, env="GRAPH_MAX_HOPS")
    
    # CLIP 模型
    clip_model_name: str = Field(default="ViT-B/32", env="CLIP_MODEL_NAME")
    
    # 输出路径
    output_path: str = Field(default="./output", env="OUTPUT_PATH")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()


def get_settings() -> Settings:
    """获取配置实例"""
    return settings
