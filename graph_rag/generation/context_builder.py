"""
上下文构建模块
将检索到的实体、关系和文本块构建成结构化上下文
"""

import logging
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextBuilder:
    """上下文构建器"""

    def __init__(self, max_entities: int = 20, max_relations: int = 30, max_chunks: int = 5):
        """
        初始化上下文构建器

        Args:
            max_entities: 最大实体数量
            max_relations: 最大关系数量
            max_chunks: 最大文本块数量
        """
        self.max_entities = max_entities
        self.max_relations = max_relations
        self.max_chunks = max_chunks

    def build_context(self,
                      entities: List[Dict],
                      relations: List[Dict],
                      chunks: List[Dict],
                      query: str = "") -> str:
        """
        构建完整上下文

        Args:
            entities: 实体列表 [{"name": "", "type": "", "description": ""}]
            relations: 关系列表 [{"source": "", "target": "", "relation": ""}]
            chunks: 文本块列表 [{"chunk_id": "", "text": ""}]
            query: 用户查询

        Returns:
            格式化的上下文字符串
        """
        # 限制数量
        entities = entities[:self.max_entities]
        relations = relations[:self.max_relations]
        chunks = chunks[:self.max_chunks]

        context_parts = []

        # 1. 查询信息
        if query:
            context_parts.append(f"用户问题：{query}\n")

        # 2. 实体信息
        if entities:
            context_parts.append("=" * 60)
            context_parts.append("【相关实体】")
            context_parts.append("=" * 60)

            for i, entity in enumerate(entities, 1):
                name = entity.get('name', '')
                entity_type = entity.get('type', '')
                description = entity.get('description', '')

                entity_info = f"{i}. {name}"
                if entity_type:
                    entity_info += f" ({entity_type})"
                if description:
                    entity_info += f"\n   描述：{description}"

                context_parts.append(entity_info)

            context_parts.append("")

        # 3. 关系信息
        if relations:
            context_parts.append("=" * 60)
            context_parts.append("【实体关系】")
            context_parts.append("=" * 60)

            # 按源实体分组
            relation_groups = {}
            for rel in relations:
                source = rel.get('source', '')
                if source not in relation_groups:
                    relation_groups[source] = []
                relation_groups[source].append(rel)

            for source, rels in relation_groups.items():
                context_parts.append(f"\n• {source}：")
                for rel in rels:
                    target = rel.get('target', '')
                    relation = rel.get('relation', '')
                    context_parts.append(f"  - [{relation}] → {target}")

            context_parts.append("")

        # 4. 原始文本块
        if chunks:
            context_parts.append("=" * 60)
            context_parts.append("【参考文本】")
            context_parts.append("=" * 60)

            for i, chunk in enumerate(chunks, 1):
                text = chunk.get('text', '')
                source_file = chunk.get('source_file', '')

                chunk_header = f"\n文本片段 {i}"
                if source_file:
                    chunk_header += f" (来源: {source_file})"
                chunk_header += "："

                context_parts.append(chunk_header)
                context_parts.append(text)
                context_parts.append("")

        context = "\n".join(context_parts)

        return context

    def build_graph_context(self, entities: List[Dict], relations: List[Dict]) -> str:
        """
        构建图结构上下文（只包含实体和关系）

        Args:
            entities: 实体列表
            relations: 关系列表

        Returns:
            图结构上下文
        """
        entities = entities[:self.max_entities]
        relations = relations[:self.max_relations]

        context_parts = []

        # 实体部分
        if entities:
            context_parts.append("【知识图谱中的实体】")
            for entity in entities:
                name = entity.get('name', '')
                entity_type = entity.get('type', '')
                description = entity.get('description', '')

                entity_str = f"- {name} ({entity_type})"
                if description:
                    entity_str += f": {description}"
                context_parts.append(entity_str)
            context_parts.append("")

        # 关系部分
        if relations:
            context_parts.append("【实体之间的关系】")
            for rel in relations:
                source = rel.get('source', '')
                target = rel.get('target', '')
                relation = rel.get('relation', '')
                context_parts.append(f"- {source} --[{relation}]--> {target}")
            context_parts.append("")

        return "\n".join(context_parts)

    def build_text_context(self, chunks: List[Dict]) -> str:
        """
        构建文本上下文（只包含文本块）

        Args:
            chunks: 文本块列表

        Returns:
            文本上下文
        """
        chunks = chunks[:self.max_chunks]

        context_parts = ["【相关文档片段】\n"]

        for i, chunk in enumerate(chunks, 1):
            text = chunk.get('text', '')
            source_file = chunk.get('source_file', '')

            if source_file:
                context_parts.append(f"片段 {i} (来源: {source_file}):")
            else:
                context_parts.append(f"片段 {i}:")

            context_parts.append(text)
            context_parts.append("")

        return "\n".join(context_parts)

    def build_compact_context(self,
                              entities: List[Dict],
                              relations: List[Dict],
                              chunks: List[Dict]) -> str:
        """
        构建紧凑版上下文（适合 token 限制严格的场景）

        Args:
            entities: 实体列表
            relations: 关系列表
            chunks: 文本块列表

        Returns:
            紧凑版上下文
        """
        context_parts = []

        # 实体（只保留名称和类型）
        if entities:
            entity_strs = [f"{e['name']}({e['type']})" for e in entities[:10]]
            context_parts.append(f"实体: {', '.join(entity_strs)}")

        # 关系（简化表示）
        if relations:
            relation_strs = [f"{r['source']}-{r['relation']}-{r['target']}"
                             for r in relations[:15]]
            context_parts.append(f"关系: {'; '.join(relation_strs)}")

        # 文本块（截断）
        if chunks:
            context_parts.append("\n参考文本:")
            for i, chunk in enumerate(chunks[:3], 1):
                text = chunk.get('text', '')
                # 截断到200字符
                text_preview = text[:200] + "..." if len(text) > 200 else text
                context_parts.append(f"{i}. {text_preview}")

        return "\n".join(context_parts)

    def format_for_prompt(self,
                          context: str,
                          query: str,
                          instruction: str = "") -> str:
        """
        将上下文格式化为适合 LLM 的 Prompt

        Args:
            context: 构建好的上下文
            query: 用户查询
            instruction: 额外指令

        Returns:
            完整的 Prompt
        """
        prompt_parts = []

        # 系统指令
        if instruction:
            prompt_parts.append(instruction)
            prompt_parts.append("")

        # 上下文
        prompt_parts.append(context)
        prompt_parts.append("")

        # 查询
        prompt_parts.append("=" * 60)
        prompt_parts.append(f"问题：{query}")
        prompt_parts.append("=" * 60)
        prompt_parts.append("")
        prompt_parts.append("请基于以上信息回答问题：")

        return "\n".join(prompt_parts)

    def get_context_stats(self, context: str) -> Dict:
        """获取上下文统计信息"""
        return {
            "total_length": len(context),
            "line_count": context.count('\n'),
            "approximate_tokens": len(context) // 2  # 粗略估计（中文约2字符=1token）
        }


# 使用示例
if __name__ == "__main__":
    builder = ContextBuilder()

    # 示例数据
    entities = [
        {"name": "阿里巴巴", "type": "组织", "description": "中国最大的电子商务公司"},
        {"name": "马云", "type": "人物", "description": "阿里巴巴创始人"},
        {"name": "杭州", "type": "地点", "description": "浙江省省会"}
    ]

    relations = [
        {"source": "马云", "target": "阿里巴巴", "relation": "创立"},
        {"source": "阿里巴巴", "target": "杭州", "relation": "位于"},
        {"source": "马云", "target": "杭州", "relation": "出生于"}
    ]

    chunks = [
        {
            "chunk_id": "chunk_001",
            "text": "马云于1999年在杭州创立了阿里巴巴集团。",
            "source_file": "doc1.txt"
        }
    ]

    # 构建完整上下文
    context = builder.build_context(
        entities=entities,
        relations=relations,
        chunks=chunks,
        query="阿里巴巴的创始人是谁？"
    )
    print(context)
    print("\n" + "=" * 60)

    # 构建紧凑版
    compact = builder.build_compact_context(entities, relations, chunks)
    print("\n紧凑版上下文：")
    print(compact)

    # 统计信息
    stats = builder.get_context_stats(context)
    print(f"\n上下文统计: {stats}")