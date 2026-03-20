"""
实体关系抽取模块
输入文本，输出实体和三元组
"""

import json
import logging
import re
from typing import Dict, List
from graph_rag.utils.qwen_client import QwenClient, QwenAPIError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntityExtractor:
    """使用 LLM 进行实体关系抽取"""

    # 系统提示词 - 强制 JSON 输出
    SYSTEM_PROMPT = """你是一个专业的知识图谱构建助手。你的任务是从文本中提取实体和关系。

**重要规则：**
1. 你必须只返回有效的 JSON 格式，不要有任何额外文字、解释或 markdown 标记
2. 不要使用 ```json 或 ``` 标记
3. 直接返回 JSON 对象，以 { 开头，以 } 结尾

**JSON 格式：**
{
  "entities": [
    {"name": "实体名称", "type": "实体类型", "description": "简短描述"}
  ],
  "relations": [
    {"source": "源实体", "target": "目标实体", "relation": "关系类型", "description": "关系描述"}
  ]
}

记住：只返回 JSON，不要有其他内容！"""

    # 用户提示词
    USER_PROMPT = """从以下文本中提取实体和关系：

{text}

提取要求：
1. 实体类型：人物、组织、地点、事件、概念、产品、时间等
2. 为每个实体生成 1-2 句简短描述
3. 关系类型自由定义（如：创立、任职、位于、拥有、参与等）
4. 为重要关系添加简短描述

直接返回 JSON："""

    def __init__(self):
        """初始化实体抽取器"""
        self.client = QwenClient()
        logger.info("✓ 实体抽取器已初始化")

    def extract(self, text: str) -> Dict:
        """
        抽取实体和关系

        Args:
            text: 输入文本

        Returns:
            {
                "entities": [{"name": "", "type": "", "description": ""}],
                "relations": [{"source": "", "target": "", "relation": "", "description": ""}]
            }
        """
        if not text.strip():
            return {"entities": [], "relations": []}

        try:
            # 构建消息
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": self.USER_PROMPT.format(text=text)}
            ]

            # 调用 API
            response = self.client.chat(
                messages=messages,
            )

            # 提取内容
            response_text = self.client.extract_content(response)


            # 解析 JSON
            result = self._extract_json(response_text)

            # 清洗数据
            entities = self._clean_entities(result.get('entities', []))
            relations = self._clean_relations(result.get('relations', []), entities)

            logger.info(f"✓ 提取: {len(entities)} 实体, {len(relations)} 关系")

            return {
                "entities": entities,
                "relations": relations
            }

        except QwenAPIError as e:
            logger.error(f"✗ API 调用失败: {str(e)}")
            return {"entities": [], "relations": [], "error": str(e)}

        except Exception as e:
            logger.error(f"✗ 抽取失败: {str(e)}")
            return {"entities": [], "relations": [], "error": str(e)}

    def _extract_json(self, text: str) -> dict:
        """从响应中提取 JSON"""
        # 清理可能的 markdown 标记
        text = text.strip()
        text = re.sub(r'^```json\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^```\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()

        # 方法1：直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 方法2：提取第一个 { 到最后一个 }
        try:
            start = text.index('{')
            end = text.rindex('}') + 1
            json_str = text[start:end]
            return json.loads(json_str)
        except (ValueError, json.JSONDecodeError):
            pass

        # 方法3：使用正则提取
        try:
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                return json.loads(match.group())
        except json.JSONDecodeError:
            pass

        logger.error(f"✗ 无法解析 JSON，响应内容: {text[:200]}...")
        return {"entities": [], "relations": []}

    def _clean_entities(self, entities: list) -> list:
        """清洗实体数据"""
        clean = {}
        for e in entities:
            name = e.get('name', '').strip()
            if name:
                clean[name] = {
                    'name': name,
                    'type': e.get('type', '其他').strip(),
                    'description': e.get('description', '').strip()
                }
        return list(clean.values())

    def _clean_relations(self, relations: list, entities: list) -> list:
        """清洗关系数据"""
        entity_names = {e['name'] for e in entities}
        clean = []
        seen = set()

        for r in relations:
            source = r.get('source', '').strip()
            target = r.get('target', '').strip()
            relation = r.get('relation', '').strip()
            description = r.get('description', '').strip()

            # 验证：实体存在、无自环、无重复
            if (source and target and relation and
                source in entity_names and target in entity_names and
                source != target):

                key = (source, target, relation)
                if key not in seen:
                    seen.add(key)
                    clean.append({
                        'source': source,
                        'target': target,
                        'relation': relation,
                        'description': description
                    })

        return clean

    def extract_batch(self, texts: List[str]) -> List[Dict]:
        """
        批量抽取

        Args:
            texts: 文本列表

        Returns:
            抽取结果列表
        """
        results = []
        total = len(texts)

        for i, text in enumerate(texts, 1):
            logger.info(f"进度: {i}/{total}")
            result = self.extract(text)
            results.append(result)

        return results

    def print_result(self, result: Dict, show_description: bool = True):
        """
        打印单个抽取结果

        Args:
            result: extract() 返回的结果
            show_description: 是否显示描述信息
        """
        print("\n" + "=" * 80)

        # 检查错误
        if 'error' in result:
            print(f"✗ 抽取失败: {result['error']}")
            return

        # 打印实体
        entities = result.get('entities', [])
        print(f"【实体】 共 {len(entities)} 个")
        print("-" * 80)

        if entities:
            for i, e in enumerate(entities, 1):
                print(f"{i}. {e['name']} ({e['type']})")
                if show_description and e.get('description'):
                    print(f"   描述: {e['description']}")
        else:
            print("  (未提取到实体)")

        # 打印关系
        relations = result.get('relations', [])
        print(f"\n【关系】 共 {len(relations)} 个")
        print("-" * 80)

        if relations:
            for i, r in enumerate(relations, 1):
                print(f"{i}. {r['source']} --[{r['relation']}]--> {r['target']}")
                if show_description and r.get('description'):
                    print(f"   描述: {r['description']}")
        else:
            print("  (未提取到关系)")

        print("=" * 80)

    def print_batch_results(self, results: List[Dict], show_description: bool = False):
        """
        打印批量抽取结果

        Args:
            results: extract_batch() 返回的结果列表
            show_description: 是否显示描述信息
        """
        print("\n" + "=" * 80)
        print(f"批量抽取结果统计")
        print("=" * 80)

        total_entities = 0
        total_relations = 0
        failed_count = 0

        for i, result in enumerate(results, 1):
            if 'error' in result:
                failed_count += 1
                print(f"\n文本 {i}: ✗ 失败 - {result['error']}")
            else:
                entity_count = len(result.get('entities', []))
                relation_count = len(result.get('relations', []))
                total_entities += entity_count
                total_relations += relation_count

                print(f"\n文本 {i}: ✓ {entity_count} 实体, {relation_count} 关系")

                # 可选：显示详细内容
                if show_description:
                    self.print_result(result, show_description=True)

        print("\n" + "=" * 80)
        print("总计统计")
        print("-" * 80)
        print(f"处理文本: {len(results)} 个")
        print(f"失败数量: {failed_count} 个")
        print(f"提取实体: {total_entities} 个")
        print(f"提取关系: {total_relations} 个")
        print("=" * 80)


# 使用示例
if __name__ == "__main__":
    # 初始化（从配置读取 API Key）
    extractor = EntityExtractor()

    # 示例文本
    text = """马云于1999年在杭州创立了阿里巴巴集团。阿里巴巴是中国最大的电子商务公司，
    旗下有淘宝网和天猫。2014年阿里巴巴在纽约证券交易所上市，创下当时最大的IPO记录。
    马云担任阿里巴巴董事局主席多年，直到2019年卸任。"""

    # 单个抽取
    print("\n" + "=" * 80)
    print("示例1: 单个文本抽取")
    print("=" * 80)

    result = extractor.extract(text)
    extractor.print_result(result, show_description=True)

    # # 批量抽取
    # print("\n" + "=" * 80)
    # print("示例2: 批量文本抽取")
    # print("=" * 80)
    #
    # texts = [
    #     "苹果公司由史蒂夫·乔布斯于1976年创立。",
    #     "北京是中国的首都，位于华北平原北部。",
    #     "深度学习是机器学习的一个分支，使用神经网络进行学习。"
    # ]
    #
    # results = extractor.extract_batch(texts)
    # extractor.print_batch_results(results, show_description=False)