from llm_service import LLMService

import json
import re
import asyncio
from typing import List, Dict, Any


class ExtractorAgent:
    def __init__(self, service: LLMService, max_concurrency: int = 5):
        self.service = service
        # 限制并发数
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.system_prompt = (
            "你是一个多模态知识图谱专家。你的任务是从文本中提取关键语义实体。\n"
            "提取要求：\n"
            "1. 识别具有检索价值的实体，包括：Building, Location, Color, Biological, Attribute, Event。\n"
            "2. 严禁返回任何 Markdown 代码块标签（如 ```json），只返回纯 JSON 字符串。\n"
            "3. 格式必须为对象列表：[{\"name\": \"...\", \"type\": \"...\"}]。"
        )

    def _parse_json(self, raw_response: str) -> List[Dict]:
        """健壮的 JSON 解析逻辑"""
        try:
            # 去除 markdown 标签及前后空格
            json_str = re.sub(r"```json|```", "", raw_response).strip()
            # 找到第一个 '[' 和最后一个 ']'
            start = json_str.find('[')
            end = json_str.rfind(']') + 1
            if start != -1 and end != 0:
                json_str = json_str[start:end]
            return json.loads(json_str)
        except Exception:
            raise ValueError("Invalid JSON format from LLM")

    async def extract_single(self, raw_array: list, retries: int = 2) -> dict:
        """抽取单个条目，带重试逻辑"""
        doc_id, text, img_ids = raw_array[0], raw_array[1], raw_array[2]
        clean_text = text.replace("<PIC>", "").strip()

        async with self.semaphore:
            for attempt in range(retries + 1):
                try:
                    # 注意：此处假设 LLMService.ask 已被封装为异步方法，
                    # 如果是同步方法，请使用 asyncio.to_thread(self.service.ask, ...)
                    raw_response = await asyncio.to_thread(self.service.ask, {
                        "system": self.system_prompt,
                        "query": f"请提取以下文本中的实体：\n{clean_text}"
                    })

                    entities = self._parse_json(raw_response)

                    return {
                        "chunk": {"id": f"{doc_id}_0", "text": text},
                        "entities": entities,
                        "images": [{"id": i, "path": f"MRAMG-Bench/IMAGE/images/WEB/{i}.jpg"} for i in img_ids]
                    }
                except Exception as e:
                    if attempt < retries:
                        print(f"[*] 正在重试 ({attempt + 1}/{retries}) - DocID: {doc_id}")
                        await asyncio.sleep(1)  # 避让一下
                    else:
                        print(f"[!] 最终抽取失败 - DocID: {doc_id}: {e}")
                        return {
                            "chunk": {"id": f"{doc_id}_0", "text": text},
                            "entities": [],
                            "images": [{"id": i, "path": f"MRAMG-Bench/IMAGE/images/WEB/{i}.jpg"} for i in img_ids]
                        }

    async def batch_extract(self, data_list: List[list]) -> List[dict]:
        """批量抽取入口"""
        tasks = [self.extract_single(item) for item in data_list]
        return await asyncio.gather(*tasks)


class QueryAgent:
    """问题理解 Agent：将自然语言问题转换为结构化意图（实体和关系）"""
    
    def __init__(self, service: LLMService):
        self.service = service
        self.system_prompt = (
            "你是一个知识图谱问题理解专家。你的任务是从用户问题中提取关键信息。\n"
            "提取要求：\n"
            "1. 识别问题中提到的实体（如建筑名称、地点、颜色等）。\n"
            "2. 识别问题关注的属性或关系（如颜色、位置、类型等）。\n"
            "3. 返回 JSON 格式：{\"entities\": [\"实体1\", \"实体2\"], \"relations\": [\"关系1\", \"关系2\"]}\n"
            "4. 严禁返回任何 Markdown 代码块标签，只返回纯 JSON 字符串。\n"
            "示例：\n"
            "问题：\"National Museum of the American Indian 和 Xanadu House 颜色一样吗？\"\n"
            "输出：{\"entities\": [\"National Museum of the American Indian\", \"Xanadu House\"], \"relations\": [\"color\"]}"
        )
    
    def _parse_json(self, raw_response: str) -> Dict[str, List[str]]:
        """健壮的 JSON 解析逻辑"""
        try:
            # 去除 markdown 标签及前后空格
            json_str = re.sub(r"```json|```", "", raw_response).strip()
            # 找到第一个 '{' 和最后一个 '}'
            start = json_str.find('{')
            end = json_str.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = json_str[start:end]
            return json.loads(json_str)
        except Exception:
            raise ValueError("Invalid JSON format from LLM")
    
    def parse(self, question: str) -> Dict[str, List[str]]:
        """解析问题，提取实体和关系"""
        try:
            raw_response = self.service.ask({
                "system": self.system_prompt,
                "query": f"请分析以下问题并提取实体和关系：\n{question}"
            })
            
            result = self._parse_json(raw_response)
            return {
                "entities": result.get("entities", []),
                "relations": result.get("relations", [])
            }
        except Exception as e:
            print(f"  ⚠️  问题解析失败: {e}")
            # 返回默认值
            return {"entities": [], "relations": []}


class CypherAgent:
    def __init__(self, service: LLMService):
        self.service = service
        self.system_prompt = (
            "你是一个 Neo4j Cypher 专家。请根据用户问题生成查询语句。\n"
            "图谱 Schema 如下：\n"
            "- (:Chunk {id, text}) 存储原始文本块\n"
            "- (:Entity {name, type}) 存储提取的实体\n"
            "- (:Image {id, path}) 存储图片路径\n"
            "关系：\n"
            "- (Entity)-[:MENTIONED_IN]->(Chunk)\n"
            "- (Chunk)-[:HAS_IMAGE]->(Image)\n\n"
            "要求：\n"
            "1. 只返回 Cypher 语句，不带 Markdown 格式，不要解释。\n"
            "2. 优先考虑多跳查询。若问题涉及比较，寻找连接两个不同 Chunk 的共同 Entity。\n"
            "3. 结果必须 RETURN i.path 以便展示图片。"
        )

    def generate(self, question: str) -> str:
        return self.service.ask({
            "system": self.system_prompt,
            "query": f"将此问题转化为 Cypher：{question}"
        }).replace("```cypher", "").replace("```", "").strip()


class Synthesizer:
    def __init__(self, service: LLMService):
        self.service = service
        self.system_prompt = (
            "你是一个具备多模态视觉理解能力的 AI 助手。\n"
            "你会收到两部分证据：\n"
            "1. [Vector Context]: 语义相关的原始文本片段。\n"
            "2. [Graph Context]: 经过图数据库推理得到的结构化事实及关联图片路径。\n\n"
            "要求：\n"
            "1. 综合证据回答用户问题。若两者颜色、数量等属性一致，需明确指出。\n"
            "2. 回答中必须引用图片路径，格式固定为：[IMAGE: path/to/image.jpg]。\n"
            "3. 保持专业、简洁，不要提及'根据证据显示'等废话。"
        )

    def finalize(self, question: str, v_ctx: list, g_ctx: list) -> str:
        query = f"问题: {question}\n\n向量库证据: {v_ctx}\n\n图数据库证据: {g_ctx}"
        return self.service.ask({"system": self.system_prompt, "query": query})
 

class ChatAgent:
    def __init__(self, service: LLMService):
        self.service = service
        # 优化后的 System Prompt：强调证据融合与图片回溯
        self.system_prompt = (
            "你是一个多模态知识图谱助手，擅长结合文本证据和视觉证据回答问题。\n"
            "你会收到两类背景资料：\n"
            "1. [向量检索文本]: 包含原始的描述段落。\n"
            "2. [图谱推理事实]: 包含实体关系、属性以及图片路径。\n\n"
            "回答准则：\n"
            "- 如果图谱证据中包含图片路径，必须以 `[IMAGE: 路径]` 格式在回答中引用。\n"
            "- 优先通过图谱中的共享实体（如相同的颜色、学名、地点）来回答比较类问题。\n"
            "- 如果证据之间存在冲突，请以图谱推理事实为准，并指出冲突点。\n"
            "- 严禁编造图片路径，只能使用背景资料中提供的 'MRAMG-Bench/...' 路径。"
        )

    def ask(self, query: str, vector_context: list, graph_context: list) -> str:
        """
        合成最终答案
        :param query: 用户的问题
        :param vector_context: 来自 Chroma 的 documents
        :param graph_context: 来自 Neo4j 的查询结果（含图片路径）
        """
        # 构造给 LLM 的上下文
        user_input = {
            "system": self.system_prompt,
            "query": (
                f"用户问题: {query}\n\n"
                f"--- 背景资料 ---\n"
                f"[向量检索文本]: {vector_context}\n\n"
                f"[图谱推理事实]: {graph_context}\n"
                f"----------------\n"
                f"请结合以上证据给出结论。"
            )
        }

        # 调用 LLMService 的 ask 方法
        return self.service.ask(user_input)


if __name__ == "__main__":
    # 你的 LLM 服务初始化
    service = LLMService(model_name="deepseek-chat")
    agent = ExtractorAgent(service, max_concurrency=10)  # 允许同时 10 个请求

    # 模拟从 jsonl 读取的数据
    raw_data_batch = [
        [10000,
         "The National Museum of the American Indian in Washington, D.C., opened in 2004 on the Mall, embodies a harmonious fusion of cultural respect and architectural beauty. Its curvilinear structure, adorned with a smooth, beige-colored exterior, exudes an organic and natural feel that resonates with the landscapes it reflects. The building's flowing design draws inspiration from natural forms, blending seamlessly with its surroundings. Under a warm-hued sky, the museum is elegantly illuminated, with soft lighting accentuating the textures and contours of its facade. This institution stands as a testament to the cultural significance of American Indians, who are comfortable with terms like Indian, American Indian, and Native American\u2014reflected in the museum's name.<PIC>",
         [30321533]],
        [
            10001,
            "The Xanadu House in Kissimmee, Florida, built in 1985, showcases a unique and futuristic architectural design with a distinctive beige exterior. Known for its bulbous, organic shapes that seamlessly blend with the surrounding natural elements, the house reflects in a serene pond and is enveloped by lush trees, creating a harmonious integration of nature and innovation. This architectural marvel was ahead of its time, incorporating an automated system managed by Commodore microcomputers. Within its fifteen rooms, spaces like the kitchen, party room, health spa, and bedrooms were heavily equipped with computers and electronic gadgets, emphasizing advanced technology in their design.<PIC>",
            [
                30278153]]

    ]

    # 运行异步批量抽取
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(agent.batch_extract(raw_data_batch))

    for res in results:
        print(f"ID: {res['chunk']['id']} | Entities: {res['entities']}")
