"""
答案生成模块
基于结构化上下文使用 LLM 生成答案
"""

import logging
import requests
import json
from typing import Optional, Dict
from graph_rag.config import QWEN_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnswerGenerator:
    """答案生成器"""

    # 系统指令模板
    SYSTEM_PROMPT = """你是一个专业的知识问答助手。你会收到以下信息：
1. 从知识图谱中检索到的相关实体和关系
2. 从文档库中检索到的相关文本片段

请基于这些信息回答用户的问题。

回答要求：
- 准确：答案必须基于提供的信息，不要编造
- 完整：充分利用实体、关系和文本信息
- 简洁：直接回答问题，避免冗余
- 有据：重要陈述需要引用来源
- 如果信息不足以回答问题，请明确说明

回答格式：
- 先给出直接答案
- 然后提供支撑细节
- 必要时说明信息来源或局限性"""

    def __init__(self,
                 api_key: str = None ,
                 api_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                 model: str = "qwen-plus"):
        """
        初始化答案生成器

        Args:
            api_key: API 密钥
            api_url: API 端点
            model: 模型名称
        """
        self.api_key = api_key or QWEN_API_KEY
        self.api_url = api_url
        self.model = model

    def generate(self,
                 context: str,
                 query: str,
                 temperature: float = 0.7,
                 max_tokens: int = 1000,
                 system_prompt: Optional[str] = None) -> Dict:
        """
        生成答案

        Args:
            context: 结构化上下文
            query: 用户查询
            temperature: 温度参数
            max_tokens: 最大 token 数
            system_prompt: 自定义系统指令

        Returns:
            {
                "answer": "生成的答案",
                "success": True/False,
                "error": "错误信息（如果有）"
            }
        """
        try:
            # 构建完整 prompt
            full_prompt = f"{context}\n\n问题：{query}\n\n请回答："

            # 调用 API
            answer = self._call_api(
                prompt=full_prompt,
                system_prompt=system_prompt or self.SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=max_tokens
            )

            logger.info(f"✓ 答案生成成功，长度: {len(answer)} 字符")

            return {
                "answer": answer,
                "success": True,
                "query": query
            }

        except Exception as e:
            logger.error(f"答案生成失败: {str(e)}")
            return {
                "answer": "",
                "success": False,
                "error": str(e),
                "query": query
            }

    def _call_api(self,
                  prompt: str,
                  system_prompt: str,
                  temperature: float,
                  max_tokens: int) -> str:
        """调用 LLM API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()
        return result['choices'][0]['message']['content'].strip()

    def generate_with_citation(self,
                               context: str,
                               query: str,
                               chunks: list) -> Dict:
        """
        生成带引用的答案

        Args:
            context: 结构化上下文
            query: 用户查询
            chunks: 原始文本块列表（用于引用）

        Returns:
            包含答案和引用信息的字典
        """
        # 修改系统指令，要求添加引用
        citation_prompt = self.SYSTEM_PROMPT + """

重要：在答案中引用信息来源时，使用 [文本片段N] 的格式标注。"""

        result = self.generate(
            context=context,
            query=query,
            system_prompt=citation_prompt
        )

        if result['success']:
            result['chunks'] = chunks  # 附带原始文本块

        return result

    def generate_streaming(self,
                           context: str,
                           query: str,
                           callback=None):
        """
        流式生成答案（逐字返回）

        Args:
            context: 结构化上下文
            query: 用户查询
            callback: 回调函数，接收每个生成的文本片段

        Returns:
            完整答案
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            full_prompt = f"{context}\n\n问题：{query}\n\n请回答："

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": full_prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1000,
                "stream": True  # 启用流式输出
            }
            print(payload)

            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=60
            )
            response.raise_for_status()

            full_answer = ""

            # 处理流式响应
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]  # 去掉 "data: " 前缀

                        if data_str.strip() == '[DONE]':
                            break

                        try:
                            data = json.loads(data_str)
                            delta = data['choices'][0]['delta']

                            if 'content' in delta:
                                chunk = delta['content']
                                full_answer += chunk

                                # 调用回调函数
                                if callback:
                                    callback(chunk)
                        except:
                            continue

            logger.info(f"✓ 流式生成完成，总长度: {len(full_answer)} 字符")

            return {
                "answer": full_answer,
                "success": True,
                "query": query
            }

        except Exception as e:
            logger.error(f"流式生成失败: {str(e)}")
            return {
                "answer": "",
                "success": False,
                "error": str(e),
                "query": query
            }

    def generate_summary(self, text: str, max_length: int = 200) -> str:
        """
        生成文本摘要

        Args:
            text: 原始文本
            max_length: 摘要最大长度

        Returns:
            摘要文本
        """
        prompt = f"""请为以下文本生成一个简洁的摘要（不超过{max_length}字）：

{text}

摘要："""

        try:
            summary = self._call_api(
                prompt=prompt,
                system_prompt="你是一个专业的文本摘要助手。",
                temperature=0.5,
                max_tokens=max_length * 2
            )
            return summary
        except Exception as e:
            logger.error(f"摘要生成失败: {str(e)}")
            return text[:max_length] + "..."

    def evaluate_answer_quality(self,
                                query: str,
                                answer: str,
                                context: str) -> Dict:
        """
        评估答案质量

        Args:
            query: 用户查询
            answer: 生成的答案
            context: 使用的上下文

        Returns:
            评估结果
        """
        eval_prompt = f"""请评估以下答案的质量：

问题：{query}

答案：{answer}

可用信息：{context[:500]}...

请从以下维度评分（1-5分）：
1. 准确性：答案是否基于提供的信息
2. 完整性：答案是否充分回答了问题
3. 相关性：答案是否切题
4. 清晰度：表达是否清晰易懂

以JSON格式返回评分和简短评语：
{{"accuracy": 分数, "completeness": 分数, "relevance": 分数, "clarity": 分数, "comment": "评语"}}"""

        try:
            result = self._call_api(
                prompt=eval_prompt,
                system_prompt="你是一个客观的答案质量评估专家。",
                temperature=0.3,
                max_tokens=500
            )

            # 提取 JSON
            result = result.strip()
            if result.startswith('```json'):
                result = result[7:]
            if result.startswith('```'):
                result = result[3:]
            if result.endswith('```'):
                result = result[:-3]

            evaluation = json.loads(result.strip())
            return evaluation

        except Exception as e:
            logger.error(f"答案评估失败: {str(e)}")
            return {}


# 使用示例
if __name__ == "__main__":
    generator = AnswerGenerator()

    # 示例上下文
    context = """
【相关实体】
1. 阿里巴巴 (组织)
   描述：中国最大的电子商务公司
2. 马云 (人物)
   描述：阿里巴巴创始人

【实体关系】
• 马云：
  - [创立] → 阿里巴巴
  - [出生于] → 杭州

【参考文本】
文本片段 1:
马云于1999年在杭州创立了阿里巴巴集团。阿里巴巴是中国最大的电子商务公司之一。
"""

    query = "阿里巴巴的创始人是谁？"

    # 生成答案
    result = generator.generate(context, query)

    if result['success']:
        print(f"问题: {query}")
        print(f"\n答案:\n{result['answer']}")
    else:
        print(f"生成失败: {result['error']}")

    # 流式生成示例
    print("\n\n流式生成:")


    def print_chunk(chunk):
        print(chunk, end='', flush=True)


    stream_result = generator.generate_streaming(context, query, callback=print_chunk)
    print("\n")