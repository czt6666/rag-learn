"""
通义千问 API 客户端
处理 API 调用、错误处理、重试逻辑
"""

import time
import logging
import requests
from typing import Optional, Dict, List
from graph_rag.config import QWEN_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================== 异常定义 ==================

class QwenAPIError(Exception):
    """Qwen API 错误基类"""
    pass


class QwenBalanceError(QwenAPIError):
    """余额不足错误"""
    pass


class QwenRateLimitError(QwenAPIError):
    """并发限流错误"""
    pass


class QwenInvalidRequestError(QwenAPIError):
    """请求参数错误"""
    pass


# ================== 客户端 ==================

class QwenClient:
    """通义千问 API 客户端"""

    # 默认配置（写死在客户端，简单直观）
    DEFAULT_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    DEFAULT_MODEL = "qwen-plus"

    MAX_RETRIES = 3
    RETRY_DELAY = 2

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or QWEN_API_KEY
        self.api_url = api_url or self.DEFAULT_API_URL
        self.model = model or self.DEFAULT_MODEL

        if not self.api_key:
            raise ValueError("API Key 未设置")

        logger.info(f"✓ Qwen 客户端初始化: {self.model}")

    # ================== 主调用 ==================

    def chat(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }

        for attempt in range(self.MAX_RETRIES):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                )
                if response.status_code == 200:
                    return response.json()

                self._handle_error(response, attempt)

            except requests.exceptions.Timeout:
                logger.warning(f"请求超时 ({attempt + 1}/{self.MAX_RETRIES})")
                self._retry_or_raise(attempt, "API 请求超时")

            except requests.exceptions.ConnectionError:
                logger.warning(f"连接失败 ({attempt + 1}/{self.MAX_RETRIES})")
                self._retry_or_raise(attempt, "网络连接失败")

            except Exception as e:
                logger.error(f"未知错误: {e}")
                raise QwenAPIError(str(e))

        raise QwenAPIError("API 调用失败，已达最大重试次数")

    # ================== 错误处理 ==================

    def _handle_error(self, response: requests.Response, attempt: int):
        try:
            data = response.json()
            message = data.get("message", "")
        except Exception:
            message = response.text

        message_lower = message.lower()

        if "balance" in message_lower or "insufficient" in message_lower:
            raise QwenBalanceError("API 余额不足")

        if response.status_code == 429 or "rate" in message_lower or "throttl" in message_lower:
            self._retry_or_raise(attempt, "并发限流", QwenRateLimitError)

        if response.status_code == 400:
            raise QwenInvalidRequestError(f"请求参数错误: {message}")

        if response.status_code == 401:
            raise QwenAPIError("API Key 无效或已过期")

        self._retry_or_raise(attempt, f"API 错误: {message}")

    def _retry_or_raise(self, attempt: int, msg: str, exc=QwenAPIError):
        if attempt < self.MAX_RETRIES - 1:
            time.sleep(self.RETRY_DELAY * (attempt + 1))
        else:
            raise exc(msg)

    # ================== 响应解析 ==================

    def extract_content(self, response: Dict) -> str:
        try:
            return response["choices"][0]["message"]["content"].strip()
        except Exception:
            raise QwenAPIError(f"无法解析响应: {response}")


# ================== 使用示例 ==================

if __name__ == "__main__":
    client = QwenClient()

    messages = [{"role": "user", "content": "你好，请介绍一下自己"}]

    try:
        resp = client.chat(messages=messages)
        print(client.extract_content(resp))

    except QwenBalanceError as e:
        print("余额不足:", e)
    except QwenRateLimitError as e:
        print("并发限流:", e)
    except QwenAPIError as e:
        print("API 错误:", e)
