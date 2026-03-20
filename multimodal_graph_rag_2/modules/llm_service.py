import os
from typing import Union, List, Dict, Optional, Generator
from dotenv import load_dotenv

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    BaseMessage,
)

from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatTongyi

load_dotenv()

# -----------------------------
# Model → Provider 映射规则
# -----------------------------
MODEL_PROVIDER_MAP = {
    "deepseek": ["deepseek"],
    "openai": ["gpt", "o1"],
    "claude": ["claude"],
    "qwen": ["qwen", "tongyi"],
    "doubao": ["doubao"],
}


class LLMService:
    def __init__(
            self,
            model_name: str,
            provider: Optional[str] = None,
            temperature: float = 0.0,
            streaming: bool = False,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.streaming = streaming

        # 自动识别 provider（除非显式指定）
        self.provider = provider or self._infer_provider(model_name)
        self.llm = self._build_llm()

    # -----------------------------
    # Provider 自动识别
    # -----------------------------
    def _infer_provider(self, model_name: str) -> str:
        name = model_name.lower()
        for provider, keywords in MODEL_PROVIDER_MAP.items():
            if any(k in name for k in keywords):
                return provider
        raise ValueError(f"Cannot infer provider from model name: {model_name}")

    # -----------------------------
    # 构造 LLM
    # -----------------------------
    def _build_llm(self):
        if self.provider == "deepseek":
            return ChatDeepSeek(
                model=self.model_name,
                temperature=self.temperature,
                streaming=self.streaming,
            )

        elif self.provider == "openai":
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                streaming=self.streaming,
            )

        elif self.provider == "claude":
            return ChatAnthropic(
                model_name=self.model_name,
                temperature=self.temperature,
                streaming=self.streaming,
            )

        elif self.provider == "qwen":
            return ChatTongyi(
                model=self.model_name,
                streaming=self.streaming,
            )

        elif self.provider == "doubao":
            return ChatOpenAI(
                model=self.model_name,
                base_url=os.getenv("DOUBAO_BASE_URL"),
                api_key=os.getenv("DOUBAO_API_KEY"),
                temperature=self.temperature,
                streaming=self.streaming,
            )

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    # -----------------------------
    # 消息构造
    # -----------------------------
    def _build_messages(
            self,
            input_data: Union[str, Dict[str, str], List[BaseMessage]]
    ) -> List[BaseMessage]:
        if isinstance(input_data, str):
            return [HumanMessage(content=input_data)]

        if isinstance(input_data, dict):
            messages = []
            if "system" in input_data:
                messages.append(SystemMessage(content=input_data["system"]))
            if "query" in input_data:
                messages.append(HumanMessage(content=input_data["query"]))
            return messages

        if isinstance(input_data, list):
            return input_data

        raise TypeError("Unsupported input format for ask()")

    # -----------------------------
    # 非流式调用
    # -----------------------------
    def ask(self, input_data: Union[str, Dict[str, str], List[BaseMessage]]) -> str:
        messages = self._build_messages(input_data)
        response = self.llm.invoke(messages)
        return response.content

    # -----------------------------
    # 流式调用
    # -----------------------------
    def stream(
            self,
            input_data: Union[str, Dict[str, str], List[BaseMessage]]
    ) -> Generator[str, None, None]:
        if not self.streaming:
            raise RuntimeError("streaming=False, cannot call stream()")

        messages = self._build_messages(input_data)
        for chunk in self.llm.stream(messages):
            if chunk.content:
                yield chunk.content


# -----------------------------
# 测试（只用 DeepSeek）
# -----------------------------
if __name__ == "__main__":
    llm = LLMService(
        model_name="deepseek-chat",
        streaming=True,
    )

    prompt = {
        "system": "你是一个知识图谱与 GraphRAG 专家",
        "query": "用大白话解释什么是 Cypher 查询语言",
    }

    print("🚀 Streaming output:\n")
    for token in llm.stream(prompt):
        print(token, end="", flush=True)
