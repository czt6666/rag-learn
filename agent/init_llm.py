import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# 加载环境变量
load_dotenv(override=True)

# 从 .env 中读取各个模型的配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL")

# 初始化 DeepSeek 模型
deepseek_llm = init_chat_model(
    model_provider="deepseek",
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

openai_llm = init_chat_model(
    model_provider="openai",
    model="gpt-4.1-mini",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

qwen_llm = init_chat_model(
    model_provider="openai",
    model="qwen3.5-plus",
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
)

# print(qwen_llm.invoke("你好"))
