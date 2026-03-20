import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

DeepseekChat = ChatOpenAI(
    model="deepseek-chat",  # 官方模型名
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",  # 必须写对
)

response = DeepseekChat.invoke("用LangChain对接DeepSeek模型的核心步骤是什么？")
print(response.content)
