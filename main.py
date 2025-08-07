import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 中的环境变量
load_dotenv()

# 从环境变量中获取 API Key
api_key = os.getenv("OPENAI_API_KEY")

# 初始化 Client
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# 发送流式请求
stream = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "你好，你是谁"},
    ],
    stream=True  # ✅ 开启流式
)

# 实时打印内容
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

print()  # 输出换行
