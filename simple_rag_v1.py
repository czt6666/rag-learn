import os
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


# 关键词匹配
def retrieval(query):
    context = ""

    path_list = list(Path("my_knowledge").glob("*.txt"))

    for path in path_list:
        if path.stem in query:
            context += path.read_text(encoding="utf-8") + "\n\n"

    return context

# print(retrieval("无人机有什么"))

# 增强 Query
def augmented(query, context=""):
    if not context:
        return f"请简要回答下面问题：{query}"

    return f"""请根据上下文信息来回答问题。如果上下文不足以回答问题，请直接说: “根据上下文信息无法回答问题”
上下文：
{context}
问题：
{query}
"""

query = "无人机型号是什么"
print(augmented(query, retrieval(query)))

# 生成回答



def generation(prompt):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": f"{prompt}"},
            {"role": "user", "content": "你好，你是谁"},
        ],
        stream=False
    )

    return response.choices[0].message.content

print(generation(augmented(query, retrieval(query))))