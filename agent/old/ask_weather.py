from dotenv import load_dotenv
import os
import json

from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from utils import to_json_safe

load_dotenv()  # 自动读取当前项目根目录的 .env

llm = ChatDeepSeek(
    model="deepseek-chat",  # 或 deepseek-reasoner
    temperature=0
)


@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


search_tool = DuckDuckGoSearchRun()

agent = create_agent(
    model=llm,
    tools=[get_weather, search_tool],
    system_prompt="You are a helpful assistant",
)

print(llm.profile)
print(llm.dev)

# Run the agent
# result = agent.invoke(
#     {"messages": [{"role": "user", "content": "哈尔滨今天的天气怎么样？"}]}
# )
# print(json.dumps(result, indent=2, default=to_json_safe, ensure_ascii=False))

# for chunk in agent.stream(
#         {"messages": [{"role": "user", "content": "哈尔滨今天的天气怎么样？"}]},
#         stream_mode="messages",
# ):
#     for step, data in chunk.items():
#         print(f"step: {step}")
#         # print(f"content: {data['messages'][-1].content_blocks}")
#         print(json.dumps(data, indent=2, default=to_json_safe, ensure_ascii=False))


# for token, metadata in agent.stream(
#         {"messages": [{"role": "user", "content": "哈尔滨今天的天气怎么样？"}]},
#         stream_mode="messages",
# ):
#     print(f"node: {metadata['langgraph_node']}")
#     print(f"content: {token.content_blocks}")
#     print("\n")
