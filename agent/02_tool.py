from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from agent.init_llm import deepseek_llm


@tool
def get_weather(city: str) -> str:
    """
    获取城市的天气
    Args:
        city (str): 城市名称
        例如: "北京"
    Returns:
        str: 城市的天气信息
    """
    return f"{city}的天气是晴朗的，温度是25摄氏度"


model_bind_tool = deepseek_llm.bind_tools([get_weather])
message = [HumanMessage(content="北京的天气")]

res = model_bind_tool.invoke(message)
message.append(res)
print(res.tool_calls)

for tool_call in res.tool_calls:
    print(tool_call)
    if tool_call["name"] == "get_weather":
        tool_result = get_weather.invoke(tool_call)
        message.append(tool_result)

res = model_bind_tool.invoke(message)

print(message)
print(res)
