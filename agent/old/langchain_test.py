import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_community.utilities import SerpAPIWrapper
from utils import print_result


# 2. 定义数学工具
@tool
def calculator(expr: str) -> str:
    """
    提供一个简单计算器，执行表达式并返回结果。
    expr: 例如 "3 * 4 + 2"
    """
    try:
        # 安全一点 eval
        result = eval(expr, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


# 3. SerpAPI 搜索工具
search = SerpAPIWrapper()


@tool
def web_search(query: str) -> str:
    """用 SerpAPI 搜索网络，返回摘要结果"""
    return search.run(query)


# 4. 工具列表
tools = [
    calculator,
    web_search,
]

# 5. 初始化 LLM
llm = ChatDeepSeek(model="deepseek-chat")

# 6. 构建 Prompt
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "你是一个智能助手，能使用搜索和数学工具回答问题。"),
#     ("human", "{input}"),
#     MessagesPlaceholder("agent_scratchpad"),
# ])

# 7. 创建代理
agent = create_agent(model=llm, tools=tools)

# 8. 运行测试
query = "现任中国主席是谁？他的年龄的平方是多少？请先搜索出名字和年龄，然后算年龄的平方。"
result = agent.invoke(
    {"messages": [{"role": "user", "content": query}]}
)

print_result(result)
