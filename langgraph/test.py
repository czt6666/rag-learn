import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.agents.middleware import SummarizationMiddleware
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver, MemorySaver
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore

# 环境变量
load_dotenv()
base_url = os.getenv("YUNWU_BASE_URL")
api_key = os.getenv("YUNWU_API_KEY")

model = init_chat_model("gpt-4o-mini", model_provider="openai", base_url=base_url, api_key=api_key)

checkpointer = MemorySaver()

summarization = SummarizationMiddleware(
    model="gpt-4.1-mini",  # 专门拿来做摘要，便宜一点
    trigger=("tokens", 2000),  # 到 2000 token 开始压缩
    keep=("messages", 8),  # 保留最近 8 条消息
)

store = InMemoryStore()
user_id = "1001"
namespace = ("user_id", user_id)
store.put(namespace, "profile", {
    "name": "小明",
    "age": 18,
    "city": "北京"
})


@tool
def get_user_info(runtime: ToolRuntime) -> str:
    """根据当前用户id查询用户信息"""

    user_id = runtime.context["user_id"]
    print(f"user_id={user_id}")
    namespace = ("users", user_id)
    print(f"namespace={namespace}")

    item = runtime.store.get(namespace, "profile")
    print(f"item={item}")

    if item is None:
        return "查无此人"

    return f"用户信息：{item.value}"


@dataclass
class Context:
    user_id: str


agent = create_agent(
    model=model,
    tools=[get_user_info],
    checkpointer=checkpointer,
    middleware=[summarization],
    store=store,
    context_schema=Context
)

config: RunnableConfig = {
    "configurable": {
        "thread_id": "user_1"
    },
}

res = agent.invoke({"messages": [{"role": "user", "content": "帮我查一下我的信息"}]}, config=config,
                   context=Context(user_id="1001"))
print(res.content)

# for chunk in agent.stream({"messages": [{"role": "user", "content": "你是谁？"}]}, config=config, stream_mode="updates"):
#     print(chunk)
