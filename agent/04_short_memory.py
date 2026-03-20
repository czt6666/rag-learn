"""
LangChain 短期记忆（Short-term memory）示例

这个文件把官方文档里最常见的几种用法，整理成一个更容易看懂的 Python 示例。

覆盖内容：
1. 基础用法：给 agent 加 checkpointer，让同一个 thread_id 下的对话能“记住上下文”
2. 自定义状态：除了 messages，还能额外保存 user_id / preferences 这类字段
3. 裁剪消息：在模型调用前，只保留必要消息，避免上下文过长
4. 摘要消息：把旧消息压缩成摘要，减少 token 开销
5. 工具读取记忆：tool 里读取当前 state
6. 工具写入记忆：tool 执行后把结果写回 state，给后续步骤复用

注意：
- 这是“短期记忆”，本质上是“当前会话 / 当前线程”的状态持久化。
- 它不是向量数据库，不是长期知识库，也不是 RAG。
- 你可以把它理解成：给 agent 挂一个“会话状态存档器”。

参考文档：
https://docs.langchain.com/oss/python/langchain/short-term-memory
"""

from __future__ import annotations

from typing import Any, TypedDict

# ===== 基础依赖 =====
from langchain.agents import AgentState, create_agent
from langchain.tools import ToolRuntime, tool
from langchain.messages import RemoveMessage, ToolMessage
from langchain.agents.middleware import (
    SummarizationMiddleware,
    before_model,
    dynamic_prompt,
    ModelRequest,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel


# -----------------------------------------------------------------------------
# 1) 基础用法：同一个 thread_id 下，agent 能记住前面对话
# -----------------------------------------------------------------------------
# 原理：
# - 短期记忆不是“模型自己记住了”，而是 LangGraph / LangChain 在每一步执行后，
#   把 state（尤其是 messages）保存到 checkpointer。
# - 下次再带着同一个 thread_id 调用时，会先把旧 state 读出来，再继续跑。
# - 所以 thread_id 就像“会话主键”。相同 thread_id => 同一段会话；不同 thread_id => 隔离。
# -----------------------------------------------------------------------------

def demo_basic_short_term_memory() -> None:
    checkpointer = InMemorySaver()  # 演示环境常用；生产里一般换成数据库后端

    agent = create_agent(
        model="gpt-4.1-mini",  # 这里换成你自己的模型名也行
        tools=[],
        checkpointer=checkpointer,
    )

    # 同一个线程：thread_id = "demo-thread-1"
    config: RunnableConfig = {"configurable": {"thread_id": "demo-thread-1"}}

    # 第一次对话：告诉 agent 我的名字
    agent.invoke(
        {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},
        config,
    )

    # 第二次对话：同一个 thread_id，所以它能接着上文
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What's my name?"}]},
        config,
    )

    print("\n[基础短期记忆示例]")
    print(result["messages"][-1].content)


# -----------------------------------------------------------------------------
# 2) 自定义状态：除了 messages，还能额外保存结构化字段
# -----------------------------------------------------------------------------
# 原理：
# - 默认 AgentState 里最核心的是 messages。
# - 但很多业务里，你不只想存聊天记录，还想顺手存：
#   user_id / 用户偏好 / 当前任务ID / 已选语言 / 中间推理结果 等。
# - 这时就扩展 AgentState，定义自己的 state_schema。
# -----------------------------------------------------------------------------

class CustomAgentState(AgentState):
    user_id: str
    preferences: dict


def demo_custom_state() -> None:
    agent = create_agent(
        model="gpt-4.1-mini",
        tools=[],
        state_schema=CustomAgentState,
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "user_id": "user_123",
            "preferences": {"theme": "dark", "language": "zh-CN"},
        },
        {"configurable": {"thread_id": "custom-state-thread"}},
    )

    print("\n[自定义状态示例]")
    print(result["messages"][-1].content)


# -----------------------------------------------------------------------------
# 3) 裁剪消息：上下文太长时，只保留必要消息
# -----------------------------------------------------------------------------
# 原理：
# - 模型真正吃进去的是 messages 列表。
# - 对话一长，token 成本变高，模型还容易被旧信息干扰。
# - 所以常见做法是：在“调用模型之前”先做裁剪。
# - before_model 中间件就是干这个的。
#
# 这个示例策略比较朴素：
# - 永远保留第一条消息（通常它可能很重要，比如初始任务 / 初始约束）
# - 再保留最近几条消息
# -----------------------------------------------------------------------------

@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    messages = state["messages"]

    # 消息不多，就不裁
    if len(messages) <= 4:
        return None

    first_msg = messages[0]
    recent_messages = messages[-4:]
    new_messages = [first_msg] + recent_messages

    # RemoveMessage(id=REMOVE_ALL_MESSAGES) 的意思是：
    # 先把当前消息列表清空，再放入我们挑好的消息
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages,
        ]
    }


def demo_trim_messages() -> None:
    agent = create_agent(
        model="gpt-4.1-mini",
        tools=[],
        middleware=[trim_messages],
        checkpointer=InMemorySaver(),
    )

    config: RunnableConfig = {"configurable": {"thread_id": "trim-thread"}}

    agent.invoke({"messages": [{"role": "user", "content": "我叫 Bob"}]}, config)
    agent.invoke({"messages": [{"role": "user", "content": "给我写一首关于猫的诗"}]}, config)
    agent.invoke({"messages": [{"role": "user", "content": "再写一首关于狗的"}]}, config)
    result = agent.invoke({"messages": [{"role": "user", "content": "我叫什么名字？"}]}, config)

    print("\n[裁剪消息示例]")
    print(result["messages"][-1].content)


# -----------------------------------------------------------------------------
# 4) 摘要消息：不是直接删，而是压缩旧对话
# -----------------------------------------------------------------------------
# 原理：
# - 直接 trim / delete 很省事，但可能把关键事实也删了。
# - 更稳一点的做法是：把旧历史先总结成摘要，再保留最近若干条原始消息。
# - 这样模型既不会上下文爆炸，也不至于把关键信息全忘掉。
#
# 适合场景：
# - 多轮客服
# - 长流程 agent
# - 任务持续很久，历史很多，但又不能全扔
# -----------------------------------------------------------------------------

def demo_summarization_memory() -> None:
    agent = create_agent(
        model="gpt-4.1",
        tools=[],
        middleware=[
            SummarizationMiddleware(
                model="gpt-4.1-mini",  # 用更便宜的小模型做摘要，常见省钱套路
                trigger=("tokens", 4000),  # 超过这个 token 阈值时触发摘要
                keep=("messages", 20),     # 最近 20 条原始消息保留不动
            )
        ],
        checkpointer=InMemorySaver(),
    )

    config: RunnableConfig = {"configurable": {"thread_id": "summary-thread"}}

    agent.invoke({"messages": [{"role": "user", "content": "Hi, my name is Bob."}]}, config)
    agent.invoke({"messages": [{"role": "user", "content": "I like cats."}]}, config)
    result = agent.invoke({"messages": [{"role": "user", "content": "What's my name?"}]}, config)

    print("\n[摘要消息示例]")
    print(result["messages"][-1].content)


# -----------------------------------------------------------------------------
# 5) 工具里读取短期记忆：tool 直接访问 runtime.state
# -----------------------------------------------------------------------------
# 原理：
# - 有些信息不想让模型从自然语言里“猜”，而是希望工具直接读取 state。
# - 比如：当前 user_id、租户ID、会话配置、上一步工具产物。
# - ToolRuntime 是很关键的入口：
#   tool 看不见这个参数的自然语言签名，但运行时能用它拿到 state / context。
# -----------------------------------------------------------------------------

class ToolReadState(AgentState):
    user_id: str


@tool
def get_user_info(runtime: ToolRuntime) -> str:
    """根据当前 state 中的 user_id 查询用户信息。"""
    user_id = runtime.state["user_id"]
    return "User is John Smith" if user_id == "user_123" else "Unknown user"


def demo_tool_read_memory() -> None:
    agent = create_agent(
        model="gpt-4.1-mini",
        tools=[get_user_info],
        state_schema=ToolReadState,
    )

    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "Please look up my user information."}],
            "user_id": "user_123",
        }
    )

    print("\n[工具读取记忆示例]")
    print(result["messages"][-1].content)


# -----------------------------------------------------------------------------
# 6) 工具里写入短期记忆：tool 返回 Command(update=...)
# -----------------------------------------------------------------------------
# 原理：
# - 有时候 tool 的价值不只是“返回一段文本”，
#   更重要的是把结果写进 state，给后续步骤继续用。
# - 这特别像多步骤流水线：
#   第一个工具查资料 -> 写入 state
#   第二个工具 / 后续 prompt 直接消费这个 state
#
# 这比让模型自己在上下文里翻聊天记录更稳，
# 因为你把“关键中间结果”结构化存下来了。
# -----------------------------------------------------------------------------

class ToolWriteState(AgentState):
    user_name: str


class ToolContext(BaseModel):
    user_id: str


@tool
def update_user_info(runtime: ToolRuntime[ToolContext, ToolWriteState]) -> Command:
    """查询用户信息，并把结果写回短期记忆 state。"""
    user_id = runtime.context.user_id
    name = "John Smith" if user_id == "user_123" else "Unknown user"

    return Command(
        update={
            # 结构化字段写回 state
            "user_name": name,
            # 也可以顺手往 messages 里补一条 tool message
            "messages": [
                ToolMessage(
                    content="Successfully looked up user information.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def greet(runtime: ToolRuntime[ToolContext, ToolWriteState]) -> str | Command:
    """从 state 里读取 user_name，如果没有就提示先更新。"""
    user_name = runtime.state.get("user_name")

    if user_name is None:
        # 这里也可以返回 Command，把引导信息写进消息流
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="Please call update_user_info first to fetch the user's name.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    return f"Hello {user_name}!"


def demo_tool_write_memory() -> None:
    agent = create_agent(
        model="gpt-4.1-mini",
        tools=[update_user_info, greet],
        state_schema=ToolWriteState,
        context_schema=ToolContext,
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Greet the user."}]},
        context=ToolContext(user_id="user_123"),
    )

    print("\n[工具写入记忆示例]")
    print(result["messages"][-1].content)


# -----------------------------------------------------------------------------
# 7) 动态提示词：把短期记忆 / 上下文喂给 system prompt
# -----------------------------------------------------------------------------
# 原理：
# - 有些信息你不想每轮都手工拼 system prompt。
# - 可以在 dynamic_prompt 里，按当前上下文动态生成系统提示。
# - 这很适合“记住用户称呼 / 风格偏好 / 业务身份”等轻量信息。
# -----------------------------------------------------------------------------

class PromptContext(TypedDict):
    user_name: str


@tool
def get_weather(city: str) -> str:
    """获取天气（这里为了演示，返回固定值）。"""
    return f"The weather in {city} is always sunny!"


@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context["user_name"]
    return f"You are a helpful assistant. Address the user as {user_name}."


def demo_dynamic_prompt() -> None:
    agent = create_agent(
        model="gpt-4.1-mini",
        tools=[get_weather],
        middleware=[dynamic_system_prompt],
        context_schema=PromptContext,
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
        context=PromptContext(user_name="John Smith"),
    )

    print("\n[动态提示词示例]")
    print(result["messages"][-1].content)


# -----------------------------------------------------------------------------
# 8) 什么时候该用哪种方式？
# -----------------------------------------------------------------------------
# 一个大白话版总结：
#
# - 只想让对话“能续上”
#   -> 最基础：checkpointer + thread_id
#
# - 想存结构化会话字段（用户ID、偏好、阶段状态）
#   -> 自定义 AgentState
#
# - 对话越来越长，成本高、效果差
#   -> before_model 做 trim
#
# - 不能直接删历史，否则会丢信息
#   -> SummarizationMiddleware 做摘要
#
# - tool 需要知道“当前会话里已经有什么信息”
#   -> ToolRuntime 读 state
#
# - tool 查到结果后，后面步骤还想继续复用
#   -> Command(update=...) 写 state
#
# 你可以把它想象成前端里的状态管理：
# - messages ≈ 聊天历史 state
# - checkpointer ≈ 持久化层
# - thread_id ≈ 会话 key
# - middleware ≈ 调用前后的拦截器
# - tool 写 state ≈ action/reducer 更新 store
# -----------------------------------------------------------------------------


def main() -> None:
    # 这些 demo 默认不强依赖真实外部工具，但需要你本地配置好 LangChain / LangGraph
    # 以及可用的模型 API Key。
    #
    # 运行前建议安装（版本以你项目实际为准）：
    # pip install -U langchain langgraph pydantic
    #
    # 如果你要用 PostgreSQL 持久化：
    # pip install langgraph-checkpoint-postgres

    print("开始演示 LangChain 短期记忆...\n")

    # 你可以按需只打开其中几个 demo
    # demo_basic_short_term_memory()
    # demo_custom_state()
    # demo_trim_messages()
    # demo_summarization_memory()
    # demo_tool_read_memory()
    # demo_tool_write_memory()
    # demo_dynamic_prompt()

    print("这个文件主要是教学示例。请按需取消注释某个 demo 再运行。")


if __name__ == "__main__":
    main()
