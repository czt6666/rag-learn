"""
LangChain Tools 全面使用示例
展示所有工具创建和使用方法
"""

from dotenv import load_dotenv
import json
from typing import Literal, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from langchain.tools import tool, ToolRuntime
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.types import Command
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.store.memory import InMemoryStore

load_dotenv()


# ==================== 1. 基础工具定义 ====================
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city.

    Args:
        city: Name of the city to get weather for
    """
    return f"It's always sunny in {city}!"


# ==================== 2. 自定义工具名称 ====================
@tool("web_search")
def search(query: str) -> str:
    """Search the web for information.

    Args:
        query: Search query string
    """
    return f"Results for: {query}"


# ==================== 3. 自定义工具描述 ====================
@tool(
    "calculator",
    description="Performs arithmetic calculations. Use this for any math problems."
)
def calc(expression: str) -> str:
    """Evaluate mathematical expressions.

    Args:
        expression: Mathematical expression to evaluate
    """
    try:
        # 注意: eval 在生产环境中不安全,这里仅作示例
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


# ==================== 4. 使用 Pydantic 模型定义复杂输入 ====================
class WeatherInput(BaseModel):
    """Input for weather queries."""
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )


@tool(args_schema=WeatherInput)
def get_advanced_weather(
        location: str,
        units: str = "celsius",
        include_forecast: bool = False
) -> str:
    """Get current weather and optional forecast.

    Args:
        location: City name or coordinates
        units: Temperature unit (celsius or fahrenheit)
        include_forecast: Whether to include 5-day forecast
    """
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny, Cloudy, Rainy, Sunny, Partly Cloudy"
    return result


# ==================== 5. 使用 ToolRuntime 访问状态 ====================
@tool
def summarize_conversation(runtime: ToolRuntime) -> str:
    """Summarize the conversation so far.

    The runtime parameter is automatically injected and hidden from the model.
    """
    messages = runtime.state.get("messages", [])

    human_msgs = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
    ai_msgs = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")
    tool_msgs = sum(1 for m in messages if m.__class__.__name__ == "ToolMessage")

    return (f"Conversation summary:\n"
            f"- User messages: {human_msgs}\n"
            f"- AI responses: {ai_msgs}\n"
            f"- Tool results: {tool_msgs}\n"
            f"- Total messages: {len(messages)}")


@tool
def get_user_preference(pref_name: str, runtime: ToolRuntime) -> str:
    """Get a user preference value.

    Args:
        pref_name: Name of the preference to retrieve
    """
    preferences = runtime.state.get("user_preferences", {})
    return preferences.get(pref_name, f"Preference '{pref_name}' not set")


# ==================== 6. 使用 Command 更新状态 ====================
@tool
def clear_conversation() -> Command:
    """Clear the conversation history.

    Returns a Command to update the agent's state.
    """
    return Command(
        update={
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
        }
    )


@tool
def update_user_name(new_name: str, runtime: ToolRuntime) -> Command:
    """Update the user's name in the state.

    Args:
        new_name: New name for the user
    """
    return Command(update={"user_name": new_name})


# ==================== 7. 访问 Context (上下文) ====================
# 模拟用户数据库
USER_DATABASE = {
    "user123": {
        "name": "Alice Johnson",
        "account_type": "Premium",
        "balance": 5000,
        "email": "[email protected]"
    },
    "user456": {
        "name": "Bob Smith",
        "account_type": "Standard",
        "balance": 1200,
        "email": "[email protected]"
    }
}


@dataclass
class UserContext:
    """用户上下文信息"""
    user_id: str
    session_id: str = "default"


@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
    """Get the current user's account information.

    Uses runtime context to identify the user.
    """
    user_id = runtime.context.user_id

    if user_id in USER_DATABASE:
        user = USER_DATABASE[user_id]
        return (f"Account Information:\n"
                f"- Name: {user['name']}\n"
                f"- Type: {user['account_type']}\n"
                f"- Balance: ${user['balance']}\n"
                f"- Email: {user['email']}")
    return "User not found in database"


# ==================== 8. 访问 Store (持久化存储) ====================
@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """Look up user info from persistent storage.

    Args:
        user_id: ID of the user to look up
    """
    store = runtime.store
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else f"User {user_id} not found"


@tool
def save_user_info(user_id: str, name: str, age: int, email: str, runtime: ToolRuntime) -> str:
    """Save user info to persistent storage.

    Args:
        user_id: User's unique identifier
        name: User's name
        age: User's age
        email: User's email address
    """
    store = runtime.store
    user_data = {
        "name": name,
        "age": age,
        "email": email
    }
    store.put(("users",), user_id, user_data)
    return f"Successfully saved info for user {user_id}"


# ==================== 9. 使用 Stream Writer ====================
@tool
def process_large_file(filename: str, runtime: ToolRuntime) -> str:
    """Process a large file with progress updates.

    Args:
        filename: Name of the file to process
    """
    writer = runtime.stream_writer

    # 模拟文件处理过程
    steps = [
        "Opening file",
        "Reading contents",
        "Processing data",
        "Validating results",
        "Saving output"
    ]

    for i, step in enumerate(steps, 1):
        writer(f"[{i}/{len(steps)}] {step}: {filename}")

    return f"Successfully processed {filename}"


# ==================== 10. 数据库搜索工具示例 ====================
class DatabaseSearchInput(BaseModel):
    """Input schema for database search."""
    query: str = Field(description="Search terms to look for")
    limit: int = Field(default=10, description="Maximum number of results")
    table: str = Field(default="customers", description="Database table to search")


@tool(args_schema=DatabaseSearchInput)
def search_database(query: str, limit: int = 10, table: str = "customers") -> str:
    """Search the database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
        table: Database table to search
    """
    # 模拟数据库搜索
    return (f"Database Search Results:\n"
            f"- Table: {table}\n"
            f"- Query: '{query}'\n"
            f"- Found {limit} matching records")


# ==================== 辅助函数 ====================
def to_json_safe(obj):
    """Convert objects to JSON-safe format."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return str(obj)


# ==================== 主程序 ====================
def main():
    """运行各种工具示例"""

    # 初始化 LLM
    llm = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0
    )

    # 初始化社区工具
    search_tool = DuckDuckGoSearchRun()

    # 初始化持久化存储
    store = InMemoryStore()

    print("=" * 60)
    print("LangChain Tools 全面使用示例")
    print("=" * 60)

    # ==================== 示例 1: 基础工具 ====================
    print("\n【示例 1】基础工具 - 天气查询")
    print("-" * 60)

    agent1 = create_agent(
        model=llm,
        tools=[get_weather, search_tool],
        system_prompt="You are a helpful weather assistant.",
    )

    result1 = agent1.invoke(
        {"messages": [{"role": "user", "content": "北京今天的天气怎么样?"}]}
    )
    print(json.dumps(result1, indent=2, default=to_json_safe, ensure_ascii=False))

    # ==================== 示例 2: 复杂输入工具 ====================
    print("\n【示例 2】复杂输入工具 - 高级天气查询")
    print("-" * 60)

    agent2 = create_agent(
        model=llm,
        tools=[get_advanced_weather],
        system_prompt="You are a weather expert.",
    )

    result2 = agent2.invoke(
        {"messages": [{"role": "user", "content": "查询上海的天气,使用华氏度,包含未来预报"}]}
    )
    print(json.dumps(result2, indent=2, default=to_json_safe, ensure_ascii=False))

    # ==================== 示例 3: 计算器工具 ====================
    print("\n【示例 3】计算器工具")
    print("-" * 60)

    agent3 = create_agent(
        model=llm,
        tools=[calc],
        system_prompt="You are a math assistant.",
    )

    result3 = agent3.invoke(
        {"messages": [{"role": "user", "content": "计算 (123 + 456) * 2 等于多少?"}]}
    )
    print(json.dumps(result3, indent=2, default=to_json_safe, ensure_ascii=False))

    # ==================== 示例 4: 访问状态 ====================
    print("\n【示例 4】访问状态 - 对话总结")
    print("-" * 60)

    agent4 = create_agent(
        model=llm,
        tools=[summarize_conversation],
        system_prompt="You are a helpful assistant that can summarize conversations.",
    )

    # 进行多轮对话
    result4a = agent4.invoke(
        {"messages": [{"role": "user", "content": "你好!"}]}
    )
    result4b = agent4.invoke(
        {"messages": result4a["messages"] + [{"role": "user", "content": "请总结一下我们的对话"}]}
    )
    print(json.dumps(result4b, indent=2, default=to_json_safe, ensure_ascii=False))

    # ==================== 示例 5: 持久化存储 ====================
    print("\n【示例 5】持久化存储 - 保存和检索用户信息")
    print("-" * 60)

    agent5 = create_agent(
        model=llm,
        tools=[save_user_info, get_user_info],
        store=store,
        system_prompt="You are a user management assistant.",
    )

    # 保存用户信息
    result5a = agent5.invoke(
        {"messages": [
            {"role": "user", "content": "保存用户信息: ID为abc123, 名字是张三, 年龄28岁, 邮箱是[email protected]"}]}
    )
    print("保存结果:", json.dumps(result5a, indent=2, default=to_json_safe, ensure_ascii=False))

    # 检索用户信息
    result5b = agent5.invoke(
        {"messages": [{"role": "user", "content": "查询用户abc123的信息"}]}
    )
    print("\n检索结果:", json.dumps(result5b, indent=2, default=to_json_safe, ensure_ascii=False))

    # ==================== 示例 6: 数据库搜索 ====================
    print("\n【示例 6】数据库搜索工具")
    print("-" * 60)

    agent6 = create_agent(
        model=llm,
        tools=[search_database],
        system_prompt="You are a database query assistant.",
    )

    result6 = agent6.invoke(
        {"messages": [{"role": "user", "content": "在customers表中搜索'张',最多返回5条记录"}]}
    )
    print(json.dumps(result6, indent=2, default=to_json_safe, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
