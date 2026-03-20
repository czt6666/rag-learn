from langchain.agents import create_agent
from langchain.agents.middleware import ModelResponse, ModelRequest, wrap_model_call, dynamic_prompt, wrap_tool_call
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import function, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from agent.init_llm import deepseek_llm, qwen_llm

_SUPPORTED_COMPANIES = ("apple", "google", "microsoft")


@tool
def get_stock_price(company: str, period: str = "current") -> str:
    """
    Return mock stock price data for the requested company and period.
    """
    company_key = company.strip().lower()
    period_key = (period or "current").strip().lower()

    price_book = {
        "current": {
            "apple": "Apple Inc. (AAPL) current price: $189.56 | daily change: +0.87%",
            "google": "Alphabet Inc. (GOOGL) current price: $148.23 | daily change: -0.32%",
            "microsoft": "Microsoft Corp. (MSFT) current price: $412.78 | daily change: +1.24%",
        },
        "last_week": {
            "apple": "Apple Inc. (AAPL) last week avg: $185.23 | weekly change: +2.35%",
            "google": "Alphabet Inc. (GOOGL) last week avg: $145.67 | weekly change: +1.78%",
            "microsoft": "Microsoft Corp. (MSFT) last week avg: $405.32 | weekly change: +1.84%",
        },
    }

    if company_key not in _SUPPORTED_COMPANIES:
        supported = ", ".join(_SUPPORTED_COMPANIES)
        return f"Mock stock data is only available for: {supported}."

    period_data = price_book.get(period_key, price_book["current"])
    return period_data[company_key]


@tool
def get_news(company: str, timeframe: str = "today") -> str:
    """
    Return mock company news headlines for the requested timeframe.
    """
    company_key = company.strip().lower()
    timeframe_key = (timeframe or "today").strip().lower()

    news_feed = {
        "today": {
            "apple": [
                "Apple confirms WWDC keynote schedule for June 2026.",
                "New M4-based MacBook Air rumored to enter trial production.",
            ],
            "google": [
                "Google announces Gemini for Workspace rollout to additional regions.",
                "Waymo expands robotaxi pilot to two more US cities.",
            ],
            "microsoft": [
                "Microsoft previews Copilot upgrades focused on security analysts.",
                "Xbox Cloud Gaming adds keyboard and mouse support for more titles.",
            ],
        },
        "this_week": {
            "apple": [
                "Analysts report stronger-than-expected Vision Pro retention.",
                "Apple Services revenue projected to hit a new record this quarter.",
            ],
            "google": [
                "Google Cloud wins major retail client focused on supply-chain AI.",
                "DeepMind open-sources lightweight reinforcement learning toolkit.",
            ],
            "microsoft": [
                "Microsoft to invest $2B in new European data center footprint.",
                "Partnership with OpenAI brings new safety benchmarks to Azure AI.",
            ],
        },
    }

    if company_key not in _SUPPORTED_COMPANIES:
        supported = ", ".join(_SUPPORTED_COMPANIES)
        return f"Mock news data is only available for: {supported}."

    company_news = news_feed.get(timeframe_key, news_feed["today"])[company_key]
    return "\n".join(f"- {headline}" for headline in company_news)


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler: function) -> ModelResponse:
    """根据对话次数动态选择模型"""
    messages_count = len(request.state['messages'])

    if messages_count > 3:
        model = deepseek_llm
    else:
        model = qwen_llm

    return handler(request.override(model=model))


@dynamic_prompt
def dynamic_prompt(request: ModelRequest) -> str:
    print(request)
    user_type = request.runtime.context.get("user_type", "normal")
    if user_type == "vip":
        prompt = "回答用户问题之前，首先称呼：尊贵的vip客户你好，然后再回答用户问题"
    else:
        prompt = "直接回答用户问题"

    return prompt


@wrap_tool_call
def handle_tool_error(request, handler):
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            tool_call_id=request.tool_call['id'],
            content=f"目前工具函数不可用，错误信息: {str(e)}"
        )


class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str


agent = create_agent(
    name="stock_agent",
    model=deepseek_llm,
    tools=[get_stock_price, get_news],
    middleware=[dynamic_model_selection, dynamic_prompt, handle_tool_error],
    response_format=ToolStrategy(ContactInfo)
)

response = agent.invoke(
    input={
        "messages": [
            {
                "role": "user",
                "content": "比较一下apple和google上周的股价",
            }
        ]
    },
    context={"user_type": "vip"}
)

print(response)
