"""
BaseTool可以定义复杂工具
实现工具间共享内存
"""
from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from pydantic import Field, BaseModel


class SearchArgs(BaseModel):
    query: str = Field(..., description="需要进行互联网查询的信息")


class MyWebSearchTool(BaseTool):
    name = "MyWebSearchTool"
    description = "使用这个工具可以进行网络搜索"
    args_schema = SearchArgs

    def _run(self, query: str) -> str:
        try:
            response = zhipuai_client.web_search.web_search(
                search_query=query,
                search_engin="seearch_pro"
            )

            if response.search_result:
                return response.search_result
            return "没有搜索到任何结果"
        except Exception as e:
            print(e)
            return "没有搜索到任何结果"

    # 支持异步调用
    async def _run(self, query: str) -> str:
        return self._run(query)


agent = create_agent(...)
agent.ainvok()
