from pydantic import BaseModel, Field

from agent.init_llm import deepseek_llm

"""
pydantic 结构化
"""


class Movie(BaseModel):
    title: str = Field(description="电影标题")
    year: int = Field(description="电影上映年份")
    director: str = Field(description="电影导演")
    rate: float = Field(description="电影评分")


model_structured_output = deepseek_llm.with_structured_output(Movie)

resp = model_structured_output.invoke("给我介绍一下肖申克的救赎")
print(resp)

"""
typeddict 结构化
"""
# from typing import TypedDict, Annotated
#
#
# class Movie(TypedDict):
#     title: Annotated[str, "电影标题"]
#     year: Annotated[int, "电影上映年份"]
#     director: Annotated[str, "电影导演"]
#     rate: Annotated[float, "电影评分"]
#
#
# model_structured_output = deepseek_llm.with_structured_output(Movie)
#
# resp = model_structured_output.invoke("给我介绍一下肖申克的救赎")
# print(type(resp))
# print(resp)
