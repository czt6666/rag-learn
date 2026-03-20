# 提示词
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain, RunnableLambda

from agent.init_llm import deepseek_llm

# prompt = ChatPromptTemplate([
#     ("system", "把用户输入的中文翻译成{Language}"),
#     ("user", "{text}"),
# ])
#
# # 输出解析器
# parser = StrOutputParser()
#
# # 链（简单）
# _chain = prompt | deepseek_llm | parser

# 调用执行链
# result = _chain.invoke({"Language": "英文", "text": "Jeff喜欢打篮球"})
# print(result)

# itemgetter
# d = {"foo": "abc", "bar": "def"}
# res = itemgetter("foo")(d)
# print(res)

template = ChatPromptTemplate.from_template("{a}+{b}是多少？")


# 获得字符串的长度
def length(t):
    return len(t)


# 将两个字符串长度的数量相乘
def mul(t1, t2):
    return len(t1) * len(t2)


# @chain是RunnableLambda的另一种写法：把函数转换为 LCEL 兼容的组件
@chain
def mul_length(d):
    return mul(d["t1"], d["t2"])


chain1 = template | deepseek_llm
chain2 = (
        {
            "a": itemgetter("name") | RunnableLambda(length),  # a=6
            "b": {"t1": itemgetter("name"), "t2": itemgetter("sex")} | mul_length,  # t1: wangwu, t2: male, b=24
        }
        | chain1
        | StrOutputParser()
)

print(chain2.invoke({"name": "wangwu", "sex": "male"}))
print('-' * 100)
