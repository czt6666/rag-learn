"""
记忆系统的用处
1. 对话上下文保持
2. 用户偏好/个人信息的保持
3. 累计历史交流，帮助后续改进

请记忆关键信息，减少token消耗
发送的信息不要超过模型窗口
"""
from langchain_community.memory.kg import ConversationKGMemory

from Agent.model import DeepseekChat

memory = ConversationBufferMemory()  # 全记，拼接到prompt里
memory = ConversationBufferWindowMemory()  # 滑动窗口
memory = ConversationSummaryMemory()  # LLM自动总结
memory = ConversationSummaryBufferMemory()  # 最近几轮存原文，更早的自动总结
memory = VectorStoreRetrieverMemory()  # 语义相似度检索记忆
memory = ConversationEntityMemory()  # none 存 {实体：value} 小KV对
memory = ConversationKGMemory()  # 存SOP三元组，存到土数据库里

conversation = ConversationChain(llm=DeepseekChat, memory=memory)
