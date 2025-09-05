from operator import itemgetter
from pathlib import Path

# from langchain.chains.summarize.map_reduce_prompt import prompt_template
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableParallel
from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

# 1 set llm
llm = ChatOpenAI(model="deepseek-v3")
embedding_model = HuggingFaceEmbeddings(model_name="/Users/v_changzhitao/Desktop/project/kg/Models/BAAI/bge-large-zh-v1___5")

# set data progress
file_dir = Path("my_knowledge")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
vector_store = Chroma(embedding_function=embedding_model,persist_directory="./chroma_v3")
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

prompt_template = PromptTemplate.from_template("""
你是一个严谨的RAG助手。请根据以下提供的上下文信息来回答问题，
如果上细纹信息不足以回答问题，请直接说“根据提供的信息无法回答”。
如果回答时使用了上下文，在回带后输出使用了那些上下文。
上下文信息：
{context}
------------
问题：{question}
""")
chain = {"question": RunnablePassthrough()} | RunnablePassthrough.assign(context=itemgetter("question") | retriever) | prompt_template | llm | StrOutputParser

