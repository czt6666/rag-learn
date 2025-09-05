from langchain.prompts import PromptTemplate

template = PromptTemplate.from_template("请告诉我一个关于{subject}的笑话")

prompt_a = template.format(subject="程序员")
prompt_b = template.format(subject="汽车")
prompt_c = template.format(subject="国际")

print(prompt_a)
print(prompt_b)
print(prompt_c)
