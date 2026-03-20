import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, TaskType, get_peft_model

model_name = "Langboat/bloom-1b4-zh"
output_dir = "./bloom-1b4-zh-lora"

# 自动选择设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

# 1. 读数据
ds = load_dataset("shibing624/alpaca-zh", split="train")
print("样本数量:", len(ds))
print("第一条样本:", ds[0])

# 2. tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

MAX_LENGTH = 256


# 3. 预处理
def process_func(example):
    instruction_text = "\n".join([
        "Human: " + example["instruction"],
        example["input"]
    ]).strip() + "\n\nAssistant: "

    instruction = tokenizer(instruction_text, add_special_tokens=False)
    response = tokenizer(example["output"] + tokenizer.eos_token, add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)

print("处理后的第一条 input:")
print(tokenizer.decode(tokenized_ds[0]["input_ids"]))
print("处理后的第一条 label 对应文本:")
print(tokenizer.decode([x for x in tokenized_ds[0]["labels"] if x != -100]))

# 4. 切分训练/验证
split_ds = tokenized_ds.train_test_split(test_size=0.02, seed=42)
train_ds = split_ds["train"]
eval_ds = split_ds["test"]
print("训练集数量:", len(train_ds))
print("验证集数量:", len(eval_ds))

# 5. 模型（CUDA 用 float16 省显存，CPU 用 float32）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
model.config.use_cache = False  # 训练时关闭 KV cache

# 6. LoRA 配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["query_key_value"],  # bloom 的注意力层名称
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
print(lora_config)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 7. collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
)

# 8. 训练参数
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,  # GPU 可以适当加大
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=1,
    fp16=False,  # ← 关闭 fp16
    bf16=True,  # ← 开启 bf16
    save_steps=500,
    save_total_limit=2,
    dataloader_pin_memory=device == "cuda",
)

# 9. 训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
)

# 10. 开始训练
print("开始训练")
trainer.train()

# 11. 保存
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("训练完成，模型已保存到:", output_dir)
