import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# --- 1. 配置模型和分词器 (与之前相同) ---
model_path = "/home/jusheng/shihao/qwen_7B_finetune/models/Qwen-7B-Chat"
data_path = "/home/jusheng/shihao/qwen_7B_finetune/data/alpaca_zh_51k/alpaca_data_51k.json"
output_dir = "/home/jusheng/shihao/qwen7b_alpaca/output/qwen7b_alpaca"

# --- 2. 加载模型和分词器 (使用QLoRA) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config, # 使用BitsAndBytesConfig加载4-bit模型
    trust_remote_code=True,
    local_files_only=True,
    device_map="auto", # 自动分配设备
)
# 注意：SFTTrainer 内部会处理 prepare_model_for_kbit_training，所以我们不再需要手动调用

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    use_fast = False
)

tokenizer.eos_token = "<|endoftext|>"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # SFTTrainer推荐将padding放在右侧

# --- 3. 数据集准备  ---
# 加载Alpaca数据集
dataset = load_dataset("json", data_files=data_path, split="train")
print(dataset[0:2])

# 让模板紧跟在prompt之后，避免引入不必要的空格。
response_template = "### 回答:"
# 完整文本的eos_token应该由Trainer或Collator处理，而不是在格式化时手动添加
def format_data(example):
    """将 instruction/input 合并为 prompt，并创建用于训练的完整文本列"""
    if example.get("input", ""):
        prompt = f"### 指令:\n{example['instruction']}\n\n### 输入:\n{example['input']}\n\n"
    else:
        prompt = f"### 指令:\n{example['instruction']}\n\n"
    
    # 完整的文本应该拼接 prompt, response_template, 和 completion
    # tokenizer.eos_token 会在分词时由SFTTrainer自动处理，这里可以不加
    return {
        "text": f"{prompt}{response_template}{example['output']}"
    }
processed_dataset = dataset.map(format_data, remove_columns=list(dataset.features))

print("处理后的数据集样本:")
print(processed_dataset[0])

# def format_data(example):
#     """将 instruction/input 合并为 prompt，并创建用于训练的完整文本列"""
#     output_text = []
#     for i in range(len(example['instruction'])):
#         if example["input"][i]:
#             prompt = f"### 指令:\n{example['instruction']}\n\n### 输入:\n{example['input']}\n\n"
#         else:
#             prompt = f"### 指令:\n{example['instruction']}\n\n"
#         output_text.append(f"{prompt}{response_template}{example['output'][i]}")
    
#     # tokenizer.eos_token 会在分词时由SFTTrainer自动处理，这里可以不加
#     return output_text

# --- 4. 初始化 DataCollatorForCompletionOnlyLM ---
# 这是最关键的一步
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template, 
    tokenizer=tokenizer,
)

# --- 5. 配置参数---
# training_args = TrainingArguments(
#     output_dir=output_dir,
#     per_device_train_batch_size=4,
#     gradient_accumulation_steps=4,
#     learning_rate=2e-4,
#     num_train_epochs=2,
#     logging_steps=10,
#     save_strategy="epoch",
#     fp16=True,
#     optim="paged_adamw_8bit",
#     lr_scheduler_type="cosine",
#     warmup_ratio=0.03,
#     report_to="tensorboard",
# )

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["c_attn", "c_proj", "w1", "w2"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

sft_config = SFTConfig(
    dataset_text_field="text",  # 指定数据集中用于训练的文本字段
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=5e-4,
    num_train_epochs=1,
    logging_steps=200,
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    report_to="tensorboard",
    eos_token=tokenizer.eos_token,  # 使用分词器的eos_token
)

# --- 6. 初始化 SFTTrainer 并开始训练---
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=processed_dataset,
    data_collator=collator,      # 使用我们自定义的DataCollator
    peft_config=lora_config,
)
# trainer = SFTTrainer(
#     model,
#     train_dataset=dataset,
#     formatting_func=format_data,  # 使用自定义的格式化函数
#     peft_config=lora_config,        # 使用LoRA配置
#     args=training_args,             # 传入标准的TrainingArguments
#     data_collator=collator,
# )

# 打印可训练参数 (SFTTrainer 内部已经应用了 PEFT)
trainer.model.print_trainable_parameters()
# trainable params: 24,117,248 || all params: 7,744,344,064 || trainable%: 0.3114227918342335

print("开始训练...")
trainer.train()

# 保存最终的LoRA适配器
print("训练完成，保存模型。")
trainer.save_model(output_dir)