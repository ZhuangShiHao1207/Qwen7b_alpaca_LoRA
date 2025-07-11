---
license: apache-2.0
language: zh
tags:
  - Qwen
  - LLM
  - LoRA
  - Fine-tuned
  - Instruction Tuning
pipeline_tag: text-generation
---

# Qwen-7B LoRA 微调模型（中文指令微调）

本模型基于阿里巴巴通义千问 Qwen-7B-Chat，采用 LoRA 技术，使用 Alpaca-Zh-51k 数据集进行中文指令微调。适合中文指令理解与生成任务。

## 训练说明

- **基座模型**：Qwen-7B-Chat（未上传原始权重，仅上传 LoRA adapter）
- **微调方法**：LoRA
- **数据集**：Alpaca-Zh-51k
- **训练脚本**：见仓库 `train_qwen7b_lora.py`
- **推理/对比脚本**：见仓库 `test_compare.py`

## 使用方法

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 替换为你的 Hugging Face 用户名和模型名
model_name = "Josh1207/qwen7b-alpaca-lora"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, model_name)

prompt = "指令: 请介绍一下你自己。"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))