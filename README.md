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

# 📌 English Overview: Qwen-7B LoRA Fine-tuned Model (Chinese Instruction Tuning)

This model is fine-tuned from Alibaba’s Qwen-7B-Chat using LoRA technique on the Alpaca-Zh-51k dataset. It is suitable for instruction-following tasks in Chinese.

(I found that after making adjustments to Chat model, the effect actually got worse. Perhaps making adjustments to the base version would be better)

## 🧾 Model Information

* **Base model**: [`Qwen/Qwen-7B-Chat`](https://huggingface.co/Qwen/Qwen-7B-Chat)
* **Tuning method**: LoRA (via `peft`)
* **Dataset**: Alpaca-Zh-51k
* **Training script**: `train_qwen7b_lora.py`
* **Inference script**: `test_compare.py`
* ⚠️ This repository includes only LoRA adapter weights, not the original base model.

## 🚀 Usage Example

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_name = "Josh1207/qwen7b-alpaca-lora"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, model_name)

prompt = "指令: 请介绍一下你自己。"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---
# 📌 中文简介：Qwen-7B LoRA 微调模型（中文指令微调）

本模型基于阿里巴巴通义千问 Qwen-7B-Chat，采用 LoRA 技术，使用 Alpaca-Zh-51k 数据集进行了中文指令微调，适用于中文任务的理解与生成。

注： 对Chat进行微调后效果反而变差了，或许对base版本微调会好一些

## 🧾 模型信息

- **基座模型**：[`Qwen/Qwen-7B-Chat`](https://huggingface.co/Qwen/Qwen-7B-Chat)
- **微调方法**：LoRA（使用 PEFT 库）
- **训练数据集**：Alpaca-Zh-51k
- **训练脚本**：`train_qwen7b_lora.py`
- **推理脚本**：`test_compare.py`
- ⚠️ 本模型仅包含 LoRA adapter，不包含原始基座权重

## 🚀 使用示例

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_name = "Josh1207/qwen7b-alpaca-lora"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, model_name)

prompt = "指令: 请介绍一下你自己。"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
````


---


