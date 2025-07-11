---
license: apache-2.0
---
---
library_name: peft
model_name: qwen7b_alpaca
tags:
- base_model:adapter:/home/jusheng/shihao/qwen_7B_finetune/models/Qwen-7B-Chat
- lora
- sft
- transformers
- trl
licence: license
base_model: /home/jusheng/shihao/qwen_7B_finetune/models/Qwen-7B-Chat
pipeline_tag: text-generation
---

# Model Card for qwen7b_alpaca

This model is a fine-tuned version of [None](https://huggingface.co/None).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with SFT.

### Framework versions

- PEFT 0.16.0
- TRL: 0.19.1
- Transformers: 4.53.1
- Pytorch: 2.7.1
- Datasets: 3.6.0
- Tokenizers: 0.21.4.dev0

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```