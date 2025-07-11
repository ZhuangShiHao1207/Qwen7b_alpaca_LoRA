import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

origin_model_path = "/home/jusheng/shihao/qwen_7B_finetune/models/Qwen-7B-Chat"
finetuned_model_path = "/home/jusheng/shihao/qwen7b_alpaca/output/qwen7b_alpaca"

prompts = [
    "请用一句话介绍一下人工智能。",
    "写一个Python函数，实现斐波那契数列。",
    "你怎么看待未来的自动驾驶技术？",
    "请将下面的英文翻译成中文：Machine learning is a subfield of artificial intelligence.",
]

def generate(model, tokenizer, prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=128,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

def test_model(model_path, tokenizer_path, prompts, tag, file):
    file.write(f"\n{'='*20}\n{tag}\n{'='*20}\n")
    print(f"\n{'='*20}\n{tag}\n{'='*20}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
    model.eval()
    for i, prompt in enumerate(prompts):
        file.write(f"\n【指令{i+1}】：{prompt}\n")
        print(f"\n【指令{i+1}】：{prompt}")
        response = generate(model, tokenizer, prompt)
        file.write(f"【回答】：{response}\n")
        print(f"【回答】：{response}")

if __name__ == "__main__":
    with open("compare_output.txt", "w", encoding="utf-8") as f:
        test_model(origin_model_path, origin_model_path, prompts, "原始模型", f)
        test_model(finetuned_model_path, origin_model_path, prompts, "微调后模型", f)