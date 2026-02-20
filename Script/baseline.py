# scripts/baseline.py
"""Baseline: 使用 HF 原生 KV Cache"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_name = '../Llama-2-7b-hf'  # 或 'meta-llama/Llama-2-7b-hf'
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    
    prompt = "Hello, I am a software engineer"
    print(f"Prompt: {prompt}")
    
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    
    print("Generating...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nResult:\n{result}")
    
    # 显存统计
    memory_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\nPeak Memory: {memory_gb:.2f} GB")


if __name__ == '__main__':
    main()