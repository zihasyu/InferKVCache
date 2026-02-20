# scripts/benchmark.py
"""统一性能评测脚本"""
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.append('..')
from model_wrapper import InferKVWrapper


def benchmark_memory(model, tokenizer, prompt, max_new_tokens=100):
    """测试显存峰值"""
    torch.cuda.reset_peak_memory_stats()
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
    
    return torch.cuda.max_memory_allocated() / 1024**3


def benchmark_throughput(model, tokenizer, prompt, max_new_tokens=100):
    """测试吞吐量"""
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
    end = time.time()
    
    tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    return tokens / (end - start)


def benchmark_latency(model, tokenizer, prompt, max_new_tokens=100, num_runs=3):
    """测试平均延迟"""
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    latencies = []
    
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
        torch.cuda.synchronize()
        latencies.append(time.time() - start)
    
    return sum(latencies) / len(latencies)


def main():
    model_name = '../Llama-2-7b-hf'
    prompt = "Hello, I am a software engineer working on"
    
    print("=" * 50)
    print("KV Cache 策略性能对比")
    print("=" * 50)
    
    strategies = [
        ('naive', 'HF 原生 Cache'),
        ('paged', '分页 Cache'),
        ('prefix', '前缀 Cache'),
    ]
    
    results = []
    
    for cache_type, name in strategies:
        print(f"\n测试 {name}...")
        
        if cache_type == 'naive':
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map='auto'
            )
        else:
            wrapper = InferKVWrapper(model_name, cache_type=cache_type)
            model = wrapper.model
            tokenizer = wrapper.tokenizer
        
        memory = benchmark_memory(model, tokenizer, prompt)
        throughput = benchmark_throughput(model, tokenizer, prompt)
        latency = benchmark_latency(model, tokenizer, prompt)
        
        results.append({
            'strategy': name,
            'memory_gb': memory,
            'throughput': throughput,
            'latency': latency
        })
        
        print(f"  显存: {memory:.2f} GB")
        print(f"  吞吐: {throughput:.1f} tokens/s")
        print(f"  延迟: {latency:.3f} s")
        
        # 清理显存
        del model
        torch.cuda.empty_cache()
    
    # 汇总表格
    print("\n" + "=" * 50)
    print("结果汇总")
    print("=" * 50)
    print(f"{'策略':<15} {'显存(GB)':<12} {'吞吐(tokens/s)':<18} {'延迟(s)':<10}")
    print("-" * 50)
    for r in results:
        print(f"{r['strategy']:<15} {r['memory_gb']:<12.2f} {r['throughput']:<18.1f} {r['latency']:<10.3f}")


if __name__ == '__main__':
    main()