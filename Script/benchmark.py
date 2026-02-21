# scripts/benchmark.py
"""
KV Cache 策略性能对比
注意：所有策略都使用 InferKVWrapper，确保公平对比
"""
import torch
import time
import sys
sys.path.append('..')
from model_wrapper import InferKVWrapper


def benchmark_memory_and_latency(wrapper, prompt, max_new_tokens=100, num_runs=3):
    """统一测试显存和延迟"""
    torch.cuda.reset_peak_memory_stats()
    
    latencies = []
    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()
        result = wrapper.generate(prompt, max_new_tokens=max_new_tokens, request_id=f'bench_{i}')
        torch.cuda.synchronize()
        latencies.append(time.time() - start)
    
    memory_gb = torch.cuda.max_memory_allocated() / 1024**3
    avg_latency = sum(latencies) / len(latencies)
    
    # 估算生成的 token 数
    tokens = len(wrapper.tokenizer.encode(result)) - len(wrapper.tokenizer.encode(prompt))
    
    return {
        'memory_gb': memory_gb,
        'latency': avg_latency,
        'throughput': tokens / avg_latency if avg_latency > 0 else 0,
        'cache_stats': wrapper.get_stats()
    }


def main():
    model_name = '../Llama-2-7b-hf'
    prompt = "Hello, I am a software engineer working on"
    
    print("=" * 70)
    print("KV Cache 策略性能对比")
    print("=" * 70)
    
    results = []
    
    # 策略 1: Naive (不使用 kv_manager)
    print("\n[1/3] 测试 Naive (无自定义缓存)...")
    wrapper_naive = InferKVWrapper(model_name, cache_type='naive')
    stats = benchmark_memory_and_latency(wrapper_naive, prompt)
    stats['strategy'] = 'Naive'
    results.append(stats)
    print(f"  显存: {stats['memory_gb']:.2f} GB | 吞吐: {stats['throughput']:.1f} tokens/s")
    del wrapper_naive
    torch.cuda.empty_cache()
    
    # 策略 2: Paged Cache
    print("\n[2/3] 测试 Paged Cache...")
    wrapper_paged = InferKVWrapper(model_name, cache_type='paged')
    stats = benchmark_memory_and_latency(wrapper_paged, prompt)
    stats['strategy'] = 'Paged'
    results.append(stats)
    print(f"  显存: {stats['memory_gb']:.2f} GB | 吞吐: {stats['throughput']:.1f} tokens/s")
    print(f"  缓存统计: {stats['cache_stats']}")
    del wrapper_paged
    torch.cuda.empty_cache()
    
    # 策略 3: Prefix Cache
    print("\n[3/3] 测试 Prefix Cache...")
    wrapper_prefix = InferKVWrapper(model_name, cache_type='prefix')
    stats = benchmark_memory_and_latency(wrapper_prefix, prompt)
    stats['strategy'] = 'Prefix'
    results.append(stats)
    print(f"  显存: {stats['memory_gb']:.2f} GB | 吞吐: {stats['throughput']:.1f} tokens/s")
    print(f"  缓存统计: {stats['cache_stats']}")
    del wrapper_prefix
    torch.cuda.empty_cache()
    
    # 汇总表格
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)
    print(f"{'策略':<15} {'显存 (GB)':<12} {'吞吐 (tokens/s)':<18} {'延迟 (s)':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['strategy']:<15} {r['memory_gb']:<12.2f} {r['throughput']:<18.1f} {r['latency']:<12.3f}")


if __name__ == '__main__':
    main()