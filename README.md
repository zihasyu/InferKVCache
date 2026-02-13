# InferKVCache

KV Cache 策略复现
本项目旨在基于 Hugging Face Transformers 原生框架，在 Llama-2-7B（或 Qwen-7B）上复现并对比多种 KV Cache 优化策略。所有实验均在 无 vLLM / TensorRT-LLM 等高级引擎干扰 的环境下进行，确保公平性与可解释性。
✅ 实验目标
复现 朴素 KV Cache（HF 原生 baseline）
实现 Prefix Caching（跨请求共享公共前缀）
实现 CPU Offload（KCache 风格显存卸载）
实现 Sliding Window Attention（有限上下文窗口）
（可选）实现 简化版 PagedAttention
📁 项目结构

.
├── transformers-5.0.0/        # 修改后的 HF Transformers 源码（可编辑安装）
├── Llama-2-7b-hf/             # 模型权重（需手动下载，已加入 .gitignore）
├── kv_manager/
│   ├── __init__.py
│   ├── base.py                # BaseKVCacheManager 接口
│   ├── prefix_cache.py        # Prefix Caching 实现
│   ├── offload_cache.py       # CPU Offload 实现
│   └── sliding_window.py      # Sliding Window 实现
├── scripts/
│   ├── baseline.py            # Baseline 推理脚本
│   ├── test_prefix.py         # Prefix Caching 测试
│   └── benchmark.py           # 统一性能评测
├── results/
│   └── comparison_table.md    # 实验结果汇总（显存、吞吐、正确性）
├── .gitignore                 # 忽略模型、缓存、虚拟环境等
└── README.md                  # 本文件
🚀 快速开始
1. 环境准备


编辑



# 创建虚拟环境（推荐）
python -m venv kv-cache-env
source kv-cache-env/bin/activate  # Linux/macOS
# kv-cache-env\Scripts\activate  # Windows

# 下载最新 HF Transformers（无 Git 历史）
svn export https://github.com/huggingface/transformers/trunk transformers-dev

# 可编辑安装
cd transformers-dev && pip install -e ".[dev]" && cd ..
2. 下载模型（需 Hugging Face 账号）
bash

编辑



huggingface-cli login
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir ./Llama-2-7b-hf
💡 或使用免授权的 Qwen-7B：
python

编辑



model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)
3. 运行 Baseline
bash

编辑



python scripts/baseline.py
4. 运行自定义策略
bash

编辑



python scripts/test_prefix.py  # 示例：Prefix Caching
📊 评估指标
每种策略将从以下维度进行评测：
表格
指标	说明
显存峰值 (GB)	torch.cuda.max_memory_allocated()
吞吐量 (tokens/s)	生成 token 数 / 总耗时
正确性	输出是否与 baseline 一致（对确定性 prompt）
支持长上下文？	能否处理 >4k tokens 的输入
跨请求共享？	是否允许多个请求复用相同前缀的 KV
📝 当前进展
表格
策略	状态	备注
Baseline (Naive KV)	✅ 完成	HF 原生 past_key_values
Prefix Caching	⏳ 开发中	基于 token 序列哈希匹配
CPU Offload	🚧 待开始	使用 pinned CPU memory
Sliding Window	🚧 待开始	固定窗口大小 N=2048
PagedAttention (简化)	❌ 未计划	复杂度高，优先级低
📚 参考文献
vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention
KCache: CPU-GPU Unified KV Cache for Long Context LLM Inference
Mistral 7B Technical Report（Sliding Window）
Hugging Face Transformers Source Code
🛑 注意事项
不要提交模型权重：Llama-2-7b-hf/ 已加入 .gitignore
仅修改 transformers-dev/src/transformers/models/llama/modeling_llama.py
所有自定义逻辑应封装在 kv_manager/ 中，便于替换与测试