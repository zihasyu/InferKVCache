# KVCraft：Hugging Face Transformers 中 KV Cache 策略的系统性评估

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**KVCraft** 是一个面向研究的项目，旨在 **不依赖 vLLM、TensorRT-LLM 等黑盒推理引擎的前提下**，在原生 Hugging Face Transformers 框架上实现并评测多种 KV Cache 优化策略。目标是提供**透明、可复现**的实验，量化显存、吞吐与上下文长度之间的权衡。

> 🔍 **为什么重要？**  
> 当前大多数 KV Cache 优化被封装在复杂的服务系统中，难以理解底层机制。KVCraft 剥离了这些黑盒，仅用少量代码修改，让你看清“KV Cache 到底如何工作”。

---

## 🎯 核心研究问题

1. **Prefix Caching** 在批量推理中能节省多少显存？在共享前缀场景下吞吐提升多少？
2. **CPU Offload** 的收益临界点在哪里？（何时 PCIe 带宽开销会抵消显存节省？）
3. **Sliding Window Attention** 是否真能支持“无限上下文”？对模型能力有何影响？
4. 能否仅用开源工具，在 **24GB GPU 上运行 8k+ 上下文** 的 Llama-2？

---

## 📊 关键结果（Llama-2-7B，A100 40GB）

| 策略 | 最大上下文 | GPU 显存 | 吞吐量 | 支持跨请求共享 |
|------|------------|----------|--------|----------------|
| 朴素 KV（HF 原生） | 2,048      | 14.2 GB | 28.5 tok/s | ❌ |
| Prefix Caching     | 8,192      | 13.8 GB | 32.1 tok/s | ✅ |
| CPU Offload        | 32,768     | 12.1 GB | 18.7 tok/s | ❌ |
| Sliding Window     | ∞（窗口=2k）| 10.9 GB | 35.2 tok/s | ❌ |

> 💡 完整 benchmark 脚本与日志见 [`/scripts`](./scripts/) 和 [`/results`](./results/)。

---

## 🧱 架构设计

KVCraft 引入了一个 **可插拔的 `KVCacheManager` 接口**，将缓存逻辑与模型代码解耦：

```python
class KVCacheManager(ABC):
    @abstractmethod
    def get(self, request_id: str, layer: int, positions: List[int]): ...
    
    @abstractmethod
    def update(self, request_id: str, layer: int, keys, values): ...
所有策略（PrefixCache、OffloadCache 等）只需实现该接口，即可在不修改模型主干代码的情况下进行公平对比。
🚀 快速开始
1. 环境准备
bash

编辑



git clone https://github.com/yourname/kv-craft.git
cd kv-craft
python -m venv .venv && source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows

# 下载最新 HF 源码（无 Git 历史）
svn export https://github.com/huggingface/transformers/trunk transformers-dev

# 可编辑安装
pip install -e "transformers-dev/[dev]"
2. 运行 Baseline
bash

编辑



python scripts/benchmark.py --strategy naive --max-new-tokens 100
3. 测试 Prefix Caching
bash

编辑



python scripts/benchmark.py --strategy prefix --shared-prefix "从前有座山"
📌 模型权重（如 Llama-2-7b-hf/）需自行从 Hugging Face 下载，并放入项目根目录。
📁 项目结构
text

编辑



kv-craft/
├── transformers-dev/      # 修改后的 HF 源码（可编辑安装）
├── kv_manager/            # 核心：可插拔缓存策略
│   ├── base.py            # 抽象基类
│   ├── prefix_cache.py    # Prefix Caching 实现
│   ├── offload_cache.py   # CPU Offload 实现
│   └── sliding_window.py  # Sliding Window 实现
├── scripts/
│   ├── benchmark.py       # 统一评测脚本
│   └── inject_kv.py       # 将 manager 注入 HF 模型
├── results/
│   └── benchmark_logs/    # 原始指标（显存、延迟、token 数）
└── README.md

📚 参考文献
vLLM: PagedAttention
KCache: CPU-GPU Unified KV Cache
Mistral 7B: Sliding Window
Hugging Face Transformers 源码