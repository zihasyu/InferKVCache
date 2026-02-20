# model_wrapper.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from kv_manager.paged_cache import PagedKVCacheManager
import torch

class InferKVWrapper:
    """包装 HF 模型，不修改任何 HF 源码"""
    
    def __init__(self, model_name: str):
        # 1. 加载 HF 模型（标准方式）
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        
        # 2. 初始化你的 KV Cache 管理器
        config = self.model.config
        self.kv_manager = PagedKVCacheManager(
            num_blocks=256,
            block_size=16,
            num_heads=config.num_attention_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            num_layers=config.num_hidden_layers
        )
    
    def generate(self, prompt: str, max_new_tokens: int = 100, 
                 request_id: str = 'default') -> str:
        """自定义生成流程"""
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        
        # 3. 分配 KV 缓存
        self.kv_manager.allocate(request_id, max_seq_len=len(prompt) + max_new_tokens)
        
        # 4. 使用 HF 原生 generate，但通过 hook 管理 KV
        # 关键：use_cache=True 让 HF 返回 past_key_values
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,  # ← 启用 HF 的 KV Cache
            pad_token_id=self.tokenizer.eos_token_id,
            # 可以在这里通过 logits_processor 等 hook 注入自定义逻辑
        )
        
        # 5. 释放缓存
        self.kv_manager.free(request_id)
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def get_stats(self):
        return self.kv_manager.get_stats()