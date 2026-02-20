# model_wrapper.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from kv_manager.paged_cache import PagedKVCacheManager
from kv_manager.prefix_cache import PrefixKVCacheManager
import torch
from typing import Optional


class InferKVWrapper:
    """
    包装 HF 模型，注入自定义 KV Cache 管理
    不修改任何 HF 源码！
    """
    
    def __init__(self, model_name: str, cache_type: str = 'paged'):
        """
        Args:
            model_name: HF 模型路径或名称
            cache_type: 'paged' | 'prefix' | 'naive'
        """
        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        
        # 获取模型配置
        config = self.model.config
        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        num_layers = config.num_hidden_layers
        
        # 初始化 KV Cache 管理器
        if cache_type == 'paged':
            self.kv_manager = PagedKVCacheManager(
                num_blocks=256,
                block_size=16,
                num_heads=num_heads,
                head_dim=head_dim,
                num_layers=num_layers
            )
        elif cache_type == 'prefix':
            self.kv_manager = PrefixKVCacheManager(
                max_prefix_len=512,
                num_heads=num_heads,
                head_dim=head_dim,
                num_layers=num_layers
            )
        else:
            self.kv_manager = None  # 使用 HF 原生 cache
        
        self.cache_type = cache_type
        print(f"Model loaded with {cache_type} cache")
    
    def generate(self, prompt: str, max_new_tokens: int = 100,
                 request_id: str = 'default') -> str:
        """自定义生成流程"""
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        input_len = inputs['input_ids'].shape[1]
        
        # 分配缓存
        if self.kv_manager:
            self.kv_manager.allocate(request_id, input_len + max_new_tokens)
        
        # 使用 HF 原生 generate
        # 注意：这里简化处理，实际深度集成需要 hook model.forward()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # 释放缓存
        if self.kv_manager:
            self.kv_manager.free(request_id)
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def get_stats(self):
        if self.kv_manager:
            return self.kv_manager.get_stats()
        return {'cache_type': 'naive'}