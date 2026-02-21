# model_wrapper.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from kv_manager.paged_cache import PagedKVCacheManager
from kv_manager.prefix_cache import PrefixKVCacheManager
import torch
from typing import Optional


class InferKVWrapper:
    """
    包装 HF 模型，真正使用自定义 KV Cache
    关键：自己实现生成循环，不用 model.generate()
    """
    
    def __init__(self, model_name: str, cache_type: str = 'paged'):
        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        
        # 获取模型配置
        config = self.model.config
        num_layers = config.num_hidden_layers
        
        # 初始化 KV Cache 管理器
        if cache_type == 'paged':
            self.kv_manager = PagedKVCacheManager(
                num_blocks=256,
                block_size=16,
                num_layers=num_layers
            )
        elif cache_type == 'prefix':
            self.kv_manager = PrefixKVCacheManager(
                max_prefix_len=512,
                num_layers=num_layers
            )
        else:
            self.kv_manager = None  # naive 模式
        
        self.cache_type = cache_type
        print(f"Model loaded with {cache_type} cache")
    
    def generate(self, prompt: str, max_new_tokens: int = 100,
                 request_id: str = 'default') -> str:
        """
        自定义生成循环 - 真正使用 kv_manager
        这是关键！不用 model.generate()，自己控制每一步
        """
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # 分配缓存
        if self.kv_manager:
            self.kv_manager.allocate(request_id, len(prompt) + max_new_tokens)
            
            # Prefix Cache 需要设置 input_ids 来匹配前缀
            if self.cache_type == 'prefix':
                self.kv_manager.set_input_ids(request_id, input_ids[0])
        
        generated_ids = input_ids.clone()
        past_key_values = None
        
        # 自回归生成循环（关键！）
        for i in range(max_new_tokens):
            with torch.no_grad():
                # 从 kv_manager 获取历史 KV（如果有）
                if self.kv_manager:
                    cached_pkv = self.kv_manager.get(request_id)
                    if cached_pkv is not None:
                        past_key_values = cached_pkv
                
                # 调用 model.forward（不是 generate！）
                outputs = self.model(
                    input_ids=generated_ids if i == 0 else generated_ids[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True
                )
            
            # 更新 KV 缓存到 kv_manager
            past_key_values = outputs.past_key_values
            if self.kv_manager:
                self.kv_manager.update(request_id, past_key_values)
            
            # 采样下一个 token
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1)
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
            
            # 更新 attention_mask
            attention_mask = torch.cat([
                attention_mask,
                torch.ones_like(next_token).unsqueeze(0)
            ], dim=1)
            
            # 检查是否生成结束符
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        # 释放缓存
        if self.kv_manager:
            self.kv_manager.free(request_id)
        
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    def get_stats(self):
        if self.kv_manager:
            return self.kv_manager.get_stats()
        return {'cache_type': 'naive'}