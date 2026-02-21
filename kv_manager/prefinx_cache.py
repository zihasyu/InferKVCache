# kv_manager/prefix_cache.py
import hashlib
import torch
from typing import Dict, Tuple, Optional
from .base import BaseKVCacheManager


class PrefixKVCacheManager(BaseKVCacheManager):
    """前缀缓存管理器 - 真正复用公共前缀的 KV"""
    
    def __init__(self, max_prefix_len: int = 512,
                 num_layers: int = 32, device: str = 'cuda'):
        self.max_prefix_len = max_prefix_len
        self.num_layers = num_layers
        self.device = device
        
        # 前缀哈希 → past_key_values（每层的 KV）
        self.prefix_cache: Dict[str, Tuple] = {}
        
        # 请求 → 使用的前缀哈希
        self.request_prefix: Dict[str, str] = {}
        
        # 当前活跃请求的完整 KV
        self.active_past_key_values: Dict[str, Optional[Tuple]] = {}
        
        # 当前请求的输入 ID（用于计算前缀哈希）
        self.request_input_ids: Dict[str, torch.Tensor] = {}
    
    def compute_prefix_hash(self, input_ids: torch.Tensor) -> str:
        """计算输入前缀的哈希"""
        prefix_tokens = input_ids[:self.max_prefix_len].cpu().tolist()
        return hashlib.md5(str(prefix_tokens).encode()).hexdigest()
    
    def allocate(self, request_id: str, max_seq_len: int) -> None:
        self.active_past_key_values[request_id] = None
    
    def set_input_ids(self, request_id: str, input_ids: torch.Tensor) -> bool:
        """
        设置输入并尝试复用前缀
        返回：是否命中前缀缓存
        """
        self.request_input_ids[request_id] = input_ids
        prefix_hash = self.compute_prefix_hash(input_ids)
        
        if prefix_hash in self.prefix_cache:
            self.request_prefix[request_id] = prefix_hash
            # 复用前缀 KV
            self.active_past_key_values[request_id] = self.prefix_cache[prefix_hash]
            return True
        return False
    
    def update(self, request_id: str, new_past_key_values: Tuple) -> None:
        """存储新的 KV"""
        self.active_past_key_values[request_id] = new_past_key_values
    
    def save_prefix(self, request_id: str) -> None:
        """保存当前请求的前缀到共享缓存"""
        if request_id in self.request_prefix:
            prefix_hash = self.request_prefix[request_id]
            if prefix_hash not in self.prefix_cache:
                pkv = self.active_past_key_values.get(request_id)
                if pkv is not None:
                    # 只保存前缀部分的 KV
                    self.prefix_cache[prefix_hash] = tuple(
                        (
                            k[:, :self.max_prefix_len, :, :],
                            v[:, :self.max_prefix_len, :, :]
                        )
                        for k, v in pkv
                    )
    
    def get(self, request_id: str) -> Optional[Tuple]:
        """获取当前请求的 past_key_values"""
        return self.active_past_key_values.get(request_id)
    
    def free(self, request_id: str) -> None:
        # 释放前保存前缀
        self.save_prefix(request_id)
        
        self.active_past_key_values.pop(request_id, None)
        self.request_prefix.pop(request_id, None)
        self.request_input_ids.pop(request_id, None)
    
    def get_stats(self) -> Dict:
        return {
            'cached_prefixes': len(self.prefix_cache),
            'active_requests': len(self.active_past_key_values),
            'hit_rate': 0.0  # 实际使用需统计
        }