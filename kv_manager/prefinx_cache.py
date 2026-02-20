# kv_manager/prefix_cache.py
import hashlib
import torch
from typing import Dict, Tuple, Optional
from .base import BaseKVCacheManager


class PrefixKVCacheManager(BaseKVCacheManager):
    """前缀缓存管理器 - 复用公共前缀的 KV"""
    
    def __init__(self, max_prefix_len: int = 512,
                 num_heads: int = 32, head_dim: int = 128,
                 num_layers: int = 32, device: str = 'cuda'):
        self.max_prefix_len = max_prefix_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.device = device
        
        # 前缀哈希 → (keys, values) 每层
        self.prefix_cache: Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {}
        
        # 请求 → 使用的前缀哈希
        self.request_prefix: Dict[str, str] = {}
        
        # 当前活跃请求的 KV
        self.active_cache: Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {}
    
    def compute_prefix_hash(self, input_ids: torch.Tensor) -> str:
        """计算输入前缀的哈希"""
        prefix_tokens = input_ids[:self.max_prefix_len].cpu().tolist()
        return hashlib.md5(str(prefix_tokens).encode()).hexdigest()
    
    def allocate(self, request_id: str, max_seq_len: int) -> None:
        self.active_cache[request_id] = {}
    
    def try_reuse_prefix(self, request_id: str, input_ids: torch.Tensor) -> bool:
        """尝试复用前缀缓存"""
        prefix_hash = self.compute_prefix_hash(input_ids)
        
        if prefix_hash in self.prefix_cache:
            self.request_prefix[request_id] = prefix_hash
            # 加载前缀 KV 到 active_cache
            for layer_idx in range(self.num_layers):
                if layer_idx in self.prefix_cache[prefix_hash]:
                    self.active_cache[request_id][layer_idx] = \
                        self.prefix_cache[prefix_hash][layer_idx]
            return True
        return False
    
    def update(self, request_id: str, layer_idx: int,
               new_keys: torch.Tensor, new_values: torch.Tensor) -> None:
        if request_id not in self.active_cache:
            self.active_cache[request_id] = {}
        
        if layer_idx not in self.active_cache[request_id]:
            self.active_cache[request_id][layer_idx] = (new_keys, new_values)
        else:
            old_k, old_v = self.active_cache[request_id][layer_idx]
            self.active_cache[request_id][layer_idx] = (
                torch.cat([old_k, new_keys], dim=1),
                torch.cat([old_v, new_values], dim=1)
            )
    
    def save_prefix(self, request_id: str):
        """保存当前请求的前缀到共享缓存"""
        if request_id in self.request_prefix:
            prefix_hash = self.request_prefix[request_id]
            if prefix_hash not in self.prefix_cache:
                self.prefix_cache[prefix_hash] = {}
                for layer_idx, kv in self.active_cache[request_id].items():
                    keys, values = kv
                    self.prefix_cache[prefix_hash][layer_idx] = (
                        keys[:, :self.max_prefix_len, :].clone(),
                        values[:, :self.max_prefix_len, :].clone()
                    )
    
    def get(self, request_id: str, layer_idx: int,
            seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if request_id not in self.active_cache:
            raise KeyError(f"Request {request_id} not found")
        
        if layer_idx not in self.active_cache[request_id]:
            raise KeyError(f"Layer {layer_idx} not found")
        
        keys, values = self.active_cache[request_id][layer_idx]
        return keys[:, :seq_len, :], values[:, :seq_len, :]
    
    def free(self, request_id: str) -> None:
        # 释放前保存前缀
        self.save_prefix(request_id)
        
        self.active_cache.pop(request_id, None)
        self.request_prefix.pop(request_id, None)
    
    def get_stats(self) -> Dict:
        return {
            'cached_prefixes': len(self.prefix_cache),
            'active_requests': len(self.active_cache),
        }