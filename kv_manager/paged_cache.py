# kv_manager/paged_cache.py
import torch
from typing import Dict, List
from .base import BaseKVCacheManager


class PagedKVCacheManager(BaseKVCacheManager):
    """分页 KV 缓存管理器 - 核心实现"""
    
    def __init__(self, num_blocks: int = 256, block_size: int = 16,
                 num_heads: int = 32, head_dim: int = 128,
                 num_layers: int = 32, device: str = 'cuda'):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.device = device
        
        # 物理块池：[num_blocks, block_size, num_heads, head_dim]
        self.key_blocks = torch.zeros(
            (num_blocks, block_size, num_heads, head_dim),
            dtype=torch.float16, device=device
        )
        self.value_blocks = torch.zeros_like(self.key_blocks)
        
        # 块状态追踪
        self.free_blocks = list(range(num_blocks))
        self.block_occupied = [0] * num_blocks  # 每个块已占用的 token 数
        
        # 请求 → 块表：request_id → [block_id, ...]
        self.block_tables: Dict[str, List[int]] = {}
        
        # 请求 → 每层的 KV 缓存（用于 HF 的 past_key_values）
        self.active_cache: Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {}
    
    def allocate(self, request_id: str, max_seq_len: int) -> None:
        """分配物理块"""
        num_blocks_needed = max_seq_len // self.block_size + 1
        
        if len(self.free_blocks) < num_blocks_needed:
            raise RuntimeError(
                f"显存不足！需要{num_blocks_needed}块，剩余{len(self.free_blocks)}块"
            )
        
        allocated = self.free_blocks[:num_blocks_needed]
        self.free_blocks = self.free_blocks[num_blocks_needed:]
        self.block_tables[request_id] = allocated
        self.active_cache[request_id] = {}
    
    def update(self, request_id: str, layer_idx: int,
               new_keys: torch.Tensor, new_values: torch.Tensor) -> None:
        """更新 KV（简化版：直接存到 active_cache）"""
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
    
    def get(self, request_id: str, layer_idx: int,
            seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取 KV"""
        if request_id not in self.active_cache:
            raise KeyError(f"Request {request_id} not found")
        
        if layer_idx not in self.active_cache[request_id]:
            raise KeyError(f"Layer {layer_idx} not found")
        
        keys, values = self.active_cache[request_id][layer_idx]
        return keys[:, :seq_len, :], values[:, :seq_len, :]
    
    def free(self, request_id: str) -> None:
        """释放缓存"""
        if request_id in self.block_tables:
            self.free_blocks.extend(self.block_tables[request_id])
            del self.block_tables[request_id]
        
        if request_id in self.active_cache:
            del self.active_cache[request_id]
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        used_blocks = self.num_blocks - len(self.free_blocks)
        return {
            'total_blocks': self.num_blocks,
            'used_blocks': used_blocks,
            'free_blocks': len(self.free_blocks),
            'memory_utilization': used_blocks / self.num_blocks * 100,
            'active_requests': len(self.active_cache),
        }