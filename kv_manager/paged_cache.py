# kv_manager/paged_cache.py
import torch
from typing import Dict, List, Tuple, Optional
from .base import BaseKVCacheManager


class PagedKVCacheManager(BaseKVCacheManager):
    """分页 KV 缓存管理器 - 真正存储 past_key_values"""
    
    def __init__(self, num_blocks: int = 256, block_size: int = 16,
                 num_layers: int = 32, device: str = 'cuda'):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.device = device
        
        # 空闲块管理
        self.free_blocks = list(range(num_blocks))
        
        # 请求 → 块表：request_id → [block_id, ...]
        self.block_tables: Dict[str, List[int]] = {}
        
        # 真正的 KV 存储：block_id → layer → (keys, values)
        # 每个 block 存储 block_size 个 token 的 KV
        self.block_storage: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = \
            {i: {} for i in range(num_blocks)}
        
        # 每个块的占用情况
        self.block_occupied: Dict[int, int] = {i: 0 for i in range(num_blocks)}
        
        # 当前活跃请求的完整 KV（用于传给 HF）
        self.active_past_key_values: Dict[str, Optional[Tuple]] = {}
    
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
        self.active_past_key_values[request_id] = None
    
    def update(self, request_id: str, new_past_key_values: Tuple) -> None:
        """
        存储新的 KV 到分页缓存
        new_past_key_values: HF 返回的 (layer_idx: (k, v)) 结构
        """
        if request_id not in self.block_tables:
            raise KeyError(f"Request {request_id} not allocated")
        
        block_ids = self.block_tables[request_id]
        
        # 遍历每一层
        for layer_idx, (new_k, new_v) in enumerate(new_past_key_values):
            # new_k shape: (batch, seq_len, num_heads, head_dim)
            batch_size, seq_len, num_heads, head_dim = new_k.shape
            
            # 将新 token 的 KV 存到对应的块
            for token_idx in range(seq_len):
                global_token_idx = self.block_occupied.get(block_ids[0], 0) + token_idx
                block_idx = global_token_idx // self.block_size
                offset_in_block = global_token_idx % self.block_size
                
                if block_idx >= len(block_ids):
                    continue  # 超出分配的块
                
                actual_block_id = block_ids[block_idx]
                
                # 初始化该层的块存储
                if layer_idx not in self.block_storage[actual_block_id]:
                    self.block_storage[actual_block_id][layer_idx] = (
                        torch.zeros(
                            (1, self.block_size, num_heads, head_dim),
                            dtype=new_k.dtype, device=self.device
                        ),
                        torch.zeros(
                            (1, self.block_size, num_heads, head_dim),
                            dtype=new_v.dtype, device=self.device
                        )
                    )
                
                # 写入单个 token 的 KV
                k_block, v_block = self.block_storage[actual_block_id][layer_idx]
                k_block[0, offset_in_block, :, :] = new_k[0, token_idx, :, :]
                v_block[0, offset_in_block, :, :] = new_v[0, token_idx, :, :]
        
        # 更新占用计数
        for block_id in block_ids:
            self.block_occupied[block_id] = min(
                self.block_occupied.get(block_id, 0) + seq_len,
                self.block_size
            )
        
        # 保存完整的 past_key_values 用于传给 HF（简化处理）
        self.active_past_key_values[request_id] = new_past_key_values
    
    def get(self, request_id: str) -> Optional[Tuple]:
        """获取当前请求的完整 past_key_values（传给 HF 用）"""
        return self.active_past_key_values.get(request_id)
    
    def free(self, request_id: str) -> None:
        """释放缓存"""
        if request_id in self.block_tables:
            block_ids = self.block_tables[request_id]
            
            # 清空块存储
            for block_id in block_ids:
                if block_id in self.block_storage:
                    self.block_storage[block_id] = {}
                self.block_occupied[block_id] = 0
            
            self.free_blocks.extend(block_ids)
            del self.block_tables[request_id]
        
        self.active_past_key_values.pop(request_id, None)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        used_blocks = self.num_blocks - len(self.free_blocks)
        return {
            'total_blocks': self.num_blocks,
            'used_blocks': used_blocks,
            'free_blocks': len(self.free_blocks),
            'memory_utilization': used_blocks / self.num_blocks * 100,
            'active_requests': len(self.active_past_key_values),
        }