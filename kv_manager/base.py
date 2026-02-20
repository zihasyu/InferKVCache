# kv_manager/base.py
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import torch


class BaseKVCacheManager(ABC):
    """KV Cache 管理器统一接口"""
    
    @abstractmethod
    def allocate(self, request_id: str, max_seq_len: int) -> None:
        """为请求分配缓存空间"""
        pass
    
    @abstractmethod
    def update(self, request_id: str, layer_idx: int,
               new_keys: torch.Tensor, new_values: torch.Tensor) -> None:
        """更新 KV 缓存"""
        pass
    
    @abstractmethod
    def get(self, request_id: str, layer_idx: int,
            seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取 KV 缓存"""
        pass
    
    @abstractmethod
    def free(self, request_id: str) -> None:
        """释放请求的缓存"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict:
        """获取统计信息"""
        pass