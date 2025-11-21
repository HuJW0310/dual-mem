import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from sklearn.cluster import KMeans


@dataclass
class ExplicitMemoryManager:
    schema: Dict[str, Any] = field(default_factory=lambda: {
        "rooms": [],
        "objects": [],
        "connections": []
    })

    def to_json_str(self) -> str:
        return json.dumps(self.schema, indent=2)

    def load_from_json_str(self, s: str):
        try:
            self.schema = json.loads(s)
        except json.JSONDecodeError:
            # 如果 JSON 有问题，可以保留旧的 schema
            pass


@dataclass
class MemoryItem:
    global_feature: torch.Tensor       # (D,)
    index: int                         # 第几个窗口
    timestamp: float                   # 若无真实时间，用 index * window_stride
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ImplicitMemoryBank:
    items: List[MemoryItem] = field(default_factory=list)
    cluster_centers: Optional[torch.Tensor] = None  # (K, D) 聚类中心
    n_clusters: int = 10  # 默认聚类数量
    
    def append(self, feature: torch.Tensor, index: int, timestamp: float, meta=None):
        """添加新的memory item"""
        if meta is None:
            meta = {}
        self.items.append(MemoryItem(feature.detach().cpu(), index, timestamp, meta))
        # 添加新item后，重置聚类中心，需要重新聚类
        self.cluster_centers = None

    def as_matrix(self) -> torch.Tensor:
        """将所有memory items的特征堆叠成矩阵"""
        if not self.items:
            return torch.empty(0, 0)
        feats = [it.global_feature for it in self.items]
        return torch.stack(feats, dim=0)   # (N, D)
    
    def cluster(self, n_clusters: Optional[int] = None) -> torch.Tensor:
        """
        对implicit memory进行聚类，返回聚类中心作为片段全局特征
        Args:
            n_clusters: 聚类数量，如果为None则使用self.n_clusters
        Returns:
            cluster_centers: (K, D) 聚类中心tensor
        """
        if not self.items:
            return torch.empty(0, 0)
        
        if n_clusters is None:
            n_clusters = self.n_clusters
        
        # 获取所有特征
        feats = self.as_matrix()  # (N, D)
        
        # 如果items数量少于聚类数，直接返回所有特征
        if len(self.items) <= n_clusters:
            self.cluster_centers = feats
            return feats
        
        # 转换为numpy进行聚类
        feats_np = feats.numpy()
        
        # 使用KMeans聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(feats_np)
        
        # 将聚类中心转换回tensor
        self.cluster_centers = torch.from_numpy(kmeans.cluster_centers_).float()
        
        return self.cluster_centers
    
    def get_cluster_centers(self, force_recluster: bool = False) -> torch.Tensor:
        """
        获取聚类中心，如果还没有聚类或force_recluster=True，则先进行聚类
        """
        if self.cluster_centers is None or force_recluster:
            return self.cluster()
        return self.cluster_centers
    
    def get_size(self) -> int:
        """返回当前memory bank中item的数量"""
        return len(self.items)
