import torch
from typing import TypeAlias

from .utils.typing_hints import ScalarTensor, Batch


#* Type Alias
Dataset: TypeAlias = torch.utils.data.Dataset
Network: TypeAlias = torch.nn.Module
LossDict: TypeAlias = dict[str, ScalarTensor]
MetricDict: TypeAlias = dict[str, float]

class _BaseModel:
    loss_weights: dict[str, float] = {}
    evaluate_metrics: list[str] = []
    
    def compute_losses(self, network: Network, batch: Batch) -> LossDict:
        raise NotImplementedError
    
    def summary_losses(self, loss_dict: LossDict) -> ScalarTensor:
        total_loss = 0
        for loss_name in loss_dict:
            total_loss += self.loss_weights[loss_name] * loss_dict[loss_name]
        return total_loss
    
    def init_weights(self, network):
        raise NotImplementedError
    
    def evaluate(self, network: Network, dataset: Dataset) -> MetricDict:
        return {}