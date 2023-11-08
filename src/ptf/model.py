import torch
from typing import TypeAlias

from .utils.typing_hints import ScalarTensor, Batch


#* Type Alias
Network: TypeAlias = torch.nn.Module
LossDict: TypeAlias = dict[str, ScalarTensor]

class _BaseModel:
    loss_weights: dict[str, float] = {}
    
    def compute_losses(self, network: Network, batch: Batch) -> LossDict:
        raise NotImplementedError
    
    def summary_losses(self, loss_dict: LossDict) -> ScalarTensor:
        total_loss = 0
        for loss_name in loss_dict:
            total_loss += self.loss_weights[loss_name] * loss_dict[loss_name]
        return total_loss