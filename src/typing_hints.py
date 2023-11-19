from __future__ import annotations
import torch
from typing import TypeVar, TypeAlias, TypedDict, Sequence, Mapping, TYPE_CHECKING


if TYPE_CHECKING:
    from .modules import SizedDataset, NeuralNetwork

__all__ = [
    "ScalarType", "DataType", "DeviceType", "TrainerStateDict"
]

r"""
#* Tensor Types

#? ScalarTensor: tensor that can call its `backward()` method
We do not really check the dimension of a scalar tensor typed tensor, only de-
clare a type alias of :type Tensor: here for distinguishment of scalar tensors 
and ordinary tensors. Typically, loss value should be `ScalarTensor` type.

#? BatchedTensor: tensor type owning a batch dimension at dimension 0.
We do not really check the dimension of a batched-tensor typed tensor, only de-
clare a type alias of :type Tensor: here for distinguishment of batched-tensors 
and ordinary tensors

#? Batch: tensor batch fetched from any DataLoader
Three typical types of batch are legitimate in this framework:
- an isolated batched tensor
- a sequence (list, tuple) of batched tensor
- a mapping (dictionary) of batched tensor
"""
ScalarType: TypeAlias = torch.Tensor
DataType = TypeVar("DataType", torch.Tensor, Sequence[torch.Tensor], Mapping[str, torch.Tensor])

DeviceType: TypeAlias = torch.device | int | str

class TrainerStateDict(TypedDict):
    seed: int
    epoch: int
