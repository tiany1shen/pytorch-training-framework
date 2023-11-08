import torch
from typing import TypeAlias, TypedDict
from collections.abc import Sequence, Mapping


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
ScalarTensor: TypeAlias = torch.Tensor
BatchedTensor: TypeAlias = torch.Tensor
Batch: TypeAlias = BatchedTensor | Sequence[BatchedTensor] | Mapping[BatchedTensor]


#* Metric State Types

MetricCache: TypeAlias = float | Sequence[float] | None

class MetricStateDict(TypedDict):
    r"""
    metric tracker state dictionary type. 
    :key cache (float, list[float]): metric value tracked during training 
    :key count (int): the counting of all data received
    """
    cache: MetricCache
    count: int
