import torch
import pathlib
from typing import cast
from .typing_hints import DataType

__all__ = [
    "check_file_path", "move_batch"
]

def check_file_path(path: pathlib.Path | str) -> pathlib.Path:
    r""" Check and return the given path.
    
    If the given path exists, return a `pathlib.Path` object of it.
    """
    _path = pathlib.Path(path)
    if _path.exists():
        return _path
    else:
        raise FileNotFoundError(
            f"No such file: '{_path}'"
        )

def move_batch(batch: DataType, device: torch.device) -> DataType:
    r""" Move a tensor batch to a specific device.
    
    The batch should be among the following forms:
    - `torch.Tensor`
    - `list[torch.Tensor]`
    - `dict[str, torch.Tensor]`
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    
    elif isinstance(batch, list):
        new_list: list[torch.Tensor] = []
        for tensor in batch:
            new_list.append(tensor.to(device))
        return cast(DataType, new_list)
    
    elif isinstance(batch, dict):
        new_dict: dict[str, torch.Tensor] = {}
        for key, tensor in batch.items():
            new_dict[key] = tensor.to(device)
        return cast(DataType, new_dict)
    
    else:
        raise TypeError(
            f"A batch should be of type `torch.Tensor`, `list` or `dict`, but "
            f"got {type(batch)}"
        )

def reset_parameters_data(module: torch.nn.Module) -> None:
    if hasattr(module, "reset_parameters"):
        module.reset_parameters() # type: ignore
        return 
    for child_module in module.children():
        reset_parameters_data(child_module)