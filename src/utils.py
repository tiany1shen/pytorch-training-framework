import torch
import pathlib
from .typing_hints import Batch

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


def move_batch(batch: Batch, device: torch.device) -> Batch:
    r""" Move a tensor batch to a specific device.
    
    The batch should be among the following forms:
    - `torch.Tensor`
    - `list[torch.Tensor]`
    - `dict[str, torch.Tensor]`
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, list):
        return [tensor.to(device) for tensor in batch]
    elif isinstance(batch, dict):
        return {key: batch[key].to(device) for key in batch}
    else:
        raise TypeError(
            f"A batch should be of type `torch.Tensor`, `list` or `dict`, but "
            f"got {type(batch)}"
        )