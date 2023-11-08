import torch
import random
from math import ceil
from torch.utils.data import DataLoader
from typing import Optional, Any, Sequence

from .model import _BaseModel
from .plugin import (
    _BasePlugin, 
    WeightsUpdatePlugin, 
    ReproduciblePlugin,
    ProgressBarPlugin
)
from .utils.typing_hints import Batch
from .utils.trackers import _BaseTracker, LossTracker


class _BaseTrainer:
    r"""
    Base class for all trainers. Only basic functionals are implemented in this class: 
        1. batch training loop 
        2. update the parameters according to gradients every several batches. 
    
    Args:
        hparams (dict): Hyper-parameters controls the training procedure, including:
            - `epoch_num` (int): number of epoch in this training procedure
            - `batch-size` (int): number of data for every batch
            - `gradient_accumulate` (int): number of batch in one update step
        modules (dict): Modules interact with this trainer, including:
            - `dataset (torch.utils.data.Dataset)`: dataset from which to load the data
            - `network` (torch.nn.Module): neural network to be trained
            - `model` (model._BaseModel): model to make use of the neural network
            - `optimizer` (torch.optim.Optimizer): optimizer to update neural network weights
    """
    def __init__(
        self, hparams: dict[str, int], modules: dict[str, Any]
    ) -> None:
        self.hparams = hparams
        self.epoch_num = hparams["epoch_num"]
        self.batch_size = hparams["batch_size"]
        self.gradient_accumulate = hparams["gradient_accumulate"]
        self.seed = hparams["seed"]
        
        self.modelus = modules
        self.dataset = modules["dataset"]
        self.network = modules["network"]
        self.model = modules["model"]
        self.optimizer = modules["optimizer"]
        
        self.start_epoch = 0
        self.local_epoch = 0
        self.local_step = 0
        self.total_loss = 0
        
        self.plugins = []
        self.loss_trackers = {}
        
    @property
    def epoch_length(self) -> int:
        loader_length = len(self.dataset) // self.batch_size
        return ceil(loader_length / self.gradient_accumulate)
    
    @property
    def epoch(self) -> int:
        return self.start_epoch + self.local_epoch
    
    @property
    def step(self) -> int:
        return self.start_epoch * self.epoch_length + self.local_step
    
    def loop(self) -> None:
        self.before_loop()
        for epoch in range(self.epoch_num):
            self.train_one_epoch()
        self.after_loop()
        self.network.eval()
        
    def train_one_epoch(self) -> None:
        self.local_epoch += 1
        self.before_epoch()
        for step in range(self.epoch_length * self.gradient_accumulate):
            self.network.train()
            self.train_one_step()
        self.after_epoch()
    
    def train_one_step(self) -> None:
        self.local_step += 1
        self.before_step()
        
        # 对一个 batch 的计算
        batch = self._get_next_batch()
        loss_dict = self.model.compute_losses(self.network, batch)
        for loss_name, loss_tensor in loss_dict.items():
            self.loss_trackers[loss_name].add(loss_tensor.item())
        total_loss = self.model.summary_losses(loss_dict)
        total_loss.backward()
        
        self.after_step()
    
    def _get_next_batch(self) -> Batch:
        try:
            batch = next(self.data_iterator)
        except (StopIteration, AttributeError):
            self.data_iterator = iter(DataLoader(
                self.dataset, 
                self.batch_size, 
                shuffle=True, drop_last=True
            ))
            batch = next(self.data_iterator)
        return batch
    
    def before_loop(self) -> None:
        for plugin in self.plugins:
            plugin.before_loop()
        
    def after_loop(self) -> None:
        for plugin in self.plugins:
            plugin.after_loop()
        
    def before_epoch(self) -> None:
        for plugin in self.plugins:
            plugin.before_epoch()
    
    def after_epoch(self) -> None:
        for plugin in self.plugins:
            plugin.after_epoch()
    
    def before_step(self) -> None:
        for plugin in self.plugins:
            plugin.before_step()
    
    def after_step(self) -> None:
        for plugin in self.plugins:
            plugin.after_step()
    
    def add_plugin(self, plugin: _BasePlugin) -> None:
        plugin.trainer = self
        if hasattr(self, "debug_mode") and self.debug_mode: plugin.debug = True
        self.plugins.append(plugin)
        
    def add_plugins(self, plugins: list[_BasePlugin]) -> None:
        for plugin in plugins:
            self.add_plugin(plugin)
    
    def add_loss_tracker(self, name: str, tracker: _BaseTracker) -> None:
        self.loss_trackers[name] = tracker


class Trainer(_BaseTrainer):
    def __init__(
        self,
        dataset:    torch.utils.data.Dataset,
        network:    torch.nn.Module,
        model:      _BaseModel,
        optimizer:  torch.optim.Optimizer,
        
        epoch_num:  int,
        batch_size: int,
        gradient_accumulate: int = 1,
        seed: Optional[int] = None,
        
        show_debug_info = False
    ) -> None:
        modules = dict(
            dataset = dataset,
            network = network,
            model = model,
            optimizer = optimizer
        )
        
        hparams = dict(
            epoch_num = epoch_num,
            batch_size = batch_size,
            gradient_accumulate = gradient_accumulate,
            seed=seed
        )
        
        super().__init__(hparams, modules)
        
        plugins = [
            ProgressBarPlugin(),
            ReproduciblePlugin(),
            WeightsUpdatePlugin()
        ]
        self.debug_mode = show_debug_info
        self.add_plugins(plugins)
        
        for loss_name in self.model.loss_weights:
            self.add_loss_tracker(loss_name, LossTracker())