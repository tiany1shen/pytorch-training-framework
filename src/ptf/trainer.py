import torch
from math import ceil
from torch.utils.data import DataLoader
from typing import Optional, Any, Sequence

from model import _BaseModel
from plugin import _BasePlugin, WeightsUpdatePlugin


class _BaseTrainer:
    r"""
    
    """
    def __init__(self, hparams: dict[str, int], modules: dict[str, Any]) -> None:
        self.hparams = hparams
        self.epoch_num = hparams["epoch_num"]
        self.batch_size = hparams["batch_size"]
        self.gradient_accumulate = hparams["gradient_accumulate"]
        
        self.modelus = modules
        self.dataset = modules["dataset"]
        self.network = modules["network"]
        self.model = modules["model"]
        self.optimzer = modules["optimizer"]
        
        self.start_epoch = 0
        self.local_epoch = 0
        self.local_step = 0
        
        self.plugins = []
        
    @property
    def epoch_length(self) -> int:
        accumulated_batch_size = self.batch_size * self.gradient_accumulate
        return ceil(len(self.dataset) / accumulated_batch_size)
    
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
        loss = self.model.compute_loss(self.network, batch)
        loss.backward()
        
        self.after_step()
        
    def before_loop(self) -> None:
        pass
        
    def after_loop(self) -> None:
        self.optimzer.zero_grad()
        pass
        
    def before_epoch(self) -> None:
        pass 
    
    def after_epoch(self) -> None:
        pass
    
    def before_step(self) -> None:
        if self.local_step % self.gradient_accumulate == 1:
            self.optimzer.zero_grad()
        pass
    
    def after_step(self) -> None:
        if self.local_step % self.gradient_accumulate == 0:
            self.optimzer.step()
        pass
    
    def _get_next_batch(self) -> torch.Tensor | dict[str, torch.Tensor] | Sequence[torch.Tensor]:
        if not hasattr(self, "data_iterator"):
            self.data_iterator = iter(DataLoader(self.dataset, self.batch_size, shuffle=True, drop_last=True))

        try:
            batch = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(DataLoader(self.dataset, self.batch_size, shuffle=True, drop_last=True))
            batch = next(self.data_iterator)
        return batch
    
    def add_plugin(self, plugin: _BasePlugin):
        plugin.trainer = self
        self.plugins.append(plugin)
        
    def add_pulgins(self, plugins: list[_BasePlugin]):
        for plugin in plugins:
            self.add_plugin(plugin)


class Trainer(_BaseTrainer):
    def __init__(
        self,
        dataset:    torch.utils.data.Dataset,
        network:    torch.nn.Module,
        model:      _BaseModel,
        optimizer:  torch.optim.Optimizer,
        
        epoch_num:  int,
        batch_size: int,
        gradient_accumulate: Optional[int] = None,
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
            gradient_accumulate = 1 if gradient_accumulate is None else gradient_accumulate
        )
        
        super().__init__(hparams, modules)
        
        weight_updater = WeightsUpdatePlugin()
        self.add_plugin(weight_updater)