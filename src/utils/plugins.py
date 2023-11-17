from __future__ import annotations
from pathlib import Path
from math import ceil
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter

from .functionals import check_file_path
from typing import TYPE_CHECKING
from typing_extensions import override, Self

if TYPE_CHECKING:
    from .modules import Trainer, NeuralNetwork, SizedDataset, Tracker, EvaluateModel

__all__ = [
    "Plugin", "LoadCheckpointPlugin", "SaveCheckpointPlugin", 
    "InitializeNetworkPlugin", "LossLoggerPlugin", "MetricLoggerPlugin",
    "EvaluatePlugin", "ProgressBarPlugin"
]


def check_plugin(plugin: Plugin) -> None:
    if isinstance(plugin, LoadCheckpointPlugin):
        if len(plugin.trainer.plugins) > 0:
            raise Exception(
                "'LoadCheckpointPlugin' typed plugin should be registered first, " 
                f"but got {len(plugin.trainer.plugins)} plugin(s) already registered."
            )


class Plugin:
    r""" Base class for all plugins.
    
    Six callback functions are defined for add functionals into six different 
    training scopes:
    - `before_loop`:  Before the first training epoch in this loop.
    - `after_loop`:   After all training epochs in this loop.
    - `before_epoch`: Before the first batch being fetched in each epoch.
    - `after_epoch`:  After all batches being processed in each epoch.
    - `before_step`:  Before Processing batches in each step.
    - `after_step`:   After updating weights and clearing their gradients in each step.
    """
    trainer: Trainer | None = None
    def before_loop(self, *args, **kwargs):
        pass 
    
    def after_loop(self, *args, **kwargs):
        pass 
    
    def before_epoch(self, *args, **kwargs):
        pass
    
    def after_epoch(self, *args, **kwargs):
        pass 
    
    def before_step(self, *args, **kwargs):
        pass 
    
    def after_step(self, *args, **kwargs):
        pass 


class InitializeNetworkPlugin(Plugin):
    def __init__(self, weight_file: Path | str | None = None) -> None:
        self.reserved_weight_file = weight_file
    
    @property
    def weight_file(self) -> Path:
        return getattr(
            self.trainer, "pretrained_weight_file", check_file_path(self.reserved_weight_file)
        )
    
    @property
    def load_pretrained_weight(self) -> bool:
        return self.weight_file is not None
    
    @override
    def before_loop(self, network: NeuralNetwork, *args, **kwargs):
        if self.load_pretrained_weight:
            network.load_state_dict(torch.load(self.weight_file))
        else:
            network.init_weights()


class LoadCheckpointPlugin(Plugin):
    def __init__(self, checkpoint_dir: Path | str) -> None:
        self.checkpoint_dir: Path | None = check_file_path(checkpoint_dir)
        self.state_files: dict[str, Path] = {}
        self.check_checkpoint_dir()
    
    def check_checkpoint_dir(self) -> None:
        if self.checkpoint_dir is None:
            raise FileNotFoundError(
                f"No such checkpoint directory: '{self.checkpoint_dir}'"
            )
        else:
            for module_name in ["trainer", "optimizer", "network"]:
                self.check_module_state_file(module_name)
    
    def check_module_state_file(self, module_name: str) -> None:
        file_name = f"{module_name}_state_dict.pth"
        state_file: Path = self.checkpoint_dir / file_name
        if state_file.exists():
            self.state_files[module_name] = state_file
        else:
            raise FileNotFoundError(
                f"No {module_name} state file found. To load a training checkpoint, "
                f"there must exist a file: '{state_file}'"
            )
    
    @override
    def before_loop(self, optimizer: torch.optim.Optimizer, *args, **kwargs):
        self.trainer.load_state_dict(torch.load(self.state_files["trainer"]))
        setattr(self.trainer, "pretrained_weight_file", self.state_files["network"])
        optimizer.load_state_dict(torch.load(self.state_files["optimizer"]))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.trainer.device)


class SaveCheckpointPlugin(Plugin):
    def __init__(self, saving_dir: Path | str, saving_period: int) -> None:
        self.saving_dir: Path = Path(saving_dir)
        self.period: int = saving_period
    
    @property
    def saving_path(self) -> Path:
        return self.saving_dir / f"epoch-{self.trainer.epoch}"
    
    @override
    def after_epoch(self, network: NeuralNetwork, optimizer: torch.optim.Optimizer, *args, **kwargs):
        if self.period == 1 or self.trainer.epoch % self.period == 0:
            network.eval()
            self.saving_path.mkdir(parents=True, exist_ok=True)
            (
                self
                .save_trainer()
                .save_network(network)
                .save_optimizer(optimizer)
            )
    
    def save_trainer(self) -> Self:
        saving_file: Path = self.saving_path / "trainer_state_dict.pth"
        torch.save(self.trainer.state_dict(), saving_file)
        return self 
    
    def save_network(self, network: NeuralNetwork) -> Self:
        saving_file: Path = self.saving_path / "network_state_dict.pth"
        torch.save(network.to("cpu").state_dict(), saving_file)
        return self 
    
    def save_optimizer(self, optimizer: torch.optim.Optimizer) -> Self:
        saving_file: Path = self.saving_path / "optimizer_state_dict.pth"
        torch.save(optimizer.state_dict(), saving_file)
        return self
        
class _TensorboardLoggerPlugin(Plugin):
    def __init__(self, log_dir: Path | str, log_period: int) -> None:
        self.log_dir = log_dir
        self.period: int = log_period
        self.tracker_names: list[str]
    
    @property
    def writer(self) -> SummaryWriter | None:
        return self.trainer.writer
    
    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
    
    def get_tracker(self, name: str) -> Tracker:
        return self.trainer.trackers[name]
    
    @override
    def before_loop(self, *args, **kwargs):
        if self.writer is None:
            self.trainer.writer = SummaryWriter(self.log_dir)
        Path(self.writer.get_logdir()).mkdir(exist_ok=True, parents=True)


class LossLoggerPlugin(_TensorboardLoggerPlugin):
    
    @property
    def tracker_names(self) -> list[str]:
        return [
            name for name in self.trainer.train_model.loss_weights
        ]
    
    @override
    def before_loop(self, dataset: SizedDataset, *args, **kwargs):
        super().before_loop(dataset, *args, **kwargs)
        self.epoch_len = len(dataset) // self.trainer.batch_size
    
    @override
    def before_step(self, *args, **kwargs):
        if self.period == 1 or self.trainer.local_step % self.period == 1:
            for name in self.tracker_names:
                self.get_tracker(name).empty()
    
    @override
    def after_step(self, *args, **kwargs):
        if self.period == 1 or self.trainer.local_step % self.period == 0:
            current_step = self.epoch_len * self.trainer._start_epoch_index + self.trainer.local_step
            total_loss = 0
            for name in self.tracker_names:
                
                tag = f"loss/{name}"
                value = self.get_tracker(name).get_value()
                step = current_step
                total_loss += value * self.trainer.train_model.loss_weights[name]
                
                self.log_scalar(tag, value, step)
            if len(self.tracker_names) > 1:
                self.log_scalar("total_loss", total_loss, step)

class MetricLoggerPlugin(_TensorboardLoggerPlugin):
    
    @property
    def tracker_names(self) -> list[str]:
        return [
            name for name in self.trainer.trackers \
                if name not in self.trainer.train_model.loss_weights
        ]
    
    @override
    def after_epoch(self, *args, **kwargs):
        if self.period == 1 or self.trainer.epoch % self.period == 0:
            for name in self.tracker_names:
                
                tag = f"metric/{name}"
                value = self.get_tracker(name).get_value()
                step = self.trainer.epoch 
                
                self.log_scalar(tag, value, step)


class EvaluatePlugin(Plugin):
    def __init__(self, eval_model: EvaluateModel, eval_period: int) -> None:
        self.eval_model = eval_model
        self.period = eval_period
    
    @override
    def before_loop(self, *args, **kwargs):
        self.trainer.register_trackers("metric", self.eval_model.metrics)
    
    @override
    def after_epoch(self, network: NeuralNetwork, *args, **kwargs):
        if self.period == 1 or self.trainer.epoch % self.period == 0:
            metric_dict: dict[str, float] = self.eval_model.evaluate(network)
            for name, metric_value in metric_dict.items():
                assert name in self.trainer.trackers, f"No such tracked scalar: {name}"
                tracker: Tracker = self.trainer.trackers[name]
                tracker.track_value(metric_value)


class _ProgressBar:
    elapse_mark = "#"
    remain_mark = "="
    
    def __init__(self, total_step: int, start_step: int = 0, length: int = 10) -> None:
        self.total_step: int = total_step
        self.start_step: int = start_step
        self._local_step: int = 0
        
        self.length = length
    
    @property
    def current_step(self) ->int:
        return self._local_step + self.start_step
    
    @property
    def elapse_len(self) -> int:
        real_total_step = self.total_step - self.start_step
        return ceil(self._local_step / real_total_step * self.length)
    
    @property
    def remain_len(self) -> int:
        return (self.length - self.elapse_len)
    
    def step(self):
        if self.current_step == self.total_step:
            self.empty()
        self._local_step += 1
        return self
    
    def empty(self):
        self._local_step = 0
        return self
    
    def __str__(self):
        num_left_blank = len(str(self.total_step)) - len(str(self.current_step))
        ratio = num_left_blank * " " + f"{self.current_step}/{self.total_step}"
        bar = self.elapse_mark * self.elapse_len + self.remain_mark * self.remain_len
        return "[" + bar + "] (" + ratio + ")" 


class ProgressBarPlugin(Plugin):
    def __init__(self, bar_length: int = 10) -> None:
        self.bar_length: int = bar_length
        
        self.epoch_bar: _ProgressBar
        self.step_bar: _ProgressBar
    
    def __str__(self) -> str:
        bar_str = f"Epoch {str(self.epoch_bar)} | Step {str(self.step_bar)}"
        return f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] " + bar_str
        
    @override
    def before_loop(self, dataset: SizedDataset, *args, **kwargs):
        self.epoch_bar = _ProgressBar(
            total_step=self.trainer.start_epoch + self.trainer.num_epochs,
            start_step=self.trainer.start_epoch,
            length=self.bar_length
        )
        self.step_bar = _ProgressBar(
            total_step=len(dataset) // self.trainer.batch_size,
            length=self.bar_length
        )
        print(self, end="\r", flush=True)
    
    @override
    def before_epoch(self, *args, **kwargs):
        self.epoch_bar.step()
        print(self, end="\r", flush=True)
    
    @override
    def before_step(self, *args, **kwargs):
        self.step_bar.step()
        print(self, end="\r", flush=True)