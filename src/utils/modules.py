import os
import random
import numpy
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Module
from torch.optim import Optimizer
from pathlib import Path

from .functionals import move_batch
from .plugins import Plugin

from typing import cast, Callable, Literal, Iterable
from typing_extensions import Self, override
from .typing_hints import Batch, ScalarTensor, DeviceType, TrainerStateDict, TrainModuleDict

__all__ = [
    "SizedDataset", "NeuralNetwork", "TrainModel", "EvaluateModel",
    "LossTracker", "MetricTracker", "Seeder", "Trainer"
]


#===============================================================================
#   Dataset
#===============================================================================

class SizedDataset(Dataset):
    r""" Base class for finite mapping dataset.
    
    `torch.utils.data.Dataset` class with a `__len__` method.
    """
    def __len__(self):
        raise NotImplementedError


#===============================================================================
#   Neural Network
#===============================================================================

class NeuralNetwork(Module):
    r""" Base class for deep neural network class.
    
    `torch.nn.Module` with a `init_weights` method to initialize parametric
    weights, which should directly change the weight data instead of re-targeting 
    parameter variable pointers.
    """
    def __init__(self) -> None:
        super().__init__()
    
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    def init_weights(self):
        raise NotImplementedError
    

#===============================================================================
#   Models
#===============================================================================


class _Model:
    required: list[str]
    
    def __init__(self) -> None:
        self.check_required()
    
    def check_required(self) -> None:
        for name in self.required:
            if not hasattr(self, name):
                raise AttributeError(
                    f"Missing required '{name}' attribute."
                )


class TrainModel(_Model):
    r""" Base class for training model class.
    
    `TrainingModel` classes are used to manage loss functions for training deep
    neural networks. To inherent this class, two methods must be overrided:
    - `loss_weights`: Dictionary of forms {name: weighted hyper-parameter}
    - `compute_losses`: To compute loss dictionary of forms {name: scalar tensor}
    """
    required: list[str] = ["_loss_weights", "compute_loss"]
    
    @property
    def loss_weights(self) -> dict[str, float]:
        return getattr(self, "_loss_weights")
    
    def compute_loss(self, network: NeuralNetwork, batch: Batch) -> dict[str, ScalarTensor]:
        # return the loss dict computed from the given batch.
        raise NotImplementedError
    
    def summary_loss(self, loss_dict: dict[str, ScalarTensor]) -> ScalarTensor:
        total_loss: ScalarTensor = cast(ScalarTensor, 0)
        for loss_name in loss_dict:
            total_loss += self.loss_weights[loss_name] * loss_dict[loss_name]
        return total_loss


class EvaluateModel(_Model):
    r""" Base class for evaluating model class.
    
    `EvaluateModel` class are used to manage evaluating metrics. To inherent this 
    class, an `evaluate` method must be overrided.
    """
    required: list[str] = ["_metrics", "evaluate"]
    
    @property
    def metrics(self) -> list[str]:
        return getattr(self, "_metrics")
    
    def evaluate(self, network: NeuralNetwork):
        raise NotImplementedError


#===============================================================================
#   Scalar Trackers
#===============================================================================


class _Tracker:
    def __init__(self, name: str = "default-float-tracker", max_len: int = 1_000):
        self.name = name
        self.cache: list[float] = []
        self.max_len = max_len
    
    def track_value(self, value: float) -> Self:
        if len(self.cache) == self.max_len:
            self.empty()
        self.cache.append(value)
        return self 
    
    @property
    def is_empty(self) -> bool:
        return len(self.cache) == 0
    
    def empty(self) -> Self:
        self.cache = []
        return self 
    
    def get_value(self, smooth_fn: Callable[[list[float]], float] = lambda x: x[-1]) -> float:
        if self.is_empty:
            raise IndexError(f"Try to access values from an empty tracker: {self.name}")
        return smooth_fn(self.cache)


class MetricTracker(_Tracker): ...


class LossTracker(_Tracker):
    @override
    def get_value(self, smooth_fn = lambda x: sum(x) / len(x)) -> float:
        return super().get_value(smooth_fn)

#===============================================================================
#   Random Seed
#===============================================================================

class Seeder:
    r""" Class to manage epoch-level random seed. 
    
    """
    low: int = 0
    high: int = 2 ** 32 - 1
    
    def __init__(self, init_seed: int | None = None) -> None:
        if init_seed is None:
            init_seed = self.generate_seed()
        self._random_seed: int = init_seed
        self.manual_seed()
        self._next_seed: int = self.generate_seed()
    
    def get_seed(self) -> int:
        return self._random_seed
    
    def update(self) -> Self:
        self._random_seed = self._next_seed
        self.manual_seed()
        self._next_seed = self.generate_seed()
        return self
    
    def generate_seed(self) -> int:
        return int(torch.randint(self.low, self.high, (1,)).item())
    
    def manual_seed(self) -> Self:
        seed = self.get_seed()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        numpy.random.seed(seed)
        random.seed(seed)
        return self


#===============================================================================
#   Trainer
#===============================================================================


class Trainer:
    r""" Base class for trainers. 
    
    Call `loop` method to conduct a training procedure.
    """
    def __init__(
        self, 
        train_model: TrainModel,
        *,
        num_epochs: int,
        batch_size: int,
        gradient_accumulation_step: int = 1,
        init_seed: int | None = None,
        device: DeviceType = "cpu",
        
        use_deterministic: bool = False
        
    ) -> None:
        
        self.train_model: TrainModel = train_model
        
        self._start_epoch_index: int = 0
        self._local_epoch_index: int = 0
        self._local_step_in_epoch_index: int = 0
        self._local_step_in_train_index: int = 0
        self._num_epochs: int = num_epochs
        
        self.batch_size: int = batch_size
        self.gradient_accumulation_step: int = gradient_accumulation_step
        self.seeder: Seeder = Seeder(init_seed)
        
        self.device: torch.device = torch.device(device)
        if use_deterministic and self.device != torch.device("cpu"):
            self.use_deterministic_algorithms()
        
        self.plugins: list[Plugin] = []
        self.trackers: dict[str: _Tracker] = {}
        self.register_trackers("loss", self.train_model.loss_weights)
        self.writer: SummaryWriter | None = None
    
    @property
    def is_epoch_based(self) -> bool:
        return hasattr(self, "_num_epochs")
    
    @property
    def start_epoch(self) -> int:
        return self._start_epoch_index
    
    @property
    def epoch(self) -> int:
        return self._start_epoch_index + self._local_epoch_index
    
    @property
    def step_in_epoch(self) -> int:
        return self._local_in_epoch_step_index
    
    @property
    def local_step(self) -> int:
        return self._local_step_in_train_index
    
    @property
    def num_epochs(self) -> int:
        return self._num_epochs
    
    @property 
    def seed(self) -> int:
        return self.seeder.get_seed()
    
    def state_dict(self) -> TrainerStateDict:
        return {"epoch": self.epoch, "seed": self.seed}
    
    def load_state_dict(self, trainer_state_dict: TrainerStateDict) -> Self:
        self._start_epoch_index = trainer_state_dict["epoch"]
        self.seeder = Seeder(trainer_state_dict["seed"])
        return self
    
    def use_deterministic_algorithms(self) -> Self:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        if torch.version.cuda >= "10.2":
            os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8' # cuda >= 10.2
        return self
    
    def register_trackers(self, type: Literal["loss", "metric"], names: Iterable[str]) -> Self:
        TrackerClass = LossTracker if type == "loss" else MetricTracker
        for name in names:
            self.register_tracker(name, TrackerClass(f"{name}-tracker"))
        return self 
    
    def register_tracker(self, name: str, tracker: _Tracker) -> Self:
        self.trackers[name] = tracker
        return self
    
    def track_tensor_dict(self, tensor_dict: dict[str: ScalarTensor]) -> Self:
        for name, scalar_tensor in tensor_dict.items():
            assert name in self.trackers, f"No such tracked scalar: {name}"
            tracker: _Tracker = self.trackers[name]
            tracker.track_value(scalar_tensor.item())
        return self
        
    def loop(self, dataset: SizedDataset, network: NeuralNetwork, optimizer: Optimizer) -> None:
        if self.is_epoch_based:
            self.train_epoch_loop(dataset, network, optimizer)
    
    def build_dataloader(self, dataset: SizedDataset, num_workers: int = 4) -> DataLoader:
        new_batch_size: int = self.batch_size // self.gradient_accumulation_step
        self.batch_size = new_batch_size * self.gradient_accumulation_step
        new_num_samples: int = len(dataset) // self.batch_size * self.batch_size
        
        sampler = RandomSampler(
            dataset, num_samples=new_num_samples
        )
        return DataLoader(dataset, new_batch_size, sampler=sampler, num_workers=num_workers)
    
    def train_epoch_loop(
        self, 
        dataset: SizedDataset, 
        network: NeuralNetwork, 
        optimizer: Optimizer
    ) -> None:
        dataloader = self.build_dataloader(dataset)
        train_modules: TrainModuleDict = {
            "dataset": dataset, "network": network, "optimizer": optimizer
        }
        self.before_loop(train_modules)
        for local_epoch in range(self.num_epochs):
            self._local_epoch_index += 1
            self._local_step_in_epoch_index = 0
            self.seeder.update()
            self.before_epoch(train_modules)
            
            if network.device != self.device:
                network.to(self.device)
            network.train()
            for local_step, batch in enumerate(dataloader):
                if local_step % self.gradient_accumulation_step == 0:
                    self._local_step_in_epoch_index += 1
                    self._local_step_in_train_index += 1
                    self.before_step(train_modules)
                
                batch = move_batch(batch, self.device)
                loss_dict = self.train_model.compute_loss(network, batch)
                self.track_tensor_dict(loss_dict)
                
                total_loss = self.train_model.summary_loss(loss_dict) / self.gradient_accumulation_step
                total_loss.backward()
                
                if (local_step + 1) % self.gradient_accumulation_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    self.after_step(train_modules)
            self.after_epoch(train_modules)
        self.after_loop(train_modules)
    
    def before_loop(self, kwargs_dict: dict) -> None:
        for plugin in self.plugins:
            plugin.before_loop(**kwargs_dict)
    
    def after_loop(self, kwargs_dict: dict) -> None:
        for plugin in self.plugins:
            plugin.after_loop(**kwargs_dict)
    
    def before_epoch(self, kwargs_dict: dict) -> None:
        for plugin in self.plugins:
            plugin.before_epoch(**kwargs_dict)
    
    def after_epoch(self, kwargs_dict: dict) -> None:
        for plugin in self.plugins:
            plugin.after_epoch(**kwargs_dict)
    
    def before_step(self, kwargs_dict: dict) -> None:
        for plugin in self.plugins:
            plugin.before_step(**kwargs_dict)
    
    def after_step(self, kwargs_dict: dict) -> None:
        for plugin in self.plugins:
            plugin.after_step(**kwargs_dict)

    def add_plugin(self, plugin: Plugin) -> Self:
        plugin.trainer = self
        self.plugins.append(plugin)
        return self
