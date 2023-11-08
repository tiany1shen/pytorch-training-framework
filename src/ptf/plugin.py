from typing import Optional, List


class _BasePlugin:
    trainer = None
    debug = False
    def before_loop(self):
        pass 
    
    def after_loop(self):
        pass 
    
    def before_epoch(self):
        pass 
    
    def after_epoch(self):
        pass 
    
    def before_step(self):
        pass 
    
    def after_step(self):
        pass
    
    def debug_header(self, scope: str):
        assert scope in ["before-loop", "in-epoch", "in-step", "after-loop"]
        if scope == "before-loop":
            return "Before Loop"
        if scope == "after-loop":
            return "After Loop"
        if scope == "in-epoch":
            max_epoch = self.trainer.start_epoch + self.trainer.epoch_num 
            epoch_str = (
                " " * (len(str(max_epoch)) - len(str(self.trainer.epoch)))
                + str(self.trainer.epoch)
            )
            return f"Epoch: {epoch_str}"
        if scope == "in-step":
            max_epoch = self.trainer.start_epoch + self.trainer.epoch_num 
            epoch_str = (
                " " * (len(str(max_epoch)) - len(str(self.trainer.epoch)))
                + str(self.trainer.epoch)
            )
            max_step = self.trainer.epoch_length * max_epoch
            step_str = (
                " " * (len(str(max_step)) - len(str(self.trainer.step)))
                + str(self.trainer.step)
            )
            return f"Epoch: {epoch_str} | Step: {step_str}"
    
    def show_debug_infos(self, scope: str, infos: str | List[str]):
        if not self.debug: return 
        if isinstance(infos, str):
            infos = [infos]
        for info in infos:
            print(f"[{self.debug_header(scope)}] {info}")


class WeightsUpdatePlugin(_BasePlugin):
    r"""
    A basic plugin used in almost every training loop. It updates parameters registered in the optimizer object.
    
    Args:
        update_period (int): numbers of batch the loop iterates for one update.
    """
    
    def __init__(self, update_period=1) -> None:
        self.update_period = update_period
    
    @property
    def retain_gradient(self):
        if self.update_period == 1:
            return False
        return self.trainer.local_step % self.update_period != 1
    
    @property
    def update_gradient(self):
        return self.trainer.local_step % self.update_period == 0
    
    def before_loop(self):
        self.trainer.model.init_weights(self.trainer.network)
        self.trainer.optimizer.zero_grad()
        self.show_debug_infos("before-loop", "Reset network weights.")
        
    def after_loop(self):
        # Zero out gradients after training procedure finished.
        self.trainer.optimizer.zero_grad()
    
    def after_step(self):
        if not self.update_gradient: return
        self.show_debug_infos("in-step", "Update network weights.")
        
        # Update the parameters according to the gradients attached to them after each optimizing step.
        self.trainer.optimizer.step()
        self.trainer.optimizer.zero_grad()


import random
from torch import manual_seed
from torch.utils.data import DataLoader
class ReproduciblePlugin(_BasePlugin):
    r"""
    A plugin that can control the random state in single-thread training procedure. For reproducibility, we fixed the 
    random seed before each epoch, and update this seed after each epoch.
    
    Args:
        seed (int): random seed for the NEXT epoch. 
    """
    seed_range = [- 2 ** 63, 2 ** 64 - 1]
    
    @property
    def seed(self):
        return self.trainer.seed
    
    @property    
    def shutdown(self):
        return self.seed is None
        
    def fix_seed(self) -> None:
        # Fix random seed and generate next seed
        manual_seed(self.seed)
        random.seed(self.seed)
        
    def before_loop(self):
        if self.shutdown: return
        self.show_debug_infos("before-loop", f"Fix random seed {self.seed}.")
        
        # Fixing seed before loop helps with initializing weights
        self.fix_seed()
        
    def before_epoch(self):
        if self.shutdown: return
        self.show_debug_infos("in-epoch", [
            f"Fix random seed {self.seed}.",
            "Reset data iterator."
        ])
        
        # Before each epoch, fix the random seed and build a data iterator
        self.fix_seed()
        self.next_seed = random.randint(*self.seed_range)
        self.trainer.data_iterator = iter(DataLoader(
            self.trainer.dataset, 
            self.trainer.batch_size, 
            shuffle=True, drop_last=True
        ))
        
    def after_epoch(self):
        if self.shutdown: return
        self.show_debug_infos("in-epoch", f"Update random seed {self.next_seed}.")
        # update trainer random seed
        self.trainer.seed = self.next_seed
        self.next_seed = None


from pathlib import Path 
from torch import load as torch_load
from torch import save as torch_save


class CheckpointPlugin(_BasePlugin):
    
    def __init__(self, resume_from, saving_dir, saving_period):
        self.checkpoint_path = Path(resume_from) if resume_from else None
        self.saving_dir = Path(saving_dir) if saving_dir else None
        self.saving_period = saving_period
        if not saving_period and saving_dir:
            self.saving_period = 1
        
        
    @property 
    def load_checkpoint(self):
        if self.checkpoint_path is None:
            return False 
        return self.checkpoint_path.exists()
    
    @property
    def save_checkpoint(self):
        if not self.saving_dir: 
            return False 
        return self.trainer.epoch % self.saving_period == 0
    
    def before_loop(self):
        if not self.load_checkpoint: return
        self.show_debug_infos("before-loop", f"Load checkpoint from {self.checkpoint_path}")
        
        network_file = self.checkpoint_path / "network.pth"
        if network_file.exists():
            network_state_dict = torch_load(network_file)
            self.trainer.network.load_state_dict(network_state_dict)
            
        optimizer_file = self.checkpoint_path / "optimizer.pth"
        if optimizer_file.exists():
            optimizer_state_dict = torch_load(optimizer_file)
            self.trainer.optimizer.load_state_dict(optimizer_state_dict)
            
        trainer_file = self.checkpoint_path / "trainer.pth"
        if trainer_file.exists():
            trainer_state_dict = torch_load(trainer_file)
            self.trainer.seed = trainer_state_dict["seed"]
            self.trainer.start_epoch = trainer_state_dict["epoch"]
        
    def after_epoch(self):
        if not self.save_checkpoint: return
        saving_path = self.saving_dir / f"epoch-{self.trainer.epoch}"
        saving_path.mkdir(parents=True, exist_ok=True)
        self.show_debug_infos("in-epoch", f"Save checkpoint to {saving_path}")
        
        torch_save(
            self.trainer.network.state_dict(), saving_path / "network.pth")
        torch_save(
            self.trainer.optimizer.state_dict(), saving_path / "optimizer.pth")
        trainer_state_dict = {"epoch": self.trainer.epoch}
        if self.trainer.seed is not None:
            trainer_state_dict["seed"] = self.trainer.seed
        torch_save(trainer_state_dict, saving_path / "trainer.pth")

class EvaluatePlugin(_BasePlugin):
    show_result_format = {
        "correctness": lambda x: f"{x:.1%}",
        "others": lambda x: f"{x:.3f}"
    }
    
    def __init__(self, evaluate_dataset, evaluate_period):
        self.dataset = evaluate_dataset
        self.evaluate_period = evaluate_period
    
    @property
    def evaluate(self):
        return self.trainer.epoch % self.evaluate_period == 0 and \
            hasattr(self.trainer.model, "evaluate")
    
    def before_loop(self):
        if self.trainer.model.evaluate_metrics:
            metric_trackers = {}
            for metric_name in self.trainer.model.evaluate_metrics:
                metric_trackers[metric_name] = self.trainer.metric_tracker_type()
            self.trainer.metric_trackers = metric_trackers
    
    def after_epoch(self):
        if not self.evaluate: return 
        self.show_debug_infos("in-epoch", "Evaluate on validate dataset.")
        
        eval_metrics_dict = self.trainer.model.evaluate(self.trainer.network, self.dataset)
        for metric_name, metric_value in eval_metrics_dict.items():
            self.trainer.metric_trackers[metric_name].add(metric_value)
        self.show_debug_infos("in-epoch", (
            f"\t{k}:\t{self.show_result_format[k if k in self.show_result_format else 'others'](v)}"
            for k, v in eval_metrics_dict.items()
        ))
    
class ProgressBarPlugin(_BasePlugin): 
    def progress_bar(self, curr, total):
        progress = curr * 10 // total
        bar = "|" + ">" * progress + "=" * (10 - progress) + "|"
        rate = " " * (len(str(total)) - len(str(curr))) + str(curr)
        return bar + f"( {rate} / {total})"
    
    def plot_progress_bar(self):
        bar = "\rProgressing: "
        
        current_epoch= self.trainer.epoch
        total_epoch = self.trainer.start_epoch + self.trainer.epoch_num
        epoch_rate = f"{current_epoch / total_epoch:.1%}"
        bar += f"[Total Epoch {epoch_rate:>6}]"
        bar += self.progress_bar(current_epoch, total_epoch)
        
        bar += " | "
        total_local_step = self.trainer.epoch_length * self.trainer.gradient_accumulate
        current_local_step = self.trainer.local_step - (self.trainer.local_epoch - 1) * total_local_step
        local_step_rate = f"{current_local_step / total_local_step:.1%}"
        bar += f"[Local Step {local_step_rate:>6}]"
        bar += self.progress_bar(current_local_step, total_local_step)
        
        print(bar, end=" ", flush=True)
    
    def before_step(self):
        if self.debug: return
        self.plot_progress_bar()
        

class TensorboardLoggingPlugin(_BasePlugin):
    show_result_format = {
        "correctness": lambda x: f"{x:.1%}",
        "others": lambda x: f"{x:.3f}"
    }
    
    def __init__(self, loss_period: int, epoch_period: int) -> None:
        self.loss_period = loss_period
        self.epoch_period = epoch_period
    
    @property
    def loss_trackers(self): return self.trainer.loss_trackers 
    
    @property 
    def metiric_trackers(self):
        if hasattr(self.trainer, "metric_trackers"):
            return self.trainer.metric_trackers
        else:
            return None
    
    @property
    def logging_loss(self):
        return self.trainer.step % self.loss_period == 0
    
    @property
    def logging_metric(self):
        return self.trainer.epoch % self.epoch_period == 0 and self.metiric_trackers
    
    def add_scalar(self, name, value, step):
        # print(f"loss/{name}: {value:.4f}")
        pass
    
    def after_step(self):
        if not self.logging_loss: return
        for loss_name, tracker in self.loss_trackers.items():
            loss_value = tracker.report()
            self.add_scalar(f"loss/{loss_name}", loss_value, self.trainer.step)
            self.show_debug_infos("in-step", f"Logging loss/{loss_name}: {loss_value:.4f}")
    
    def after_epoch(self):
        if not self.logging_metric: return 
        for metric_name, tracker in self.metiric_trackers.items():
            metric_value = tracker.report()
            self.add_scalar(f"metric/{metric_name}", metric_value, self.trainer.epoch)
            self.show_debug_infos(
                "in-epoch", 
                f"Logging metric/{metric_name}: {self.show_result_format[metric_name](metric_value)}"
            )

