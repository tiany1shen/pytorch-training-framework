from typing import Optional


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
        
    def after_loop(self):
        # Zero out gradients after training procedure finished.
        self.trainer.optimizer.zero_grad()
    
    def after_step(self):
        # Update the parameters according to the gradients attached to them after each optimizing step.
        if not self.update_gradient: return
        
        self.trainer.optimizer.step()
        self.trainer.optimizer.zero_grad()
        if self.debug:
            print(
                f"[Epoch: {self.trainer.epoch}\tStep: {self.trainer.step}] "
                "Update network weights."
            )


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
        # Fixing seed before loop helps with initializing weights
        self.fix_seed()
        if self.debug:
            print(
                f"[Before Loop] Fix random seed {self.seed}"
            )
        
        
    def before_epoch(self):
        if self.shutdown: return
        # Before each epoch, fix the random seed and build a data iterator
        self.fix_seed()
        self.next_seed = random.randint(*self.seed_range)
        if self.debug:
            print(
                f"[Epoch: {self.trainer.epoch}] Fix random seed {self.seed}."
            )
        
        self.trainer.data_iterator = iter(DataLoader(
            self.trainer.dataset, 
            self.trainer.batch_size, 
            shuffle=True, drop_last=True
        ))
        if self.debug:
            print(f"[Epoch: {self.trainer.epoch}] Reset data iterator.")
        
    def after_epoch(self):
        if self.shutdown: return
        # update trainer random seed
        self.trainer.seed = self.next_seed
        self.next_seed = None
        if self.debug:
            print(
                f"[Epoch: {self.trainer.epoch}\tStep: {self.trainer.step}] "
                "Update random seed."
            )


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
        
        network_file = self.checkpoint_path / "network.pth"
        if network_file.exists():
            network_state_dict = torch_load(network_file)
            self.trainer.network.load_state_dict(network_state_dict)
            if self.debug:
                print(f"[Before Loop] Loading network from {network_file}.")
            
        optimizer_file = self.checkpoint_path / "optimizer.pth"
        if optimizer_file.exists():
            optimizer_state_dict = torch_load(optimizer_file)
            self.trainer.optimizer.load_state_dict(optimizer_state_dict)
            if self.debug:
                print(f"[Before Loop] Loading optimizer from {optimizer_file}.")
            
        trainer_file = self.checkpoint_path / "trainer.pth"
        if trainer_file.exists():
            trainer_state_dict = torch_load(trainer_file)
            self.trainer.seed = trainer_state_dict["seed"]
            self.trainer.start_epoch = trainer_state_dict["epoch"]
            if self.debug:
                print(f"[Before Loop] Loading trainer random seed {self.trainer.seed}")
                print(f"[Before Loop] Loaded Checkpoint saved at epoch {self.trainer.epoch}")
    
    def after_epoch(self):
        if not self.save_checkpoint: return
        saving_path = self.saving_dir / f"epoch-{self.trainer.epoch}"
        saving_path.mkdir(parents=True, exist_ok=True)
        
        torch_save(
            self.trainer.network.state_dict(), saving_path / "network.pth")
        if self.debug:
            print(f"[Epoch: {self.trainer.epoch}] Saving network to {saving_path}")
        torch_save(
            self.trainer.optimizer.state_dict(), saving_path / "optimizer.pth")
        if self.debug:
            print(f"[Epoch: {self.trainer.epoch}] Saving optimizer to {saving_path}")
        trainer_state_dict = {"epoch": self.trainer.epoch}
        if self.trainer.seed is not None:
            trainer_state_dict["seed"] = self.trainer.seed
        torch_save(trainer_state_dict, saving_path / "trainer.pth")
        if self.debug:
            print(f"[Epoch: {self.trainer.epoch}] Saving trainer state to {saving_path}")

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
        return self.trainer.epoch % self.evaluate_period == 0
    
    def after_epoch(self):
        if not hasattr(self.trainer.model, "evaluate") or not self.evaluate: return 
        if self.debug:
            print(f"[Epoch: {self.trainer.epoch}] Evaluating on validation dataset.")
        eval_metrics = self.trainer.model.evaluate(self.trainer.network, self.dataset)
        if self.debug:
            print(f"[Epoch: {self.trainer.epoch}] Evaluating results:")
            for k, v in eval_metrics.items():
                if k in self.show_result_format:
                    print(f"[Epoch: {self.trainer.epoch}] \t{k}:\t{self.show_result_format[k](v)}")
                else:
                    print(f"[Epoch: {self.trainer.epoch}] \t{k}:\t{self.show_result_format['others'](v)}")

# class TensorboardLoggingPlugin(_BasePlugin):
#     #todo: implement tensorboard plugin
#     raise NotImplementedError

# class ProgressBarPlugin(_BasePlugin):
#     #todo: implement progress bar plugin
#     raise NotImplementedError
    
