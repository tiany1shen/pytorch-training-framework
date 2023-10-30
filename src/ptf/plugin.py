class _BasePlugin:
    trainer = None
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
    def __init__(self, update_period: int) -> None:
        self.update_period = update_period
        
    def after_loop(self):
        # Zero out gradients after training procedure finished.
        self.trainer.optimzer.zero_grad()
    
    def before_step(self):
        # Reset the gradients to zero before each optimizing step.
        if self.trainer.local_step % self.update_period == 1:
            self.trainer.optimzer.zero_grad()
    
    def after_step(self):
        # Update the parameters according to the gradients attached to them after each optimizing step.
        if self.trainer.local_step % self.update_period == 0:
            self.trainer.optimzer.step()
    
    
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
    
    def __init__(self, seed: int) -> None:
        self.seed = seed
        
    def fix_seed(self) -> None:
        # Fix random seed and generate next seed
        manual_seed(self.seed)
        random.seed(self.seed)
        self.seed = random.randint(*self.seed_range)
        
    def before_epoch(self):
        # Before each epoch, fix the random seed and build a data iterator
        self.fix_seed()
        self.trainer.data_iterator = iter(DataLoader(
            self.trainer.dataset, 
            self.trainer.batch_size, 
            shuffle=True, drop_last=True
        ))
        
    def after_epoch(self):
        # update trainer random seed
        self.trainer.seed = self.seed
        
        
    
