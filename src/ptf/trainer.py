from math import ceil
from torch.utils.data import DataLoader


class _BaseTrainer:
    r"""
    
    """
    def __init__(self, hparams: dict, modules: dict) -> None:
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
        
    @property
    def epoch_length(self):
        accumulated_batch_size = self.batch_size * self.gradient_accumulate
        return ceil(len(self.dataset) / accumulated_batch_size)
    
    @property
    def epoch(self):
        return self.start_epoch + self.local_epoch
    
    @property
    def step(self):
        return self.start_epoch * self.epoch_length + self.local_step
    
    def loop(self) -> None:
        self.before_loop()
        for epoch in range(self.epoch_num):
            self.train_one_epoch()
        self.after_loop()
        self.network.eval()
        
    def train_one_epoch(self):
        self.local_epoch += 1
        self.before_epoch()
        for step in range(self.epoch_length * self.gradient_accumulate):
            self.network.train()
            self.train_one_step()
        self.after_epoch()
    
    def train_one_step(self):
        self.local_step += 1
        self.before_step()
        
        # 对一个 batch 的计算
        batch = self._get_next_batch()
        loss = self.model.compute_loss(self.network, batch)
        loss.backward()
        
        self.after_step()
        
    def before_loop(self):
        pass
        
    def after_loop(self):
        self.optimzer.zero_grad()
        pass
        
    def before_epoch(self):
        pass 
    
    def after_epoch(self):
        pass
    
    def before_step(self):
        if self.local_step % self.gradient_accumulate == 1:
            self.optimzer.zero_grad()
        pass
    
    def after_step(self):
        if self.local_step % self.gradient_accumulate == 0:
            self.optimzer.step()
        pass
    
    def _get_next_batch(self):
        if not hasattr(self, "data_iterator"):
            self.data_iterator = iter(DataLoader(self.dataset, self.batch_size, shuffle=True, drop_last=True))

        try:
            batch = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(DataLoader(self.dataset, self.batch_size, shuffle=True, drop_last=True))
            batch = next(self.data_iterator)
        return batch