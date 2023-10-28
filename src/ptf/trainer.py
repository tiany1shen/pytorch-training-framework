from torch.utils.data import DataLoader


class _BaseTrainer:
    def __init__(self, hparams: dict, modules: dict) -> None:
        self.hparams = hparams
        self.epoch_duration = hparams["epoch_duration"]
        self.batch_size = hparams["batch_size"]
        
        self.modelus = modules
        self.dataset = modules["dataset"]
        self.network = modules["network"]
        self.model = modules["model"]
        self.optimzer = modules["optimizer"]
        
        self.epoch = 0
        self.step = 0
        
    @property
    def epoch_length(self):
        return len(self.dataset) // self.batch_size
    
    def loop(self) -> None:
        self.before_loop()
        for epoch in range(self.epoch_duration):
            self.train_one_epoch()
        self.after_loop()
        
    def train_one_epoch(self):
        self.before_epoch()
        for step in range(self.epoch_length):
            self.train_one_step()
        self.after_epoch()
    
    def train_one_step(self):
        self.before_step()
        #* 对一个 batch 的计算
        batch = next(self.data_iterator)
        loss = self.model.compute_loss(self.network, batch)
        loss.backward()
        self.after_step()
        
    def before_loop(self):
        self.network.train()
        pass
        
    def after_loop(self):
        self.optimzer.zero_grad()
        self.network.eval()
        pass
        
    def before_epoch(self):
        self.data_iterator = iter(DataLoader(self.dataset, self.batch_size, shuffle=True, drop_last=True))
        pass 
    
    def after_epoch(self):
        pass
    
    def before_step(self):
        self.optimzer.zero_grad()
        pass
    
    def after_step(self):
        self.optimzer.step()
        pass