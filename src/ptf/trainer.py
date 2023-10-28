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
    
    def loop(self) -> None:
        data_loader = DataLoader(self.dataset, self.batch_size, shuffle=True)
        self.network.train()
        
        for epoch in range(self.epoch_duration):
            for step, batch in enumerate(data_loader):
                self.optimizer.zero_grad()
                loss = self.model.compute_loss(self.network, batch)
                loss.backward()
                self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.network.eval()