from torch.utils.data import DataLoader


class _BaseTrainer:
    def __init__(self, hparams: dict) -> None:
        self.epoch_duration = hparams["epoch_duration"]
        self.batch_size = hparams["batch_size"]
    
    def train_loop(
        self, 
        dataset, 
        network, 
        criterion, 
        optimizer,
        ) -> None:
        data_loader = DataLoader(dataset, self.batch_size, shuffle=True)
        network.train()
        
        for epoch in range(self.epoch_duration):
            for step, batch in enumerate(data_loader):
                optimizer.zero_grad()
                loss = criterion(network, batch)
                loss.backward()
                optimizer.step()
        
        optimizer.zero_grad()
        network.eval()