class _BaseModel:
    
    def compute_loss(self, network, batch):
        raise NotImplementedError