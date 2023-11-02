class _BaseModel:
    
    def compute_loss(self, network, batch):
        raise NotImplementedError
    
    def init_weights(self, network):
        raise NotImplementedError