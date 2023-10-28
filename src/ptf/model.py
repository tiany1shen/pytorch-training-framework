class _BaseModel:
    def __init__(self, defaults):
        self.defaults = defaults
    
    def compute_loss(self, network, batch):
        raise NotImplementedError