class _BasePlugin:
    def __init__(self, trainer=None):
        self.trainer = trainer
        
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