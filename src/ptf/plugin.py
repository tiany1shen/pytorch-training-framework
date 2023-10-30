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
    
    def after_loop(self):
        self.trainer.optimzer.zero_grad()
    
    def before_step(self):
        if self.trainer.local_step % self.trainer.gradient_accumulate == 1:
            self.trainer.optimzer.zero_grad()
    
    def after_step(self):
        if self.trainer.local_step % self.trainer.gradient_accumulate == 0:
            self.trainer.optimzer.step()