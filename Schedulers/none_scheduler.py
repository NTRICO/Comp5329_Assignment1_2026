from torch.optim.lr_scheduler import LRScheduler

class NoneScheduler(LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return list(self.base_lrs)
