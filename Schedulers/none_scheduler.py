class NoneScheduler:
    """A no-op scheduler that keeps learning rate unchanged."""

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        pass

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self):

        return {}

    def load_state_dict(self, state_dict):
        pass