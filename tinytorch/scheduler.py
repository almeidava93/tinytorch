from tinytorch.optimizer import Optimizer

from abc import ABC, abstractmethod

class Scheduler(ABC):
    @abstractmethod
    def step(self):
        pass

class ExponentialLR(Scheduler):
    def __init__(self, optimizer: Optimizer, gamma: float = 0.95):
        self.optimizer = optimizer
        self.gamma = gamma

    def step(self):
        self.optimizer.learning_rate *= self.gamma