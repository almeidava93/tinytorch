import numpy as np
from tinytorch.module import Module
from tinytorch.parameter import Parameter


class Optimizer:
  def __init__(self, model: Module, learning_rate: float):
    self.model = model
    self.learning_rate = learning_rate

  def step(self):
    for param in self.model.parameters():
      param.value = param.value - self.learning_rate * param.grad

  def zero_grad(self):
    for param in self.model.parameters():
      param.grad = np.zeros_like(param.value, dtype=np.float64)