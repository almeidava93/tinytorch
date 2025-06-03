import numpy as np
from tinytorch.module import Module


class Optimizer:
  def __init__(self, model: Module, learning_rate: float):
    self.model = model
    self.learning_rate = learning_rate

  def step(self):
    # Update the value of the existing weight Tensor
    self.model.weight.value = self.model.weight.value - self.learning_rate * self.model.weight.grad.sum()
    # Update the value of the existing bias Tensor
    if self.model.bias is not None:
      self.model.bias.value = self.model.bias.value - self.learning_rate * self.model.bias.grad.sum()

  def zero_grad(self):
      self.model.weight.grad = np.zeros_like(self.model.weight.value, dtype=np.float64)
      if self.model.bias is not None:
        self.model.bias.grad = np.zeros_like(self.model.bias.value, dtype=np.float64)