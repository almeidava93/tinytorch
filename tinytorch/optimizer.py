import numpy as np
from tinytorch.module import Module

from abc import ABC, abstractmethod

class Optimizer(ABC):
  @abstractmethod
  def step(self):
    pass

  @abstractmethod
  def zero_grad(self):
    for param in self.model.parameters():
      param.grad = np.zeros_like(param.value, dtype=np.float64)

class SGD(Optimizer):
  def __init__(self, model: Module, learning_rate: float, momentum: float = 0.0):
    self.model = model
    self.learning_rate = learning_rate
    self.momentum = momentum

    # Initialize variables to store previous update terms if momentum is used
    self.m = []

    if self.momentum > 0:
      for param in self.model.parameters():
        self.m.append(np.zeros_like(param.value))

  def step(self):
    for idx, param in enumerate(self.model.parameters()):

      if self.momentum > 0:
        m = self.m[idx]
        m = self.momentum*m + self.learning_rate*param.grad
        self.m[idx] = m # update previous term
        param.value = param.value - m
        continue

      param.value = param.value - self.learning_rate * param.grad

  def zero_grad(self):
    for param in self.model.parameters():
      param.grad = np.zeros_like(param.value, dtype=np.float64)


class Adam(Optimizer):
  """
  See https://www.datacamp.com/tutorial/adam-optimizer-tutorial
  """
  def __init__(self, model: Module, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
    self.model = model
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.time = 0

    # Initialize variables to store moving avg and squared gradients weighted average
    self.m = []
    self.v = []

    for param in self.model.parameters():
      self.m.append(np.zeros_like(param.value))
      self.v.append(np.zeros_like(param.value))

  def step(self):
    self.time += 1

    for idx, param in enumerate(self.model.parameters()):
      # Get previous averages
      m = self.m[idx]
      v = self.v[idx]

      # Update moving average
      m = self.beta1 * m + (1-self.beta1)*param.grad
      # Update squared gradients weighted average
      v = self.beta2 * v + (1-self.beta2)*(param.grad**2)

      # Update averages tracking
      self.m[idx] = m
      self.v[idx] = v

      # Apply bias correction
      m_hat = m / (1 - self.beta1**self.time)
      v_hat = v / (1 - self.beta2**self.time)

      # Update weights
      param.value = param.value - self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon))


  def zero_grad(self):
    for param in self.model.parameters():
      param.grad = np.zeros_like(param.value, dtype=np.float64)