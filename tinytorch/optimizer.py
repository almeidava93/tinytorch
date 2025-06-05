# See https://www.datacamp.com/tutorial/adam-optimizer-tutorial

import numpy as np
from tinytorch.module import Module

from abc import ABC, abstractmethod

class Optimizer(ABC):
  @abstractmethod
  def step(self):
    pass

  def zero_grad(self):
    for param in self.model.parameters():
      param.grad = np.zeros_like(param.value, dtype=np.float64)


class SGD(Optimizer):
  """
  Stochastic Gradient Descent (SGD)
  """
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


class RMSProp(Optimizer):
  """
  Root Mean Square Propagation (RMSProp)
  """
  def __init__(self, model: Module, learning_rate: float, beta: float = 0.9, epsilon: float = 1e-8):
    self.model = model
    self.learning_rate = learning_rate
    self.beta = beta
    self.epsilon = epsilon

    # Store moving average of previous squared gradients 
    self.m = []

    # Initialize them with zeros
    for param in self.model.parameters():
      self.m.append(np.zeros_like(param.value))

  def step(self):
    for idx, param in enumerate(self.model.parameters()):
      m = self.m[idx]
      m = m*self.beta + (1 - self.beta)*(param.grad**2)
      self.m[idx] = m # update previous term

      param.value = param.value - self.learning_rate*(param.grad/(np.sqrt(m) + self.epsilon))


class Adam(Optimizer):
  """
  Adam Optimization
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


class AdamW(Optimizer):
  """
  Adam with decoupled weight decay Optimization (AdamW)
  """
  def __init__(self, 
               model: Module, 
               learning_rate: float, 
               beta1: float = 0.9, 
               beta2: float = 0.999, 
               weight_decay: float = 1e-6,
               schedule_multiplier: float = 1.0, 
               epsilon: float = 1e-8,
               ):
    self.model = model
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.weight_decay = weight_decay # lambda in the paper
    self.schedule_multiplier = schedule_multiplier # eta_t in the paper
    self.epsilon = epsilon
    self.time = 0

    # Initialize variables to store moving avg and squared gradients weighted average
    self.m = []
    self.v = []

    for param in self.model.parameters():
      self.m.append(np.zeros_like(param.value))
      self.v.append(np.zeros_like(param.value))

  def step_schedule_multiplier(self):
    "This method may be overrided to update the schedule multiplier as a function of time. Defaults to a constant schedule multiplier = 1.0."
    pass

  def step(self):
    self.time += 1
    self.step_schedule_multiplier()

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
      param.value = param.value - self.schedule_multiplier * (
          self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon)) +
          self.weight_decay * param.value
        )