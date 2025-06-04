import numpy as np
from tinytorch.module import Module
from tinytorch.tensor import Tensor
import tinytorch.grads as grads

class ReLU(Module):
  def __init__(self):
    super().__init__()
    
  def forward(self, input: Tensor) -> Tensor:
    output = input * (input.value > 0)
    output._op = 'ReLUActivation'
    return output
  
class LeakyReLU(Module):
  def __init__(self, negative_slope: float = 0.01):
    super().__init__()
    self.negative_slope = negative_slope

  def forward(self, input: Tensor) -> Tensor:
    negative_values_mask = input.value < 0
    leaky_relu_mask = np.ones_like(input.value)
    leaky_relu_mask[negative_values_mask] = self.negative_slope
    output = input * leaky_relu_mask
    output._op = 'LeakyReLUActivation'
    return output

class Sigmoid(Module):
  def __init__(self):
    super().__init__()

  def _sigmoid(self, input: np.array):
    return 1/(1+np.exp(-input))
  
  def forward(self, input: Tensor) -> Tensor:
    output_value = self._sigmoid(input.value)
    output = Tensor(output_value, _children=(input,), _op='SigmoidActivation')
    output._backward = grads.grad_fn_sigmoid
    return output

class SiLU(Module):
  def __init__(self):
    super().__init__()

  def _sigmoid(self, input: np.array):
    return 1/(1+np.exp(-input))
  
  def forward(self, input: Tensor) -> Tensor:
    output_value = input.value * self._sigmoid(input.value)
    output = Tensor(output_value, _children=(input,), _op='SiLUActivation')
    output._backward = grads.grad_fn_silu
    return output