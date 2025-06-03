import numpy as np
from tinytorch.module import Module
from tinytorch.parameter import Parameter
from tinytorch.tensor import Tensor


class Linear(Module):
  def __init__(self, in_features: int, out_features: int, bias: bool = True):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight: Parameter = None
    self.bias: Parameter = None
    self.use_bias: bool = bias

    self.initialize_weights()

  def __repr__(self):
    return f'{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias})'

  def initialize_weights(self, method: str = 'random'):
    if method == 'random':
      self.weight = Parameter(np.random.randn(self.in_features, self.out_features))
      if self.use_bias:
        self.bias = Parameter(np.random.randn(1, self.out_features))
    else:
      raise NotImplementedError

  def forward(self, input: Tensor):
    output = input @ self.weight + self.bias
    return output