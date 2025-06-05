from typing import Optional
import numpy as np
from tinytorch.module import Module
from tinytorch.parameter import Parameter
from tinytorch.tensor import Tensor
import tinytorch
import tinytorch.init

class Linear(Module):
  def __init__(self, in_features: int, out_features: int, bias: bool = True, init_method: Optional[str] = None, init_method_kwargs: dict = {}):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight: Parameter = None
    self.bias: Parameter = None
    self.use_bias: bool = bias

    self.initialize_weights(init_method, **init_method_kwargs)

  def __repr__(self):
    return f'{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias})'

  def initialize_weights(self, method: Optional[str] = None, **kwargs):
    if method is None:
      method, init_method_kwargs = tinytorch.get_init_method()
    self.weight = getattr(tinytorch.init, method)(size=(self.in_features, self.out_features), **init_method_kwargs, **kwargs)
    if self.use_bias:
      self.bias = getattr(tinytorch.init, method)(size=(1, self.out_features), **init_method_kwargs, **kwargs)

  def forward(self, input: Tensor):
    output = input @ self.weight + self.bias
    return output
  

class Sequential(Module):
  def __init__(self, modules: list[Module]):
    super().__init__()
    self.modules = modules

  def __repr__(self):
    if len(self.modules) == 0:
      return f'{self.__class__.__name__}([])'
    
    modules_str = '\n\t' +  '\n\t'.join([m.__repr__() + ',' for m in self.modules]) + '\n'
    return f'{self.__class__.__name__}([{modules_str}])'
  
  def forward(self, input: Tensor):
    output = input
    for module in self.modules:
      output = module(output)
    return output