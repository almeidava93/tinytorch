from tinytorch.parameter import Parameter
from tinytorch.tensor import Tensor

class Module:
  def __init__(self, *args, **kwargs):
    self._parameters = {}
    self._submodules = {}

  def forward(self, *args, **kwargs):
    raise NotImplementedError

  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)

  def __repr__(self):
    children_modules = ''
    for attr_name, attr_value in vars(self).items():
      if isinstance(attr_value, Module):
        children_modules += '\n\t' + f'{attr_name}: {str(attr_value)},'
    if len(children_modules)>0:
      children_modules += '\n'
    return f'{self.__class__.__name__}({children_modules})'
  
  def __setattr__(self, name, value):
    if isinstance(value, Module):
        self.__dict__.setdefault('_submodules', {})[name] = value
    elif isinstance(value, Parameter):
        self.__dict__.setdefault('_parameters', {})[name] = value

    super().__setattr__(name, value)

  def parameters(self):
    for name, param in self._parameters.items():
        yield param
    for name, module in self._submodules.items():
        yield from module.parameters()