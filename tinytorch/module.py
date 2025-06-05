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

  def _add_indent(self, string: str, n_tabs: int = 1):
     lines = string.split('\n')
     # remove first and add indentation
     submodules_indented = f'\n{'\t'*n_tabs}' + f'\n{'\t'*n_tabs}'.join(lines[1:-1]) + f'\n{'\t'*n_tabs}{lines[-1]}'
     return ''.join([lines[0], submodules_indented])
  
  def __repr__(self):
    children_modules = ''
    if len(self._submodules)>0:
      for attr_name, attr_value in self._submodules.items():
        if isinstance(attr_value, Module):
          children_modules += '\n' + f'{attr_name}: {str(attr_value)},'
      if len(children_modules)>0:
        children_modules += '\n'
    
    if len(self._submodules)>0:
      repr_str = self._add_indent(f'{self.__class__.__name__}({children_modules})')
    else:
      repr_str = f'{self.__class__.__name__}({children_modules})'
    return repr_str
  
  def __setattr__(self, name, value):
    if isinstance(value, Module):
        self.__dict__.setdefault('_submodules', {})[name] = value
    elif isinstance(value, list) and isinstance(value[0], Module):
        for idx, module in enumerate(value):
            self.__dict__.setdefault('_submodules', {})[f'{name}_{idx}'] = module
    elif isinstance(value, Parameter):
        self.__dict__.setdefault('_parameters', {})[name] = value

    super().__setattr__(name, value)

  def parameters(self):
    for name, param in self._parameters.items():
        yield param
    for name, module in self._submodules.items():
        yield from module.parameters()