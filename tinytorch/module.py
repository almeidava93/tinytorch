class Module:
  def forward(self, *args, **kwargs):
    raise NotImplementedError

  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)

  def __repr__(self):
    children_modules = ''
    for attr_name, attr_value in vars(self).items():
      if isinstance(attr_value, Module):
        children_modules += '\n\t' + f'{attr_name}: {str(attr_value)}'
    if len(children_modules)>0:
      children_modules += '\n'
    return f'{self.__class__.__name__}({children_modules})'