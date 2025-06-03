from tinytorch.tensor import Tensor

class Parameter(Tensor):
  def __init__(self, value, _children=(), _op='', label=None, requires_grad: bool = True, *args, **kwargs):
    super().__init__(value, _children, _op, label, requires_grad, *args, **kwargs)

  def __repr__(self):
    return f'Parameter(value={self.value}{", label="+self.label if self.label else ""}{", op="+self._op if self._op else ""}{", requires_grad="+str(self.requires_grad)})'
