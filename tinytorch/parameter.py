import numpy as np
from tinytorch.tensor import Tensor

class Parameter(Tensor):
  def __init__(self, *args, **kwargs):
    super().__init__( *args, **kwargs)
    self.retain_grads = True

  def __repr__(self):
    return f'Parameter(value={self.value}{", label="+self.label if self.label else ""}{", op="+self._op if self._op else ""}{", requires_grad="+str(self.requires_grad)})'