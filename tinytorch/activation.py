from tinytorch.module import Module
from tinytorch.tensor import Tensor

class ReLU(Module):
  def __init__(self):
    super().__init__()
    
  def forward(self, input: Tensor) -> Tensor:
    output = input * (input.value > 0)
    return output