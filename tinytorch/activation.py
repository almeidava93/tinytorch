from ast import Module

from tinytorch.tensor import Tensor


class ReLU(Module):
  def forward(self, input: Tensor) -> Tensor:
    output = input * (input.value > 0)
    return output