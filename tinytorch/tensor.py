from typing import List
import numpy as np

import tinytorch.grads as grads

class Tensor():
  def __init__(self, value, _children=(), _op='', label=None, requires_grad: bool = True, retain_grads: bool = False, *args, **kwargs):
    self.value: np.array = np.array(value)
    self._children = _children
    self._op = _op
    self._backward = None
    self.grad = np.zeros_like(self.value, dtype=np.float64)
    self.label = label
    self.requires_grad = requires_grad
    self.retain_grads = retain_grads

  def __repr__(self):
    return f'Tensor(value={self.value}{", label="+self.label if self.label else ""}{", op="+self._op if self._op else ""}{", requires_grad="+str(self.requires_grad)})'

  def __add__(self, other):
    if not isinstance(other, Tensor): other = Tensor(other)
    # Compute the output
    output = Tensor(self.value + other.value, _children=(self, other), _op='add')
    # Define gradient function
    output._backward = grads.grad_fn_add
    return output

  def __radd__(self, other):
    if not isinstance(other, Tensor): other = Tensor(other)
    # Compute the output
    output = Tensor(other.value + self.value, _children=(self, other), _op='add')
    # Define gradient function
    output._backward = grads.grad_fn_add
    return output

  def __sub__(self, other):
    if not isinstance(other, Tensor): other = Tensor(other)
    # Compute the output
    output = Tensor(self.value - other.value, _children=(self, other), _op='sub')
    # Define gradient function
    output._backward = grads.grad_fn_sub
    return output

  def __rsub__(self, other):
    if not isinstance(other, Tensor): other = Tensor(other)
    # Compute the output
    output = Tensor(other.value - self.value, _children=(other, self), _op='sub')
    # Define gradient function
    output._backward = grads.grad_fn_sub
    return output

  def __mul__(self, other):
    if not isinstance(other, Tensor): other = Tensor(other)
    # Compute the output
    output = Tensor(self.value * other.value, _children=(self, other), _op='mul')
    # Define gradient function
    output._backward = grads.grad_fn_mul
    return output

  def __rmul__(self, other):
    if not isinstance(other, Tensor): other = Tensor(other)
    # Compute the output
    output = Tensor(other.value * self.value, _children=(other, self), _op='mul')
    # Define gradient function
    output._backward = grads.grad_fn_mul
    return output

  def __truediv__(self, other):
    if not isinstance(other, Tensor): other = Tensor(other)
    # Compute the output
    output = Tensor(self.value / other.value, _children=(self, other), _op='div')
    # Define gradient function
    output._backward = grads.grad_fn_truediv
    return output

  def __rtruediv__(self, other):
    if not isinstance(other, Tensor): other = Tensor(other)
    # Compute the output
    output = Tensor(other.value / self.value, _children=(other, self), _op='div')
    # Define gradient function
    output._backward = grads.grad_fn_truediv
    return output

  def __pow__(self, other):
    if not isinstance(other, Tensor): other = Tensor(other)
    # Compute the output
    output = Tensor(self.value ** other.value, _children=(self, other), _op='pow')
    # Define gradient function
    if other.value.dtype == int:
      output._backward = grads.grad_fn_pow
    else:
      raise NotImplementedError

    return output

  def __matmul__(self, other):
    if not isinstance(other, Tensor): other = Tensor(other)
    # Compute the output
    output = Tensor(self.value @ other.value, _children=(self, other), _op='matmul')
    # Define gradient function
    output._backward = grads.grad_fn_matmul
    return output
  
  def sum(self):
    output = Tensor(self.value.sum(), _children=(self,), _op='sum')
    output._backward = grads.grad_fn_sum
    return output

  def dot(self, other):
    if not isinstance(other, Tensor): other = Tensor(other)
    assert self.value.ndim == 1 and other.value.ndim == 1, \
    f'Dot product is intended for 1D vectors. Received {self.value.ndim} and {other.value.ndim} dim vectors.'
    assert self.value.shape[0] == other.value.shape[0], \
    f'Dot product is intended for vectors of the same length. Received {self.value.shape[0]} and {other.value.shape[0]}.'
    output = self * other
    output = output.sum()
    return output

  def to_dict(self):
    return {
        'value': self.value,
        'grad': self.grad,
        'label': self.label
    }

  @property
  def shape(self):
    return self.value.shape

  def backward(self):
    # Initialize the gradient of the output node
    self.grad = np.ones_like(self.value, dtype=np.float64)

    # Order graph nodes with topological sort
    ordered_nodes = []
    seen_nodes = set()
    def topo_sort(v):
      if v not in seen_nodes:
        seen_nodes.add(v)
        for child in v._children:
          topo_sort(child)
        ordered_nodes.append(v)
    # Start ordering from the last node
    topo_sort(self)
    # Reverse de order to do backward
    r_ordered_nodes: List[Tensor] = reversed(ordered_nodes)

    # Call _backward on every node in reversed topological order
    for node in r_ordered_nodes:
      if node._backward is None: continue

      grads = node._backward(node.grad, *[child.value for child in node._children])

      if node.retain_grads == False:
        node.grad = np.zeros_like(node.value, dtype=np.float64)

      for child, grad in zip(node._children, grads):
        for idx, (a_i, b_i) in enumerate(zip(reversed(child.grad.shape), reversed(grad.shape))):
          idx = -1*idx - 1
          if (a_i != b_i) and (a_i != 1) and (b_i != 1):
            raise ValueError(f'Cannot broadcast {child.grad.shape} and {grad.shape}.')
          elif (a_i == 1):
            grad = grad.sum(axis=idx, keepdims=True)

        child.grad += grad