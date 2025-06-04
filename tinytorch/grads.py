import numpy as np

def grad_fn_add(output_grad: np.array = None, first_value: np.array = None, second_value: np.array = None) -> tuple[np.array]:
  """
  Derivative of the add operation. Applies chain rule to the input gradients by multiplying with the output gradient.
  """
  return 1*output_grad, 1*output_grad

def grad_fn_sub(output_grad: np.array = None, first_value: np.array = None, second_value: np.array = None) -> tuple[np.array]:
  """
  Derivative of the subtract operation. Applies chain rule to the input gradients by multiplying with the output gradient.
  """
  return 1*output_grad, -1*output_grad

def grad_fn_mul(output_grad: np.array = None, first_value: np.array = None, second_value: np.array = None) -> tuple[np.array]:
  """
  Derivative of the multiply operation. Applies chain rule to the input gradients by multiplying with the output gradient.
  """
  return second_value*output_grad, first_value*output_grad

def grad_fn_truediv(output_grad: np.array = None, first_value: np.array = None, second_value: np.array = None) -> tuple[np.array]:
  """
  Derivative of the divide operation. Applies chain rule to the input gradients by multiplying with the output gradient.
  """
  return (1/second_value)*output_grad, (-first_value/(second_value**2))*output_grad

def grad_fn_pow(output_grad: np.array = None, first_value: np.array = None, second_value: np.array = None) -> tuple[np.array]:
  """
  Derivative of the power operation. Applies chain rule to the input gradients by multiplying with the output gradient.
  """
  assert second_value.dtype == int, \
  f"The power must be an integer. Received type {second_value.dtype}, value {second_value}"

  return second_value*first_value**(second_value-1)*output_grad, ((first_value**second_value)*np.log(first_value)*output_grad).sum()

def grad_fn_matmul(output_grad: np.array = None, first_value: np.array = None, second_value: np.array = None) -> tuple[np.array]:
  """
  Derivative of the matmul operation. Applies chain rule to the input gradients by multiplying with the output gradient.
  """
  return output_grad@second_value.T, first_value.T@output_grad

def grad_fn_sum(output_grad: np.array = None, first_value: np.array = None, second_value: np.array = None) -> tuple[np.array]:
  """
  Derivative of the sum operation. Applies chain rule to the input gradients by multiplying with the output gradient.
  """
  return np.full_like(first_value, output_grad), None

def grad_fn_sigmoid(output_grad: np.array = None, first_value: np.array = None, second_value: np.array = None) -> tuple[np.array]:
  """
  Derivative of the sigmoid operation. Applies chain rule to the input gradients by multiplying with the output gradient.
  """
  sigmoid = lambda x: 1/(1+np.exp(-x))
  return sigmoid(first_value)*(1-sigmoid(first_value))*output_grad,

def grad_fn_silu(output_grad: np.array = None, first_value: np.array = None, second_value: np.array = None) -> tuple[np.array]:
  """
  Derivative of the SiLU operation. Applies chain rule to the input gradients by multiplying with the output gradient.
  """
  sigmoid = lambda x: 1/(1+np.exp(-x))
  x = first_value
  silu_grad = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))
  return silu_grad*output_grad,