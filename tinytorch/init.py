from typing import Union
import numpy as np

from tinytorch.parameter import Parameter

def uniform(size: tuple = None, low: float = 0.0, high: float = 1.0, as_parameter: bool = True) -> Union[Parameter, np.ndarray]:
    if as_parameter:
        return Parameter(np.random.uniform(low, high, size))
    return np.random.uniform(low, high, size)

def normal(size: tuple = None, mean: float = 0.0, std: float = 1.0, as_parameter: bool = True) -> Union[Parameter, np.ndarray]:
    if as_parameter:
        return Parameter(np.random.normal(loc=mean, scale=std, size=size))
    return np.random.normal(loc=mean, scale=std, size=size)

def constant(size: tuple = None, value: float = 0.0, as_parameter: bool = True) -> Union[Parameter, np.ndarray]:
    if as_parameter:
        return Parameter(np.full(size, value))
    return np.full(size, value)

def ones(size: tuple = None, as_parameter: bool = True) -> Union[Parameter, np.ndarray]:
    if as_parameter:
        return Parameter(np.ones(size))
    return np.ones(size)

def zeros(size: tuple = None, as_parameter: bool = True) -> Union[Parameter, np.ndarray]:
    if as_parameter:
        return Parameter(np.zeros(size))
    return np.zeros(size)

def xavier_uniform(size: tuple = None, gain: float = 1.0, as_parameter: bool = True) -> Union[Parameter, np.ndarray]:
    # initialize bias with zeros
    if size[0] == 1:
        return zeros(size, as_parameter)
    # define bounds of uniform distribution
    fan_in = size[0]
    fan_out = size[1]
    bound = gain * np.sqrt(6/(fan_in + fan_out))
    return uniform(size, -bound, bound, as_parameter)

def xavier_normal(size: tuple = None, gain: float = 1.0, as_parameter: bool = True) -> Union[Parameter, np.ndarray]:
    # initialize bias with zeros
    if size[0] == 1:
        return zeros(size, as_parameter)
    # define bounds of normal distribution
    fan_in = size[0]
    fan_out = size[1]
    bound = gain * np.sqrt(2/(fan_in + fan_out))
    return normal(size, 0, bound, as_parameter)