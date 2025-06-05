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
    if len(size) > 2:
        raise ValueError(f"Initialization for {len(size)}D tensor is not supported")

    # initialize bias with zeros
    if size[0] == 1:
        return zeros(size, as_parameter)
    
    # define bounds of uniform distribution
    fan_in = size[0]
    fan_out = size[1]
    bound = gain * np.sqrt(6/(fan_in + fan_out))
    return uniform(size, -bound, bound, as_parameter)

def xavier_normal(size: tuple = None, gain: float = 1.0, as_parameter: bool = True) -> Union[Parameter, np.ndarray]:
    if len(size) > 2:
        raise ValueError(f"Initialization for {len(size)}D tensor is not supported")
    
    # initialize bias with zeros
    if size[0] == 1:
        return zeros(size, as_parameter)
    
    # define bounds of normal distribution
    fan_in = size[0]
    fan_out = size[1]
    bound = gain * np.sqrt(2/(fan_in + fan_out))
    return normal(size, 0, bound, as_parameter)

def get_nonlinearity_gain(nonlinearity: str, alpha: float = 0.0):
    """
    From recommendations in pytorch docs. See https://docs.pytorch.org/docs/stable/nn.init.html#torch.nn.init.calculate_gain
    """
    if nonlinearity == 'relu':
        return np.sqrt(2)
    elif nonlinearity == 'selu':
        return 3/4
    elif nonlinearity == 'tanh':
        return 5/3
    elif nonlinearity in ['sigmoid', 'linear', 'conv1d', 'conv2d', 'conv3d']:
        return 1.0
    elif nonlinearity == 'leaky_relu':
        return np.sqrt(2 / (1 + alpha**2))
    else:
        raise ValueError(f"Nonlinearity '{nonlinearity}' is not supported")

def kaiming_uniform(size: tuple = None, mode: str = 'fan_in', nonlinearity: str = 'relu', as_parameter: bool = True, alpha: float = 0.0, *args, **kwargs):
    if len(size) > 2:
        raise ValueError(f"Initialization for {len(size)}D tensor is not supported")
    
    # initialize bias with zeros
    if size[0] == 1:
        return zeros(size, as_parameter)
    
    # get mode
    fan = None
    if mode == 'fan_in': # mantains weights variance during the forward pass
        fan = size[0]
    elif mode == 'fan_out': # mantains weights variance during the backward pass
        fan = size[1]
    else:
        raise ValueError(f"Mode '{mode}' is not supported")
    
    # define gain and bound
    gain = get_nonlinearity_gain(nonlinearity, alpha, *args, **kwargs)
    bound = gain * np.sqrt(3/fan)

    return uniform(size, -bound, bound, as_parameter)

def kaiming_normal(size: tuple = None, mode: str = 'fan_in', nonlinearity: str = 'relu', as_parameter: bool = True, alpha: float = 0.0, *args, **kwargs):
    if len(size) > 2:
        raise ValueError(f"Initialization for {len(size)}D tensor is not supported")
    
    # initialize bias with zeros
    if size[0] == 1:
        return zeros(size, as_parameter)

    # get mode
    fan = None
    if mode == 'fan_in': # mantains weights variance during the forward pass
        fan = size[0]
    elif mode == 'fan_out': # mantains weights variance during the backward pass
        fan = size[1]
    else:
        raise ValueError(f"Mode '{mode}' is not supported")
    
    # define gain and bound
    gain = get_nonlinearity_gain(nonlinearity, alpha, *args, **kwargs)
    std = gain / np.sqrt(fan)
    mean = 0

    return normal(size, mean, std, as_parameter)