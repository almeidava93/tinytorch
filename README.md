# tinytorch

This is a personal learning endeavour inspired by Andrej Karpathy's micrograd series. It extends his idea by developing a tensor based auto-differentiation engine with the building blocks to train simple neural networks. I developed it also inspired by pytorch design, which focus on simplicity and flexibility. 

## Supported features

### Tensor
Basic building block for all data operations. It extends numpy array's functionalities by:
- defining tensor operations and their gradient computations methods
- keeping track of the computational graph
- defining methods for automatic gradient computation in the backward pass
- defining gradient accumulation rules taking into consideration array broadcasting 

Every Tensor has its gradient computed during back propagation. This can be turned of by setting `requires_grad = False`.

Tensors do not store intermediate gradient values, except if they are Parameters (see next section). This can be enabled by setting `retain_grads = True`.

### Parameter
A Parameter object is an extension of the tensor object used to define learnable parameters. These objects are updated by the optimizer. 