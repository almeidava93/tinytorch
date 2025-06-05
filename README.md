# tinytorch

This is a personal learning endeavour inspired by Andrej Karpathy's micrograd series. It extends his idea by developing a tensor based auto-differentiation engine with the building blocks to train simple neural networks. I developed it also inspired by pytorch design, which focus on simplicity and flexibility. All built with numpy and good old Python from scratch. 

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

### Optimizers
Optimizers are objects that inherit from the abstract class `Optimizer`. They encapsulate the optimization algorithms used in deep learning. The implemented algorithms include:
- Gradient Descent
- Stochastic Gradient Descent
- Root Mean Square Propagation (RMSProp)
- Adam 
- Adam with decoupled weight decay Optimization (AdamW)

### Schedulers
The `Scheduler` object controls dynamic changes in the learning_rate. The implemented schedulers include:
- `ExponentialLR`: exponential decay of the learning rate

### `Module` object and neural network predefined layers
The `Module` object is the building block of neural networks. It defines a blueprint for every Module implemented. Includes parameter and submodule tracking, meaningful printer of the structure of each module and requires the implementation of a forward method for the forward pass. Also, it allows an arbritary composition of modules, allowing for great flexibility in building model architectures. Predefined layers built on top of the `Module` class includes:
- `Linear`