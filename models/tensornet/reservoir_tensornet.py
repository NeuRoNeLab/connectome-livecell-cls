from typing import Type
import torch
from models.tensornet.functional_tensornet import TensorNetwork


def get_reservoir_tensornet_class(tn_class: Type[TensorNetwork] = TensorNetwork) -> Type[TensorNetwork]:
    """
    Creates a new class that extends a given TensorNetwork class, with a modified forward method
    that performs the forward pass without tracking gradients (i.e., in inference mode) and without trainable
    parameters. The primary use case of this function is to create an untrainable version of a TensorNetwork,
    acting as a highly non-linear reservoir.

    Note:
        - The dynamically created class inherits from the provided base class (`tn_class`), and is named
          "Reservoir{BaseClassName}", where `{BaseClassName}` is the name of the provided tensor network class.
        - The `__init__` method of the base class is called during the initialization of the new class.
        - The `forward` method is overridden to use `torch.no_grad()` context, calling the parent class's
          `forward` method within this context to prevent gradient computation.
        - The class name and qualified name are dynamically set to reflect the new class name
          (`Reservoir{BaseClassName}`).
        - This function assumes that the provided base class (`tn_class`) has a `forward` method that
          accepts a tensor as the first argument.
        - The generated class retains all other methods and properties of the base class.

    Args:
        tn_class (Type[TensorNetwork], optional): The base class to extend. Defaults to TensorNetwork.

    Returns:
        Type[TensorNetwork]: A new class that extends the provided TensorNetwork class with a modified
                             forward method.

    Example:
        >>> MyTensorNetwork = get_reservoir_tensornet_class(TensorNetwork)
        >>> model = TensorNetwork("celegans.graphml", 256, 512)
        >>> x_ = torch.randn(2, 3, 256)
        >>> output = model(x_)
        >>> output.shape
        >>> torch.Size([2, 3, 512])
        >>> output.requires_grad_
        >>> False
    """

    base_class_name = tn_class.__name__
    class_name = f"Reservoir{base_class_name}"

    class DummyReservoirTensorNetwork(tn_class):
        def __init__(self, *args, **kwargs):
            super(DummyReservoirTensorNetwork, self).__init__(*args, **kwargs)
            for param in self.parameters():
                param.requires_grad = False

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            with torch.no_grad():
                out = super().forward(x, *args, **kwargs)
            return out

    DummyReservoirTensorNetwork.__name__ = class_name
    complete_name = DummyReservoirTensorNetwork.__qualname__
    DummyReservoirTensorNetwork.__qualname__ = f"{'.'.join(complete_name.split('.')[:-1])}.{class_name}"

    return DummyReservoirTensorNetwork
