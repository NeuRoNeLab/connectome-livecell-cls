import math
from typing import Union, Tuple, List, Optional, Final, Dict
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as nnf

_ACTIVATIONS: Final[Dict[str, torch.nn.Module]] = {
    "linear": torch.nn.Identity(),
    "relu": torch.nn.ReLU(),
    "leaky_relu": torch.nn.LeakyReLU(),
    "rrelu": torch.nn.RReLU(),
    "relu6": torch.nn.ReLU6(),
    "gelu": torch.nn.GELU(),
    "elu": torch.nn.ELU(),
    "celu": torch.nn.CELU(),
    "glu": torch.nn.GLU(),
    "selu": torch.nn.SELU(),
    "prelu": torch.nn.PReLU(),
    "silu": torch.nn.SiLU(),
    "hardswish": torch.nn.Hardswish(),
    "tanh": torch.nn.Tanh(),
    "sigmoid": torch.nn.Sigmoid(),
    "log_sigmoid": torch.nn.LogSigmoid(),
    "softmax": torch.nn.Softmax(dim=-1),
    "hardtanh": torch.nn.Hardtanh()
}


def resolve_activation(activation: str) -> torch.nn.Module:
    return _ACTIVATIONS[activation]


def nans(shape: Union[int, Tuple[int, ...], torch.Size, List[int]],
         dtype: Optional[torch.dtype] = None,
         layout: torch.layout = torch.strided,
         device: Optional[torch.device] = None,
         requires_grad: bool = False) -> torch.Tensor:
    """
    Returns a tensor filled with the scalar value nan, with the shape defined by the given argument.

    :param shape: A sequence of integers defining the shape of the output tensor.
    :type shape: Union[int, Tuple[int, ...]
    :param dtype: The desired data type of returned tensor. Default: if None, uses a global default (see
        torch.set_default_tensor_type()).
    :type dtype: Optional[torch.dtype]
    :param layout: the desired layout of returned Tensor. Default: torch.strided.
    :type layout: torch.layout
    :param device: The desired device of returned tensor. Default: if None, uses the current device for the default
        tensor type (see torch.set_default_tensor_type()). device will be the CPU for CPU tensor types and the current
        CUDA device for CUDA tensor types.
    :type device: Optional[torch.device]
    :param requires_grad: Whether autograd should record operations on the returned tensor. Default: False.
    :type requires_grad: bool

    :return: A tensor filled with the scalar value nan, with the given shape.
    :rtype: torch.Tensor
    """
    return torch.zeros(shape, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad) + torch.nan


def nans_like(tensor: torch.Tensor,
              dtype: Optional[torch.dtype] = None,
              layout: torch.layout = torch.strided,
              device: Optional[torch.device] = None,
              requires_grad: bool = False,
              memory_format: torch.memory_format = torch.preserve_format) -> torch.Tensor:
    """
    Returns a tensor filled with the scalar value nan, with the shape defined by the input tensor.

    :param tensor: A tensor whose size size will determine size of the output tensor.
    :type tensor: torch.Tensor
    :param dtype: The desired data type of returned tensor. Default: if None, uses a global default (see
        torch.set_default_tensor_type()).
    :type dtype: Optional[torch.dtype]
    :param layout: The desired layout of returned Tensor. Default: torch.strided.
    :type layout: torch.layout
    :param device: The desired device of returned tensor. Default: if None, uses the current device for the default
        tensor type (see torch.set_default_tensor_type()). device will be the CPU for CPU tensor types and the current
        CUDA device for CUDA tensor types.
    :type device: Optional[torch.device]
    :param requires_grad: Whether autograd should record operations on the returned tensor. Default: False.
    :type requires_grad: bool
    :param memory_format: The desired memory format of returned Tensor. Default: torch.preserve_format.
    :type memory_format: torch.memory_format

    :return: A tensor filled with the scalar value nan, with the given shape.
    :rtype: torch.Tensor
    """
    return torch.zeros_like(
        tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
        memory_format=memory_format
    ) + torch.nan


class SerializableModule(torch.nn.Module, ABC):
    # TODO: all subclasses of this should be refactored to dynamically use the "eval" func to get the class from string
    def __init__(self, *args, **kwargs):
        """
        An abstract mixin class representing a PyTorch module whose constructor arguments are serializable through
        dictionary format.
        """
        super(SerializableModule, self).__init__(*args, **kwargs)

    @abstractmethod
    def serialize_constructor_params(self, *args, **kwargs) -> dict:
        """
        Returns a dictionary of the parameters that were passed to the constructor of the module.
        """
        raise NotImplementedError(
            f"Any non-abstract {self.__class__.__name__} subclass must implement serialize_constructor_params() method."
        )

    @classmethod
    def from_constructor_params(cls, constructor_params: dict, *args, **kwargs):
        """
        Takes a dictionary of constructor parameters and optional additional args and kwargs, and returns an instance of
        the class with the constructor parameters set to the values in the dictionary and the given args/kwargs.

        :param constructor_params: a dictionary of the parameters that will be passed to the constructor
        :type constructor_params: dict

        :return: An instance of the class with the constructor parameters set to the values in the dictionary and the
            given args/kwargs.
        """
        return cls(*args, **constructor_params, **kwargs)


def pad_to_match_height_width(a: torch.Tensor,
                              b: torch.Tensor,
                              mode: str = 'constant',
                              value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads two input tensors `a` and `b` along their height and width dimensions to match their sizes. Padding is applied
    along the height and width dimensions of the tensors to make them match in size. If the height or width difference
    between the tensors is odd, the extra padding is added to the bottom/right side.

     Args:
         a (torch.Tensor): Input tensor a with shape (..., height_a, width_a).
         b (torch.Tensor): Input tensor b with shape (..., height_b, width_b).
         mode (str, optional): The padding mode to use (either 'constant', 'reflect', 'replicate' or 'circular').
            Default is 'constant'.
         value (float, optional): The value to fill the padding with in 'constant' mode. Default is 0.0.


     Returns:
         Tuple[torch.Tensor, torch.Tensor]: Tuple containing two tensors `a` and `b` after padding to match their sizes,
         both with shape (..., max(height_a, height_b), max(width_a, width_b))

     Examples:
         >>> t_a = torch.randn(3, 2, 5)  # Tensor with shape [3, 2, 5]
         >>> t_b = torch.randn(3, 4, 3)  # Tensor with shape [3, 4, 3]
         >>> t_a_padded, t_b_padded = pad_to_match_height_width(t_a, t_b)
         >>> t_a_padded.shape
         >>> (3, 4, 5)
         >>> t_b_padded.shape
         >>> (3, 4, 5)
     """
    height_a, width_a = a.shape[-2], a.shape[-1]
    height_b, width_b = b.shape[-2], b.shape[-1]

    # Compute the right/left width differences and the up/down height differences
    diff_height = height_a - height_b
    diff_width = width_a - width_b
    diff_height_up = math.floor(diff_height / 2)
    diff_height_down = math.ceil(diff_height / 2)
    diff_width_r = math.floor(diff_width / 2)
    diff_width_l = math.ceil(diff_width / 2)

    # Construct b tensor padding and pad it if needed
    pad_b = (max(0, diff_width_l), max(0, diff_width_r), max(0, diff_height_up), max(0, diff_height_down))
    if sum(pad_b) > 0:
        b = nnf.pad(b, pad=pad_b, mode=mode, value=value)

    # Construct the a tensor padding and pad it if needed
    pad_a = (max(0, -diff_width_l), max(0, -diff_width_r), max(0, -diff_height_up), max(0, -diff_height_down))
    if sum(pad_a) > 0:
        a = nnf.pad(a, pad=pad_a, mode=mode, value=value)

    return a, b


def pad_h_w_to_nearest_power_of_2(t: torch.Tensor,
                                  squared: bool = True,
                                  mode: str = 'constant',
                                  value: float = 0.0) -> torch.Tensor:
    """
    Pads the input tensor `t` along its height and width dimensions to the nearest power of 2. If `squared` is set to
    True, it pads to the nearest power of 2 in a squared manner, ensuring that both height and width are padded to the
    same value (the maximum of the two nearest powers of 2). If `squared` is False, it pads to the nearest power of 2
    independently for height and width.

    Args:
        t (torch.Tensor): Input tensor with shape (..., height, width).
        squared (bool, optional): Flag indicating whether to pad to the nearest power of 2 in a squared manner.
                                  If True, pads both height and width to the maximum of the two. Defaults to True.
        mode (str, optional): The padding mode to use (either 'constant', 'reflect', 'replicate' or 'circular').
            Default is 'constant'.
        value (float, optional): The value to fill the padding with in 'constant' mode. Default is 0.0.

    Returns:
        torch.Tensor: Padded tensor with shape (..., padded_height, padded_width).

    Examples:
        >>> t_ = torch.randn(3, 4, 5)  # Tensor with shape [3, 4, 5]
        >>> padded_t_ = pad_h_w_to_nearest_power_of_2(t_, squared=True)
        >>> padded_t_.shape
        >>> (3, 8, 8)  # Height and width padded to the nearest power of 2, both squared to the maximum of the two.
    """
    height, width = t.shape[-2], t.shape[-1]

    # Compute the nearest power of 2 height/width
    factor_h = math.ceil(math.log2(height))
    factor_w = math.ceil(math.log2(width))
    h_power2 = height * factor_h
    w_power2 = width * factor_w

    # Make desired height/width equal if squared padding is needed
    if squared:
        h_power2 = max(h_power2, w_power2)
        w_power2 = max(h_power2, w_power2)

    # Pad to the desired height/width creating a dummy tensor
    t, _ = pad_to_match_height_width(
        a=t,
        b=torch.zeros(t.shape[:-2] + (h_power2, w_power2), device=t.device),
        mode=mode,
        value=value
    )

    return t


def pad_to_squared(x: torch.Tensor, mode: str = 'constant', value: float = 0.0) -> torch.Tensor:
    """
    Pads the input tensor `x` to make its height and width dimensions equal, creating a squared tensor. If the height is
    greater than the width, it pads the width dimension to match the height, and vice-versa. The padding is applied
    symmetrically on both sides of the tensor.

    Args:
        x (torch.Tensor): Input tensor with shape (..., height, width).
        mode (str, optional): The padding mode to use (either 'constant', 'reflect', 'replicate' or 'circular').
            Default is 'constant'.
        value (float, optional): The value to fill the padding with in 'constant' mode. Default is 0.0.

    Returns:
        torch.Tensor: Padded tensor with shape (..., padded_height, padded_width).

    Examples:
        >>> x_ = torch.randn(3, 4, 5)  # Tensor with shape [3, 4, 5]
        >>> padded_x_ = pad_to_squared(x_)
        >>> padded_x_.shape
        >>> (3, 5, 5)  # Tensor padded to have equal height and width dimensions.
    """
    if x.shape[-2] > x.shape[-1]:
        diff = x.shape[-2] - x.shape[-1]
        x = nnf.pad(x, pad=(math.ceil(diff / 2), math.floor(diff / 2), 0, 0), mode=mode, value=value)

    elif x.shape[-2] < x.shape[-1]:
        diff = x.shape[-2] - x.shape[-1]
        x = nnf.pad(x, pad=(0, 0, math.ceil(diff / 2), math.floor(diff / 2)), mode=mode, value=value)

    return x


def interpolate_pos_encoding(patch_size: int,
                             position_embeddings_weights: torch.Tensor,
                             embeddings: torch.Tensor,
                             height: int,
                             width: int,
                             eps: float = 0.1) -> torch.Tensor:
    """
    Interpolates the pre-trained position encodings to adapt the model for higher resolution images than it was
    originally trained on. It interpolates the positional embeddings to match the new image dimensions.

    Source:
    https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174

    Args:
        patch_size (int): The size of each patch in the image.
        position_embeddings_weights (torch.Tensor): The pre-trained position embeddings.
        embeddings (torch.Tensor): The embeddings from the model.
        height (int): The height of the new image.
        width (int): The width of the new image.
        eps (float, optional): A small number added to avoid floating point errors during interpolation. Default is 0.1.

    Returns:
        torch.Tensor: The interpolated position embeddings.

    Example:
        >>> patch_size_ = 16
        >>> position_embeddings_weights_ = torch.randn(1, 197, 768)  # Example tensor
        >>> embeddings_ = torch.randn(1, 197, 768)  # Example tensor
        >>> height_ = 256
        >>> width_ = 256
        >>> interpolated_embeddings_ = interpolate_pos_encoding(patch_size_,
        >>>                                                     position_embeddings_weights_,
        >>>                                                     embeddings_,
        >>>                                                     height_,
        >>>                                                     width_)
        >>> print(interpolated_embeddings_.shape)
        torch.Size([1, 256, 768])
    """

    num_patches = embeddings.shape[1] - 1
    num_positions = position_embeddings_weights.shape[1] - 1
    if num_patches == num_positions and height == width:
        return position_embeddings_weights
    class_pos_embed = position_embeddings_weights[:, 0]
    patch_pos_embed = position_embeddings_weights[:, 1:]
    dim = embeddings.shape[-1]
    h0 = height // patch_size
    w0 = width // patch_size

    # Add a small number to avoid floating point errors in the interpolation, see discussion at
    # https://github.com/facebookresearch/dino/issues/8
    h0, w0 = h0 + eps, w0 + eps
    patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
    patch_pos_embed = nnf.interpolate(
        patch_pos_embed,
        scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
        mode="bicubic",
        align_corners=False,
    )
    assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


class ParallelModules(torch.nn.Module):
    def __init__(self, *fns):
        """
        A custom PyTorch module that applies multiple functions in parallel to the input tensor.

        Args:
            *fns (torch.nn.Module): Variable length argument list of PyTorch modules to be applied in parallel.

        Attributes:
            fns (torch.nn.ModuleList): A list of the provided PyTorch modules.
        """
        super().__init__()
        self.fns = torch.nn.ModuleList(fns)

    def forward(self, x, *args, **kwargs):
        """
        Applies each module in `self.fns` to the input tensor `x` in parallel.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            map: A map object containing the results of applying each function to `x`.
        """
        return map(lambda fn: fn(x, *args, **kwargs), self.fns)


class ParallelSum(ParallelModules):
    def __init__(self, *fns):
        """
        A custom PyTorch module that applies multiple functions in parallel to the input tensor
        and sums the results.

        Args:
            *fns (torch.nn.Module): Variable length argument list of PyTorch modules to be applied in parallel.

        Inherits from:
            ParallelModules

        Attributes:
            fns (torch.nn.ModuleList): A list of the provided PyTorch modules.
        """
        super().__init__(*fns)

    def forward(self, x, *args, **kwargs):
        """
        Applies each module in `self.fns` to the input tensor `x` in parallel and sums the results.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The sum of the results of applying each function to `x`.
        """
        return sum(super().forward(x, *args, **kwargs))
