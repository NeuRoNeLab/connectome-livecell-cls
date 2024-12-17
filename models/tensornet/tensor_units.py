import abc
from typing import Optional, Union, Final, Tuple
import torch
from models.layers import PositionWiseFeedForward
from models.m2.m2_block import ConvResBlock, NormType, DecoderBlockType
from utils.connectome_reader import SENSOR_ROLE, INTERNEURON_ROLE, MOTOR_ROLE


SENSOR: Final[str] = SENSOR_ROLE
INTERNEURON: Final[str] = INTERNEURON_ROLE
MOTOR: Final[str] = MOTOR_ROLE


class NodeTensorUnitMixin(torch.nn.Module, abc.ABC):
    def __init__(self, neuron_type: str, *args, **kwargs):
        r"""
        An abstract mixin class representing the neuron tensor unit the TensorNetwork is based on.

        :param neuron_type: type of the neuron the tensor unit represents (either SENSOR, MOTOR or INTERNEURON)
        :type neuron_type: str
        """
        super().__init__(*args, **kwargs)  # forward unused args because this is a mixin class for multiple inheritance
        self.__neuron_type: str = neuron_type

    @property
    def neuron_type(self) -> str:
        return self.__neuron_type

    # @abc.abstractmethod
    # def forward(self, x, *args, **kwargs):
    #    raise NotImplementedError(f"Every {self.__class__.__name__} non-abstract subclass should implement forward().")


class FeedForwardNodeTensorUnit(NodeTensorUnitMixin):
    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 neuron_type: str,
                 out_features: Optional[int] = None,
                 dropout: float = 0.0,
                 activation: str = "gelu",
                 is_gated: bool = False,
                 bias1: bool = True,
                 bias2: bool = True,
                 bias_gate: bool = True,
                 normalize: bool = False,
                 pre_norm: bool = False):
        """
        A Position-wise feed-forward neural network module processing multiple embeddings in parallel, representing the
        neuron tensor unit the TensorNetwork is based on.

        Args:
            in_features (int): Size of each input sample.
            hidden_features (int): Size of the hidden layer of the feed-forward network.
            neuron_type (str): Type of the neuron the tensor unit represents (either SENSOR, MOTOR, or INTERNEURON).
            out_features (Optional[int]): Size of each output sample. Defaults to None.
            dropout (float): Dropout probability for the hidden layer. Defaults to 0.0.
            activation (str): Activation function for the hidden layer. Defaults to "gelu".
            is_gated (bool): Specifies whether the hidden layer is gated. Defaults to False.
            bias1 (bool): Whether the first fully connected layer should have a learnable bias. Defaults to True.
            bias2 (bool): Whether the second fully connected layer should have a learnable bias. Defaults to True.
            bias_gate (bool): Specifies whether the fully connected layer for the gate should have a learnable bias.
                Defaults to True.
            normalize (bool): Whether layer normalization should be applied. Defaults to False.
            pre_norm (bool): Whether the layer normalization should be applied before the skip-connection (pre-norm) or
                after it. Defaults to False.

        Shape:
            - Input: (*, H_{in}) where * means any number of dimensions, including none, and H_{in} = in_features.
            - Output: (*, H_{out}) where all but the last dimension are the same shape as the input, and
                H_{out} = out_features.

        Attributes:
            in_features (int): Size of each input sample.
            hidden_features (int): Size of the hidden layer of the feed-forward network.
            out_features (int): Size of each output sample. Defaults to in_features if out_features is None.
            activation (str): Activation function for the hidden layer.
            is_gated (bool): Specifies whether the hidden layer is gated.
            bias1 (bool): Whether the first fully connected layer should have a learnable bias.
            bias2 (bool): Whether the second fully connected layer should have a learnable bias.
            normalize (bool): Whether layer normalization should be applied.
            pre_norm (bool): Whether the layer normalization should be applied before the skip-connection (pre-norm) or
                after it.
            neuron_type (str): Type of the neuron the tensor unit represents (either SENSOR, MOTOR, or INTERNEURON).

        Examples::

            >>> m = FeedForwardNodeTensorUnit(20, 80, "INTERNEURON")
            >>> inp = torch.randn(128, 20)
            >>> out = m(inp)
            >>> print(out.size())
            torch.Size([128, 20])
            >>> print(m.neuron_type)
            'INTERNEURON'
        """
        super().__init__(neuron_type=neuron_type)

        # Store attributes not stored in FF layer
        self._out_features: Optional[int] = out_features
        self._normalize: bool = normalize

        self._ff = PositionWiseFeedForward(
            d_model=in_features,
            d_ff=hidden_features,
            dropout=dropout,
            activation=activation,
            is_gated=is_gated,
            bias1=bias1,
            bias2=bias2,
            bias_gate=bias_gate,
            pre_norm=pre_norm
        )

        self._projection = None
        if out_features is not None and out_features != in_features:
            self._projection = torch.nn.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=bias2
            )

    @property
    def in_features(self) -> int:
        return self._ff.d_model

    @property
    def hidden_features(self) -> int:
        return self._ff.d_ff

    @property
    def out_features(self) -> int:
        return self._out_features if self._out_features is not None else self.in_features

    @property
    def activation(self) -> str:
        return self._ff.activation

    @property
    def is_gated(self) -> bool:
        return self._ff.is_gated

    @property
    def bias1(self) -> bool:
        return self._ff.bias1

    @property
    def bias2(self) -> bool:
        return self._ff.bias2

    @property
    def normalize(self) -> bool:
        return self._normalize

    @property
    def pre_norm(self) -> bool:
        return self._ff.pre_norm

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass of the feed-forward neural network.

        Args:
            x (torch.Tensor): Input tensor of shape (*, H_{in}), where * means any number of dimensions, including none,
                and H_{in} = in_features.

        Returns:
            torch.Tensor: Output tensor of shape (*, H_{out}), where all but the last dimension are the same shape as
                the input, and H_{out} = out_features.
        """

        # Apply point-wise feed-foward block
        x = self._ff(x, normalize=self.normalize)

        # Apply projection
        if self._projection is not None:
            x = self._projection(x)

        return x

    def extra_repr(self) -> str:
        return 'in_features={}, hidden_features={}, out_features={}, neuron_type={}, activation={}, ' \
               'is_gated={}, bias1={}, bias2={}, normalize={}, pre_norm={}'.format(
                self.in_features, self.hidden_features, self.out_features, self.neuron_type, self.activation,
                self.is_gated, self.bias1, self.bias2, self.normalize, self.pre_norm)


class LinearNodeTensorUnit(NodeTensorUnitMixin, torch.nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 neuron_type: str,
                 bias: bool = True,
                 device: Optional[Union[str, torch.device]] = None,
                 dtype=None):
        # noinspection PyUnresolvedReferences
        r"""A simple Linear layer representing the neuron tensor unit the TensorNetwork is based on. Applies a linear
                    transformation to the incoming data: :math:`y = xA^T + b`

                    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

                    On certain ROCm devices, when using float16 inputs this module will use
                    :ref:`different precision<fp16_on_mi200>` for backward.

                    Args:
                        in_features: size of each input sample
                        out_features: size of each output sample
                        neuron_type: type of the neuron the tensor unit represents (either SENSOR, MOTOR or INTERNEURON)
                        bias: If set to ``False``, the layer will not learn an additive bias.
                            Default: ``True``

                    Shape:
                        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
                          dimensions including none and :math:`H_{in} = \text{in\_features}`.
                        - Output: :math:`(*, H_{out})` where all but the last dimension
                          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

                    Attributes:
                        weight: the learnable weights of the module of shape
                            :math:`(\text{out\_features}, \text{in\_features})`. The values are
                            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
                            :math:`k = \frac{1}{\text{in\_features}}`
                        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                                If :attr:`bias` is ``True``, the values are initialized from
                                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                                :math:`k = \frac{1}{\text{in\_features}}`
                        neuron_type: type of the neuron the tensor unit represents (either SENSOR, MOTOR or INTERNEURON)

                    Examples::

                        >>> m = LinearNodeTensorUnit(20, 30, INTERNEURON)
                        >>> inp = torch.randn(128, 20)
                        >>> out = m(inp)
                        >>> print(out.size())
                        torch.Size([128, 30])
                        >>> print(m.neuron_type)
                        'I'
                    """

        super().__init__(in_features=in_features,
                         out_features=out_features,
                         bias=bias,
                         device=device,
                         dtype=dtype,
                         neuron_type=neuron_type)

        # self.__neuron_type: str = neuron_type

    # @property
    # def neuron_type(self) -> str:
    #    return self.__neuron_type

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, neuron_type={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.neuron_type
        )


class ConvNodeTensorUnit(NodeTensorUnitMixin, ConvResBlock):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 neuron_type: str,
                 width_scale: float = 1.0,
                 kernel_size: Union[int, Tuple[int, ...]] = (3, 3),
                 n_convs: int = 1,
                 pool_size: Optional[Union[int, Tuple[int, ...]]] = None,
                 activation: str = "gelu",
                 residual: bool = False,
                 norm_type: NormType = NormType.NONE,
                 dropout: float = 0.0,
                 decoder_block_type: Optional[DecoderBlockType] = None,
                 stage_before_pool_or_unpool: bool = True):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            width_scale=width_scale,
            kernel_size=kernel_size,
            n_convs=n_convs,
            pool_size=pool_size,
            activation=activation,
            residual=residual,
            norm_type=norm_type,
            dropout=dropout,
            decoder_block_type=decoder_block_type,
            stage_before_pool_or_unpool=stage_before_pool_or_unpool,
            neuron_type=neuron_type
        )
