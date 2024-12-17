from typing import Tuple, Final, Union, Sequence
import torch
from models.layers.misc import SerializableModule, resolve_activation
from models.layers.conv import MaxPool2dSamePadding, Conv2dSame


IMPALA_DIMS: Final[Tuple[int, ...]] = (16, 32, 32)
NONE_NORM: Final[str] = 'none'
BATCH_NORM: Final[str] = 'batch'
GROUP_NORM: Final[str] = 'group'
LAYER_NORM: Final[str] = 'layer'


class ImpalaResidualBlock(torch.nn.Module):
    """
    Implementation of a residual block used in ImpalaResNet architectures.

    Args:
        dim (int): Number of input channels.
        kernel_size (Union[int, Tuple[int, ...]], optional): Size of the convolutional kernel.
            Defaults to (3, 3).
        norm_type (str, optional): Type of normalization to use. Choose from NONE_NORM, LAYER_NORM, GROUP_NORM,
            BATCH_NORM.  Defaults to NONE_NORM.
        fixup_init (bool, optional): If True, initialize the second convolution weights to zero.
            Defaults to False.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        activation_fn (torch.nn.Module, optional): Activation function. Defaults to torch.nn.ReLU().
        start_with_activation (bool, optional): If True, start the block with activation and normalization.
            Defaults to True.

    Attributes:
        dim (int): Number of input channels.
        kernel_size (Union[int, Tuple[int, ...]]): Size of the convolutional kernel.
        norm_type (str): Type of normalization used.
        fixup_init (bool): If True, the second convolution weights are initialized to zero.
        dropout (float): Dropout probability.
        start_with_activation (bool): If True, the block starts with activation and normalization.
        activation_fn (torch.nn.Module): Activation function.

    Examples:
        >>> block = ImpalaResidualBlock(dim=64)
        >>> input_tensor = torch.randn((16, 64, 16, 16))
        >>> output_tensor = block(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([16, 64, 16, 16])
    """

    def __init__(self,
                 dim: int,
                 kernel_size: Union[int, Tuple[int, ...]] = (3, 3),
                 norm_type: str = NONE_NORM,
                 fixup_init: bool = False,
                 dropout: float = 0.0,
                 activation_fn: torch.nn.Module = torch.nn.ReLU(),
                 start_with_activation: bool = True):
        super().__init__()

        # Store attributes
        self._dim: int = dim
        self._kernel_size: Union[int, Tuple[int, ...]] = kernel_size
        self._norm_type: str = norm_type
        self._dropout: float = dropout
        self._fixup_init: bool = fixup_init
        self._start_with_activation: bool = start_with_activation

        # Initialize first convolution
        self._conv0 = torch.nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            stride=1,
            padding="same"
        )

        # Initialize second convolution, initializing weights to zero if `fixup_init` is True
        self._conv1 = torch.nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            stride=1,
            padding="same"
        )
        if fixup_init:
            torch.nn.init.zeros_(self._conv1.weight)
            # torch.nn.init.zeros_(self._conv1.bias)

        # Initialize normalization layer
        # if norm_type == LAYER_NORM:
        #    self._norm = torch.nn.LayerNorm(normalized_shape=dim, eps=1e-5)

        if norm_type == GROUP_NORM:
            self._norm0 = torch.nn.GroupNorm(num_channels=dim, num_groups=max(1, int(dim // 8)), eps=1e-5)
            self._norm1 = torch.nn.GroupNorm(num_channels=dim, num_groups=max(1, int(dim // 8)), eps=1e-5)

        elif norm_type == BATCH_NORM:
            self._norm0 = torch.nn.BatchNorm2d(num_features=dim, eps=1e-5)
            self._norm1 = torch.nn.BatchNorm2d(num_features=dim, eps=1e-5)

        elif self.norm_type == NONE_NORM:
            self._norm0 = lambda x: x
            self._norm1 = self._norm0

        # Initialize dropout layer
        self._dropout_layer = torch.nn.Dropout(p=dropout)

        # Store activation function
        self._activation_fn = activation_fn

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def kernel_size(self) -> Union[int, Tuple[int, ...]]:
        return self._kernel_size

    @property
    def norm_type(self) -> str:
        return self._norm_type

    @property
    def fixup_init(self) -> bool:
        return self._fixup_init

    @property
    def dropout(self) -> float:
        return self._dropout

    @property
    def start_with_activation(self) -> bool:
        return self._start_with_activation

    @property
    def activation_fn(self) -> torch.nn.Module:
        return self._activation_fn

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Performs a forward pass through the ImpalaResidualBlock, applying the following operations:
            1. If `start_with_activation` is True:
                1.1 Apply the specified activation function.
                1.2 Apply normalization based on the chosen normalization type.
                1.3 Apply dropout if specified.
            2. Perform the first convolution operation.
            3. Apply the specified activation function.
            4. Apply normalization based on the chosen normalization type.
            5. Perform the second convolution operation.
            6. Add the input tensor to the output tensor as a skip-connection.
            7. If `start_with_activation` is False:
                7.1 Apply the specified activation function.
                7.2 Apply normalization based on the chosen normalization type.
                7.3 Apply dropout if specified.

        Example:
            >>> block = ImpalaResidualBlock(dim=64)
            >>> input_tensor = torch.randn((16, 64, 16, 16))
            >>> output_tensor = block(input_tensor)
            >>> print(output_tensor.shape)
            torch.Size([16, 64, 16, 16])

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_channels, output_height, output_width).
        """
        block_input = x
        conv_out = block_input

        # If the input is the output of another layer
        if self.start_with_activation:
            # Apply activation
            conv_out = self._activation_fn(conv_out)

            # Apply normalization
            conv_out = self._norm0(conv_out)

            conv_out = self._dropout_layer(conv_out)

        # Perform convolution, apply activation and normalize
        conv_out = self._conv0(conv_out)

        # Apply activation
        conv_out = self._activation_fn(conv_out)

        # Apply normalization
        conv_out = self._norm1(conv_out)

        # Apply second convolution
        conv_out = self._conv1(conv_out)

        # Apply skip-connection
        conv_out += block_input

        # If the input isn't the output of another layer
        if not self.start_with_activation:
            # Apply activation
            conv_out = self._activation_fn(conv_out)

            # Apply normalization
            conv_out = self._norm0(conv_out)

            conv_out = self._dropout_layer(conv_out)

        return conv_out


class ImpalaResidualStage(torch.nn.Module):
    """
    A single residual stage for an Impala-style ResNet.

    Args:
        input_dim (int): Number of input channels.
        dim (int): Number of output channels.
        num_blocks (int, optional): Number of residual blocks in the stage. Defaults to 2.
        kernel_size (Union[int, Tuple[int, int]], optional): Size of the convolutional kernel.
            Defaults to (3, 3).
        use_max_pooling (bool, optional): If True, apply max pooling after the first convolution. Defaults to True.
        norm_type (str, optional): Type of normalization to use in residual blocks.
            Choose from NONE_NORM, LAYER_NORM, GROUP_NORM, BATCH_NORM. Defaults to NONE_NORM.
        fixup_init (bool, optional): If True, initialize the weights of the second convolution in each residual block to
            zero. Defaults to False.
        dropout (float, optional): Dropout probability in residual blocks. Defaults to 0.0.
        activation_fn (torch.nn.Module, optional): Activation function used in residual blocks. Defaults to
            torch.nn.ReLU().

    Attributes:
        input_dim (int): Number of input channels.
        dim (int): Number of output channels.
        num_blocks (int): Number of residual blocks in the stage.
        kernel_size (Union[int, Tuple[int, int]]): Size of the convolutional kernel.
        use_max_pooling (bool): If True, max pooling is applied after the first convolution.
        norm_type (str): Type of normalization used in residual blocks.
        fixup_init (bool): If True, the weights of the second convolution in each residual block are initialized to
            zero.
        dropout (float): Dropout probability in residual blocks.
        activation_fn (torch.nn.Module): Activation function used in residual blocks.

    Examples:
        >>> stage = ImpalaResidualStage(input_dim=64, dim=128)
        >>> input_tensor = torch.randn((16, 64, 16, 16))
        >>> output_tensor = stage(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([16, 128, 8, 8])
    """

    def __init__(self,
                 input_dim: int,
                 dim: int,
                 num_blocks: int = 2,
                 kernel_size: Union[int, Tuple[int, int]] = (3, 3),
                 use_max_pooling: bool = True,
                 norm_type: str = NONE_NORM,
                 fixup_init: bool = False,
                 dropout: float = 0.0,
                 activation_fn: torch.nn.Module = torch.nn.ReLU()):
        super().__init__()

        if num_blocks < 1:
            raise ValueError(f"`num_blocks` must be > 0. {num_blocks} given.")

        # Store attributes
        self._num_blocks: int = num_blocks
        self._use_max_pooling: bool = use_max_pooling

        # Initialize layers
        self._first_conv = torch.nn.Conv2d(
            in_channels=input_dim,
            out_channels=dim,
            kernel_size=kernel_size,
            stride=1,
            padding="same"
        )

        self._max_pool = None
        if self.use_max_pooling:
            self._max_pool = MaxPool2dSamePadding(kernel_size=kernel_size, stride=(2, 2))

        # Initialize residual blocks
        self._residual_blocks = torch.nn.ModuleList()
        for _ in range(self.num_blocks):
            block = ImpalaResidualBlock(
                dim=dim,
                kernel_size=kernel_size,
                norm_type=norm_type,
                fixup_init=fixup_init,
                dropout=dropout,
                activation_fn=activation_fn,
                start_with_activation=True
            )
            self._residual_blocks.append(block)

    @property
    def input_dim(self) -> int:
        return self._first_conv.in_channels

    @property
    def dim(self) -> int:
        return self._first_conv.out_channels

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    @property
    def kernel_size(self) -> Union[int, Tuple[int, ...]]:
        return self._first_conv.kernel_size

    @property
    def use_max_pooling(self) -> bool:
        return self._use_max_pooling

    @property
    def norm_type(self) -> str:
        return self._residual_blocks[0].norm_type

    @property
    def fixup_init(self) -> bool:
        return self._residual_blocks[0].fixup_init

    @property
    def dropout(self) -> float:
        return self._residual_blocks[0].dropout

    @property
    def activation_fn(self) -> torch.nn.Module:
        return self._residual_blocks[0].activation_fn

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Performs a forward pass through the ImpalaResidualStage, applying the following operations:
            1. Apply the first convolutional layer.
            2. Apply max pooling if `use_max_pooling` is True.
            3. Apply the specified number of ImpalaResidualBlocks sequentially.
                Each ImpalaResidualBlock consists of:
                3.1 Forward pass through the first convolution.
                3.2 Forward pass through the second convolution.
                3.3 Apply skip-connection.

        Example:
            >>> stage = ImpalaResidualStage(input_dim=64, dim=128)
            >>> input_tensor = torch.randn((16, 64, 16, 16))
            >>> output_tensor = stage(input_tensor)
            >>> print(output_tensor.shape)
            torch.Size([16, 128, 8, 8])

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_channels, output_height, output_width).
        """
        out = x

        # Apply first convolution
        out = self._first_conv(out)

        # Apply max pooling if required
        if self._max_pool is not None:
            out = self._max_pool(out)

        # Apply residual blocks
        for block in self._residual_blocks:
            out = block(out)

        return out


class ImpalaResNet(SerializableModule):
    """
    ResNet encoder based on Impala.

    Args:
        input_dim (int): Number of input channels.
        width_scale (int, optional): Width scale factor for the residual stages. Defaults to 1.
        dims (Tuple[int, ...], optional): Tuple of integers specifying the dimensions of each residual stage.
            Defaults to IMPALA_DIMS.
        kernel_size (Tuple[int, int], optional): Size of the convolutional kernel. Defaults to (3, 3).
        use_max_pooling (bool, optional): If True, apply max pooling after the first convolution in each residual stage.
            Defaults to True.
        num_blocks (int, optional): Number of residual blocks in each stage. Defaults to 2.
        norm_type (str, optional): Type of normalization to use in residual blocks.
            Choose from NONE_NORM, LAYER_NORM, GROUP_NORM, BATCH_NORM. Defaults to NONE_NORM.
        fixup_init (bool, optional): If True, initialize the weights of the second convolution in each residual block
            to zero. Defaults to False.
        dropout (float, optional): Dropout probability in residual blocks. Defaults to 0.0.
        activation (str, optional): Activation function used in residual blocks.
            Defaults to "relu".

    Attributes:
        input_dim (int): Number of input channels.
        width_scale (int): Width scale factor for the residual stages.
        dims (Tuple[int, ...]): Tuple of integers specifying the dimensions of each residual stage.
        num_blocks (int): Number of residual blocks in each stage.
        norm_type (str): Type of normalization used in residual blocks.
        fixup_init (bool): If True, the weights of the second convolution in each residual block are initialized to
            zero.
        dropout (float): Dropout probability in residual blocks.
        activation (str): Activation function used in residual blocks.
        kernel_size (Union[int, Tuple[int, ...]]): Size of the convolutional kernel.
        use_max_pooling (bool): If True, max pooling is applied after the first convolution in each residual stage.


    Examples:
        >>> resnet = ImpalaResNet(input_dim=3, width_scale=2)
        >>> input_tensor = torch.randn((16, 3, 64, 64))
        >>> output_tensor = resnet(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([16, 64, 8, 8])
    """

    def __init__(self,
                 input_dim: int,
                 width_scale: int = 1,
                 dims: Sequence[int] = IMPALA_DIMS,
                 kernel_size: Tuple[int, int] = (3, 3),
                 use_max_pooling: bool = True,
                 num_blocks: int = 2,
                 norm_type: str = NONE_NORM,
                 fixup_init: bool = False,
                 dropout: float = 0.0,
                 activation: str = "relu"):
        super().__init__()

        # Store instance variables
        self._input_dim: int = input_dim
        self._width_scale: int = width_scale
        self._dims: Sequence[int] = dims
        self._activation: str = activation

        # Resolve activation function
        self._activation_fn: torch.nn.Module = resolve_activation(activation)

        # Initialize residual stages
        self._residual_stages = torch.nn.ModuleList()

        stage_input_dim = input_dim
        for width in self.dims:
            stage = ImpalaResidualStage(
                input_dim=stage_input_dim,
                dim=int(width * width_scale),
                kernel_size=kernel_size,
                num_blocks=num_blocks,
                norm_type=norm_type,
                use_max_pooling=use_max_pooling,
                dropout=dropout,
                fixup_init=fixup_init,
                activation_fn=self._activation_fn
            )
            self._residual_stages.append(stage)
            stage_input_dim = int(width * width_scale)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def width_scale(self) -> int:
        return self._width_scale

    @property
    def dims(self) -> Sequence[int]:
        return self._dims

    @property
    def num_blocks(self) -> int:
        return self._residual_stages[0].num_blocks

    @property
    def norm_type(self) -> str:
        return self._residual_stages[0].norm_type

    @property
    def fixup_init(self) -> bool:
        return self._residual_stages[0].fixup_init

    @property
    def dropout(self) -> float:
        return self._residual_stages[0].dropout

    @property
    def activation(self) -> str:
        return self._activation

    @property
    def kernel_size(self) -> Union[int, Tuple[int, ...]]:
        return self._residual_stages[0].kernel_size

    @property
    def use_max_pooling(self) -> bool:
        return self._residual_stages[0].use_max_pooling

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
         Performs a forward pass through the ImpalaResNet applying each residual stage.

        Example:
            >>> resnet = ImpalaResNet(input_dim=3, width_scale=2)
            >>> input_tensor = torch.randn((16, 3, 64, 64))
            >>> output_tensor = resnet(input_tensor)
            >>> print(output_tensor.shape)
            torch.Size([16, 64, 8, 8])

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_channels, output_height, output_width).
        """
        out = x
        for res_stage in self._residual_stages:
            out = res_stage(out)
            out = self._activation_fn(out)

        return out

    def serialize_constructor_params(self, *args, **kwargs) -> dict:
        return {
            "input_dim": self.input_dim,
            "width_scale": self.width_scale,
            "dims": self.dims,
            "kernel_size": self.kernel_size,
            "use_max_pooling": self.use_max_pooling,
            "num_blocks": self.num_blocks,
            "norm_type": self.norm_type,
            "fixup_init": self.fixup_init,
            "dropout": self.dropout,
            "activation": self.activation
        }


class ResNetBlock(torch.nn.Module):
    """
    ResNet block.

    Args:
        input_dim (int): Number of input channels.
        dim (int): Number of output channels.
        kernel_size (Union[int, Tuple[int, ...]], optional): Size of the convolutional kernel. Defaults to (3, 3).
        stride (Union[int, Tuple[int, ...]], optional): Stride of the convolutional operation. Defaults to (1, 1).
        norm_type (str, optional): Type of normalization to use in the block.
            Choose from GROUP_NORM, BATCH_NORM, NONE_NORM. Defaults to GROUP_NORM.
        activation_fn (torch.nn.Module, optional): Activation function used in the block.
            Defaults to torch.nn.ReLU().

    Attributes:
        input_dim (int): Number of input channels.
        dim (int): Number of output channels.
        norm_type (str): Type of normalization used in the block.
        activation_fn (torch.nn.Module): Activation function used in the block.

    Methods:
        forward(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            Forward pass through the ResNetBlock.

    Examples:
        >>> resnet_block = ResNetBlock(input_dim=64, dim=128)
        >>> input_tensor = torch.randn((16, 64, 32, 32))
        >>> output_tensor = resnet_block(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([16, 128, 32, 32])
    """

    def __init__(self,
                 input_dim: int,
                 dim: int,
                 kernel_size: Union[int, Tuple[int, ...]] = (3, 3),
                 stride: Union[int, Tuple[int, ...]] = (1, 1),
                 norm_type: str = GROUP_NORM,
                 activation_fn: torch.nn.Module = torch.nn.ReLU()):

        super().__init__()

        # Store activation function
        self._activation_fn = activation_fn
        self._norm_type: str = norm_type

        # Initialize convolutions
        self._conv0 = Conv2dSame(
            in_channels=input_dim,
            out_channels=dim,
            kernel_size=kernel_size,
            stride=stride
        )
        self._conv1 = Conv2dSame(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size
        )

        # Initialize normalizations
        if norm_type == GROUP_NORM:
            self._norm0 = torch.nn.GroupNorm(num_channels=dim, num_groups=max(1, int(dim // 8)), eps=1e-5)
            self._norm1 = torch.nn.GroupNorm(num_channels=dim, num_groups=max(1, int(dim // 8)), eps=1e-5)

        elif norm_type == BATCH_NORM:
            self._norm0 = torch.nn.BatchNorm2d(num_features=dim, eps=1e-5)
            self._norm1 = torch.nn.BatchNorm2d(num_features=dim, eps=1e-5)

        elif self.norm_type == NONE_NORM:
            self._norm0 = lambda x: x
            self._norm1 = self._norm0

        # Initialize projection if needed
        if isinstance(stride, tuple):
            reduce_dim_with_strides = stride[0] > 1 or stride[1] > 1
        else:
            reduce_dim_with_strides = stride > 1
        if input_dim != dim or reduce_dim_with_strides:
            self._residual_projection = torch.nn.Conv2d(
                in_channels=input_dim,
                out_channels=dim,
                kernel_size=(1, 1),
                stride=stride
            )
            if norm_type == GROUP_NORM:
                self._norm_proj = torch.nn.GroupNorm(num_channels=dim, num_groups=max(1, int(dim // 8)), eps=1e-5)

            elif norm_type == BATCH_NORM:
                self._norm_proj = torch.nn.BatchNorm2d(num_features=dim, eps=1e-5)

            elif self.norm_type == NONE_NORM:
                self._norm_proj = lambda x: x

    @property
    def input_dim(self) -> int:
        return self._conv0.in_channels

    @property
    def dim(self) -> int:
        return self._conv0.out_channels

    @property
    def norm_type(self) -> str:
        return self._norm_type

    @property
    def activation_fn(self) -> torch.nn.Module:
        return self._activation_fn

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass through the ResNetBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        residual = x
        out = self._conv0(x)
        out = self._norm0(out)
        out = self._activation_fn(out)
        out = self._conv1(out)
        out = self._norm1(out)

        if residual.shape != out.shape:
            residual = self._residual_projection(residual)
            residual = self._norm_proj(residual)

        return self._activation_fn(residual + out)


class ResNetEncoder(SerializableModule):
    """
    ResNet encoder defaulting to ResNet 18.

    Args:
        input_dim (int): Number of input channels.
        stage_sizes (Sequence[int], optional): Number of residual blocks in each stage. Defaults to (2, 2, 2, 2).
        kernel_size (Union[int, Tuple[int, ...]], optional): Size of the convolutional kernel. Defaults to (3, 3).
        num_filters (int, optional): Number of filters in the first convolutional layer. Defaults to 64.
        first_conv_kernel_size (Union[int, Tuple[int, ...]], optional): Size of the kernel in the first convolutional
            layer. Defaults to (7, 7).
        first_conv_padding (Union[str, int, Tuple[int, ...]], optional): Padding in the first convolutional layer.
            Defaults to (3, 3).
        norm_type (str, optional): Type of normalization to use in the ResNet blocks. Choose from GROUP_NORM,
            BATCH_NORM, NONE_NORM. Defaults to GROUP_NORM.
        width_scale (float, optional): Width scale factor for the number of filters. Defaults to 1.0.
        activation (str, optional): Activation function to use. Defaults to "relu".
        strides (Sequence[int], optional): Strides for the max-pooling operation. Defaults to (2, 2, 2, 1, 1).

    Attributes:
        input_dim (int): Number of input channels.
        stage_sizes (Sequence[int]): Number of residual blocks in each stage.
        kernel_size (Union[int, Tuple[int, ...]]): Size of the convolutional kernel.
        num_filters (int): Number of filters in the first convolutional layer.
        first_conv_kernel_size (Union[int, Tuple[int, ...]]): Size of the kernel in the first convolutional layer.
        first_conv_padding (Union[str, int, Tuple[int, ...]]): Padding in the first convolutional layer.
        norm_type (str): Type of normalization used in the ResNet blocks.
        width_scale (float): Width scale factor for the number of filters.
        activation (str): Activation function used in the ResNet blocks.
        activation_fn (torch.nn.Module): Activation function module.
        strides (Sequence[int]): Strides for the max-pooling operation.

    Examples:
        >>> resnet_encoder = ResNetEncoder(input_dim=3)
        >>> input_tensor = torch.randn((16, 3, 64, 64))
        >>> output_tensor = resnet_encoder(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([16, 512, 4, 4])
    """

    def __init__(self,
                 input_dim: int,
                 stage_sizes: Sequence[int] = (2, 2, 2, 2),
                 kernel_size: Union[int, Tuple[int, ...]] = (3, 3),
                 num_filters: int = 64,
                 first_conv_kernel_size: Union[int, Tuple[int, ...]] = (7, 7),
                 first_conv_padding: Union[str, int, Tuple[int, ...]] = (3, 3),
                 norm_type: str = GROUP_NORM,
                 width_scale: float = 1.0,
                 activation: str = "relu",
                 strides: Sequence[int] = (2, 2, 2, 1, 1)):
        super().__init__()

        # Store instance variables
        self._stage_sizes: Sequence[int] = stage_sizes
        self._num_filters: int = num_filters
        self._width_scale: float = width_scale
        self._activation: str = activation
        self._strides: Sequence[int] = strides
        self._activation_fn = resolve_activation(activation)

        # Initialize first 7x7 convolution
        self._first_conv = torch.nn.Conv2d(
            in_channels=input_dim,
            out_channels=int(num_filters * width_scale),
            kernel_size=first_conv_kernel_size,
            stride=(strides[0], strides[0]),
            padding=first_conv_padding
        )
        if norm_type == GROUP_NORM:
            self._first_norm = torch.nn.GroupNorm(
                num_channels=int(num_filters * width_scale),
                num_groups=max(1, int(int(num_filters * width_scale) // 8)),
                eps=1e-5
            )

        elif norm_type == BATCH_NORM:
            self._first_norm = torch.nn.BatchNorm2d(num_features=int(num_filters * width_scale), eps=1e-5)

        elif self.norm_type == NONE_NORM:
            self._first_norm = lambda x: x

        else:
            raise ValueError(f"Unsupported norm type {norm_type}.")

        # Initialize max-pooling
        self._max_pool = MaxPool2dSamePadding(
            kernel_size=kernel_size,
            stride=(strides[1], strides[1])
        )

        # Initialize residual blocks
        self._res_blocks = torch.nn.ModuleList()

        # For each stage
        block_input_dim = int(num_filters * width_scale)  # initial stage input dim
        for i, block_size in enumerate(stage_sizes):

            # Compute block output dim for the current stage
            block_output_dim = int(num_filters * 2 ** i * width_scale)

            # For each block in the stage
            for j in range(block_size):
                # Choose the right stride
                stride = (self.strides[i + 1], self.strides[i + 1]) if i > 0 and j == 0 else (1, 1)

                # Create the block
                block = ResNetBlock(
                    input_dim=block_input_dim if j == 0 else block_output_dim,
                    dim=block_output_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    norm_type=norm_type,
                    activation_fn=self.activation_fn
                )
                self._res_blocks.append(block)

            # Update block input dim for the next stage
            block_input_dim = block_output_dim

    @property
    def input_dim(self) -> int:
        return self._first_conv.in_channels

    @property
    def stage_sizes(self) -> Sequence[int]:
        return self._stage_sizes

    @property
    def kernel_size(self) -> Union[int, Tuple[int, ...]]:
        return self._res_blocks[0].kernel_size

    @property
    def num_filters(self) -> int:
        return self._num_filters

    @property
    def first_conv_kernel_size(self) -> Union[int, Tuple[int, ...]]:
        return self._first_conv.kernel_size

    @property
    def first_conv_padding(self) -> Union[str, int, Tuple[int, ...]]:
        return self._first_conv.padding

    @property
    def norm_type(self) -> str:
        return self._res_blocks[0].norm_type

    @property
    def width_scale(self) -> float:
        return self._width_scale

    @property
    def activation(self) -> str:
        return self._activation

    @property
    def activation_fn(self) -> torch.nn.Module:
        return self._activation_fn

    @property
    def strides(self) -> Sequence[int]:
        return self._strides

    @property
    def res_blocks(self) -> torch.nn.ModuleList:
        return self._res_blocks

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass through the ResNetEncoder, pplying the following operations:
            1. Convolutional layer with a specified kernel size and padding.
            2. Normalization using GroupNorm or BatchNorm based on the chosen normalization type.
            3. Activation function specified during initialization.
            4. Max-pooling layer with a specified kernel size and stride.
            5. A series of residual blocks, each consisting of two convolutional layers, normalization, and an
                activation function.

        Example:
            >>> resnet_encoder = ResNetEncoder(input_dim=3)
            >>> input_tensor = torch.randn((16, 3, 64, 64))
            >>> output_tensor = resnet_encoder(input_tensor)
            >>> print(output_tensor.shape)
            torch.Size([16, 512, 4, 4])

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_channels, output_height, output_width).
        """
        out = x

        # Apply the first convolutional block
        out = self._first_conv(out)
        out = self._first_norm(out)
        out = self._activation_fn(out)

        # Apply max-pooling
        out = self._max_pool(out)

        # Apply residual blocks
        for block in self._res_blocks:
            out = block(out)

        return out

    def serialize_constructor_params(self, *args, **kwargs) -> dict:
        return {
            "input_dim": self.input_dim,
            "stage_sizes": self.stage_sizes,
            "kernel_size": self.kernel_size,
            "num_filters": self.num_filters,
            "first_conv_kernel_size": self.first_conv_kernel_size,
            "first_conv_padding": self.first_conv_padding,
            "norm_type": self.norm_type,
            "width_scale": self.width_scale,
            "activation": self.activation,
            "strides": self.strides
        }
