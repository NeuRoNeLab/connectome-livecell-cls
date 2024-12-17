from typing import Optional, Final
import torch
from torch.nn import LayerNorm, functional as nnf
from torch_geometric.nn.dense import Linear
from models.layers import SerializableModule
from models.layers.misc import resolve_activation, _ACTIVATIONS


class FFN(torch.nn.Module):
    def __init__(self,
                 embed_dim: int,
                 ff_dim: Optional[int] = None,
                 ff_activation: str = 'gelu',
                 gated: Optional[str] = None,
                 inner_dropout: float = 0.0):
        """
        Feed-Forward Network (FFN) module with optional gating mechanism and configurable activation function. The
        network consists of two linear layers with an activation function applied between them. If gating is enabled, an
        additional linear layer is used to compute the gate values, and the output is either added to or multiplied by
        the gate values.

        Args:
            embed_dim (int): The input embedding dimension.
            ff_dim (Optional[int]): The hidden layer dimension. If not provided, it defaults to 4 times the `embed_dim`.
            ff_activation (str): The activation function to use in the feed-forward network.
            gated (Optional[str]): The type of gating to apply ('add' for addition, any other value for multiplication).
            inner_dropout (Optional[float]): The dropout rate to use for the inner layer of feed-forward network.
        """
        super().__init__()
        ff_dim: int = ff_dim if ff_dim is not None else embed_dim * 4
        self._ff_activation: str = ff_activation
        self._ff_activation_fn: torch.nn.Module = resolve_activation(ff_activation)
        self._gated: Optional[str] = gated
        self._inner_dropout: float = inner_dropout

        # Initialize feed-forward network
        ff0 = torch.nn.Linear(embed_dim, ff_dim)
        ff1 = torch.nn.Linear(ff_dim, embed_dim)
        inner_dropout = torch.nn.Dropout(inner_dropout)
        if gated is not None:
            ff_gate = torch.nn.Linear(embed_dim, ff_dim)
            self._ff = torch.nn.ModuleDict({
                "ff0": ff0,
                "ff_gate": ff_gate,
                "inner_dropout": inner_dropout,
                "ff1": ff1
            })
        else:
            self._ff = torch.nn.Sequential(
                ff0,
                self._ff_activation_fn,
                inner_dropout,
                ff1
            )

    @property
    def embed_dim(self) -> int:
        if isinstance(self.ff, torch.nn.ModuleDict):
            return self.ff["ff0"].in_features
        return self.ff[0].in_features

    @property
    def ff_dim(self) -> int:
        if isinstance(self.ff, torch.nn.ModuleDict):
            return self.ff["ff0"].out_features
        return self.ff[0].out_features

    @property
    def gated(self) -> Optional[str]:
        return self._gated

    @property
    def ff_activation(self) -> str:
        return self._ff_activation

    @property
    def ff_activation_fn(self) -> torch.nn.Module:
        return self._ff_activation_fn

    @property
    def ff(self) -> torch.nn.Module:
        return self._ff

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass of the FFN module.

        Args:
            x (torch.Tensor): The input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The output tensor after applying the feed-forward network.
        """
        if self.gated:
            out = self.ff["ff0"](x)  # up projection
            out = self.ff_activation_fn(out)
            gate_out = self.ff["ff_gate"](x)  # gate ff
            out = out + gate_out if self.gated == "add" else out * gate_out  # gating
            out = self.ff["inner_dropout"](out)  # dropout
            out = self.ff["ff1"](out)  # down-projection
            return out
        return self._ff(x)


class PositionWiseFeedForward(SerializableModule):
    """
    Position-wise feed-forward neural network module processing multiple embeddings in parallel.
    """

    ACTIVATIONS_: Final[frozenset[str]] = frozenset(_ACTIVATIONS.keys())

    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 dropout: float = 0.0,
                 activation: str = "gelu",
                 is_gated: bool = False,
                 bias1: bool = True,
                 bias2: bool = True,
                 bias_gate: bool = True,
                 pre_norm:  bool = False):
        """
        Position-wise feed-forward neural network module processing multiple embeddings in parallel.

        * `d_model` is the number of features of an embedding
        * `d_ff` is the number of features in the hidden layer of the FFN
        * `dropout` is dropout probability for the hidden layer
        * `activation` is the activation function for the hidden layer
        * `is_gated` specifies whether the hidden layer is gated
        * `bias1` specified whether the first fully connected layer should have a learnable bias
        * `bias2` specified whether the second fully connected layer should have a learnable bias
        * `bias_gate` specified whether the fully connected layer for the gate should have a learnable bias
        * `pre_norm` whether the layer normalization should be applied before the skip-connection (pre-norm) or after it
        """
        super().__init__()

        # Layer one parameterized by weight $W_1$ and bias $b_1$
        self.dense0: Linear = Linear(d_model, d_ff, bias=bias1)

        # Layer two parameterized by weight $W_2$ and bias $b_2$
        self.dense1: Linear = Linear(d_ff, d_model, bias=bias2)

        # Layer normalization
        self.norm0 = LayerNorm(d_model, elementwise_affine=True)

        # Store d_model and d_ff
        self.__d_model: int = d_model
        self.__d_ff: int = d_ff

        # Store whether the layers should have biases
        self.__bias1: bool = bias1
        self.__bias2: bool = bias2
        self.__bias_gate: bool = bias_gate

        # Store whether the layer is pre-norm or post-norm
        self.__pre_norm: bool = pre_norm

        # Hidden layer dropout
        if not 0 <= dropout < 1:
            raise ValueError(f"Dropout rate must be between 0 and 1 excluded, {dropout} given.")
        self.__dropout: float = dropout

        # Activation function $f$
        if activation not in self.ACTIVATIONS_:
            raise ValueError(f"Activation function must be one of {self.ACTIVATIONS_}, {activation} given.")
        self.__activation: str = activation

        # Whether there is a gate
        self.__is_gated: bool = is_gated

        if is_gated:
            # If there is a gate the linear layer to transform inputs to be multiplied by the gate, parameterized by
            # weight $V$ and bias $c$
            self.dense_v: Linear = Linear(d_model, d_ff, bias=bias_gate)

    @property
    def is_gated(self) -> bool:
        return self.__is_gated

    @property
    def dropout(self) -> float:
        return self.__dropout

    @dropout.setter
    def dropout(self, dropout: float):
        self.__dropout = dropout

    @property
    def activation(self) -> str:
        return self.__activation

    @activation.setter
    def activation(self, activation: str):
        self.__activation = activation

    @property
    def d_model(self) -> int:
        return self.__d_model

    @property
    def d_ff(self) -> int:
        return self.__d_ff

    @property
    def bias1(self) -> bool:
        return self.__bias1

    @property
    def bias2(self) -> bool:
        return self.__bias2

    @property
    def bias_gate(self) -> bool:
        return self.__bias_gate

    @property
    def pre_norm(self) -> bool:
        return self.__pre_norm

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Forward pass of the position-wise feed-forward neural network.

        Args:
            x (torch.Tensor): Input tensor of shape (*, H_{in}), where * means any number of dimensions, including none,
                and H_{in} = in_features.
            normalize (bool, optional): Whether to apply (pre or post) layer normalization. Defaults to True.

        Returns:
            torch.Tensor: Output tensor of shape (*, H_{out}), where all but the last dimension are the same shape as
                the input, and H_{out} = out_features.
        """

        # Apply pre-normalization if required
        if normalize and self.pre_norm:
            x = self.norm0(x)

        # Store input for skip-connection
        x0 = x

        # $f(x W_1 + b_1)$
        g = _ACTIVATIONS[self.activation](self.dense0(x))

        # If gated, $f(x W_1 + b_1) \otimes (x V + b) $
        if self.is_gated:
            x = g * self.dense_v(x)
        # Otherwise
        else:
            x = g

        # Apply dropout
        x = nnf.dropout(x, p=self.dropout)

        # Apply second dense $(f(x W_1 + b_1) \otimes (x V + b)) W_2 + b_2$ or $f(x W_1 + b_1) W_2 + b_2$ (if gated)
        x = self.dense1(x)

        # Apply skip-connection $x + (f(x W_1 + b_1) \otimes (x V + b)) W_2 + b_2$ or $x + f(x W_1 + b_1) W_2 + b_2$
        x = x0 + x

        # Apply post-normalization if required
        if normalize and not self.pre_norm:
            x = self.norm0(x)

        return x

    def serialize_constructor_params(self, *args, **kwargs) -> dict:
        return {
            "d_model": self.__d_model,
            "d_ff": self.__d_ff,
            "dropout": self.__dropout,
            "activation": self.__activation,
            "is_gated": self.__is_gated,
            "bias1": self.__bias1,
            "bias2": self.__bias2,
            "bias_gate": self.__bias_gate,
            "pre_norm": self.__pre_norm
        }


