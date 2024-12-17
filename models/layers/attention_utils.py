import math
from typing import Optional, Tuple, Union, Final
from functools import partial
import torch
import einops
from einops.layers.torch import Rearrange
from pytorch_symbolic.symbolic_data import SymbolicTensor
from pytorch_symbolic.useful_layers import LambdaOpLayer


EPSILON: Final[float] = 1e-12


def find_n_heads(embed_dim: int, max_n_heads: int) -> int:
    """
    Finds the maximum number of heads given the embedding dimension.

    :param embed_dim: the embedding dimension
    :type embed_dim: int
    :param max_n_heads: the maximum number of attention heads
    :type max_n_heads: int

    :return: the greatest number n < max_n_heads such that embed_dim is divisible by n.
    :rtype: int
    """
    n_heads = max_n_heads
    if embed_dim % max_n_heads != 0:
        # Find the greatest number that embed_dim is divisible by
        n_heads = 1
        for i in range(max_n_heads, 1, -1):
            if embed_dim % i == 0:
                n_heads = i
                break
    return n_heads


def find_mha_embedding_dim(embed_dim: int, n_heads: int) -> int:
    """
    Find the minimum embedding dimension greater or equal than embed_dim that is multiple of n_heads.

    :param embed_dim: the minimum embedding dimension
    :type embed_dim: int
    :param n_heads: the number of attention heads
    :type n_heads: int

    :return: the minimum multiple of n_heads that is greater than embed_dim, computed as n_heads *
        ceil(embed_dim / n_heads).
    """
    return n_heads * math.ceil(embed_dim / n_heads)


class QueryProjectedMultiHeadAttention(torch.nn.MultiheadAttention):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 bias: bool = True,
                 add_bias_kv: bool = False,
                 add_zero_attn: Optional[int] = False,
                 qdim: Optional[int] = None,
                 kdim: Optional[int] = None,
                 vdim: Optional[int] = None,
                 batch_first: bool = False,
                 device=None,
                 dtype=None):
        r"""Allows the model to jointly attend to information
            from different representation subspaces as described in the paper:
            `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.
            This extended module simply leveregase an additional query projection to allows queries with different
            embedding dimensions other than embedding size.

            Multi-Head Attention is defined as:

            .. math::
                \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

            where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

            ``forward()`` will use a special optimized implementation if all of the following
            conditions are met:

            - self attention is being computed (i.e., ``query``, ``key``, and ``value`` are the same tensor. This
              restriction will be loosened in the future.)
            - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor argument
                ``requires_grad``
            - training is disabled (using ``.eval()``)
            - dropout is 0
            - ``add_bias_kv`` is ``False``
            - ``add_zero_attn`` is ``False``
            - ``batch_first`` is ``True`` and the input is batched
            - ``kdim`` and ``vdim`` are equal to ``embed_dim``
            - at most one of ``key_padding_mask`` or ``attn_mask`` is passed
            - if a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ is passed, neither ``key_padding_mask``
              nor ``attn_mask`` is passed

            If the optimized implementation is in use, a
            `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be passed for
            ``query``/``key``/``value`` to represent padding more efficiently than using a
            padding mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_
            will be returned, and an additional speedup proportional to the fraction of the input
            that is padding can be expected.

            Args:
                embed_dim: Total dimension of the model.
                num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
                    across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
                dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
                bias: If specified, adds bias to input / output projection layers. Default: ``True``.
                add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
                add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
                    Default: ``False``.
                qdim: Total number of features for queries. Default ``None`` (uses ``qdim=embed_dim``) as in regular
                    multi-head attention layer, otherwise a projection from qdim to embedding_dim will be added.
                kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
                vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
                batch_first: If ``True``, then the input and output tensors are provided
                    as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

            Examples::
                >>> embedding_dim = 512
                >>> n_heads = 8
                >>> query, key, value = torch.randn(89, 32, 256), torch.randn(60, 32, 256), torch.randn(89, 32, 256)
                >>> # Here regular PyTorch MHA would throw error because qdim != embedding_dim
                >>> multihead_attn = QueryProjectedMultiHeadAttention(embedding_dim, n_heads, qdim=256)
                >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
        """
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            device=device,
            dtype=dtype
        )
        self._qdim: int = qdim
        self._query_projection = None
        if qdim != embed_dim:
            self._query_projection = torch.nn.Linear(
                in_features=qdim,
                out_features=embed_dim,
                bias=add_bias_kv,
                device=device,
                dtype=dtype
            )

    @property
    def qdim(self) -> int:
        return self._qdim

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = True,
                attn_mask: Optional[torch.Tensor] = None,
                average_attn_weights: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self._query_projection is not None:
            query = self._query_projection(query)

        return super().forward(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights
        )

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(' \
                f'embed_dim={self.embed_dim}, ' \
                f'num_heads={self.num_heads}, ' \
                f'dropout={self.dropout}, ' \
                f'bias={self.in_proj_bias is not None}, ' \
                f'add_bias_kv={self.bias_k is not None}, ' \
                f'add_zero_attn={self.add_zero_attn}, ' \
                f'qdim={self.qdim}, ' \
                f'kdim={self.kdim}, ' \
                f'vdim={self.vdim}, ' \
                f'batch_first={self.batch_first})'


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, dropout: float = 0.0, default_mask: Optional[torch.Tensor] = None):
        """
        Layer implementing a scaled dot-product attention mechanism using einops' einsum function. It takes query, key,
        and value tensors as input and produces an attention-weighted output tensor. Attention scores are computed by
        scaling the dot product of query and key vectors.

        Args:
            dropout (float): Dropout probability applied to the output.
            default_mask (Optional[torch.Tensor]): Default value of the attention mask.

        Attributes:
            `softmax` (torch.nn.Softmax): Softmax activation function to compute attention scores.
            `dropout_rate` (float): Dropout rate applied to the output.
            `dropout` (torch.nn.Dropout): Dropout module to apply dropout regularization.
        """
        super(ScaledDotProductAttention, self).__init__()
        self._softmax = torch.nn.Softmax(dim=-1)
        self._dropout_rate: float = dropout
        self._default_mask: Optional[torch.Tensor] = default_mask
        self._dropout = torch.nn.Dropout(dropout)

    @property
    def dropout_rate(self) -> float:
        return self._dropout_rate

    @property
    def default_mask(self) -> Optional[torch.Tensor]:
        return self._default_mask

    @default_mask.setter
    def default_mask(self, default_mask: Optional[torch.Tensor]):
        self._default_mask = default_mask

    @property
    def softmax(self) -> torch.nn.Softmax:
        return self._softmax

    @property
    def dropout(self) -> torch.nn.Dropout:
        return self._dropout

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass of the ScaleDotProductAttention layer.

        Args:
            query (torch.Tensor): Query tensor of shape (batch, length, dim) or (batch, head, q_length, dim).
            key (torch.Tensor): Key tensor of shape (batch, length, dim) or (batch, head, k_length, dim).
            value (torch.Tensor): Value tensor of shape (batch, length, dim) or (batch, head, k_length, dim).
            mask (Optional[torch.Tensor]): Mask tensor of shape (batch, q_length, k_length) or
                (batch, n_heads, q_length, k_length).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor of shape (batch, q_length, dim) or
            (batch, head, q_length, dim) and attention tensor of the same shape containing attention scores.
        """

        # Compute the product of query and key tensors
        if len(query.size()) == 4:
            batch, head, length, dim = key.size()
            product = einops.einsum(query, key, 'b h s_q e, b h s_k e -> b h s_q s_k')
        else:
            batch, length, dim = key.size()
            product = einops.einsum(query, key, 'b s_q e, b s_k e -> b s_q s_k')

        # Scale the dot-product tensor
        scale_product = product * dim ** -0.5

        # Mask the dot-product tensor according to the mask (apply default mask if None is given)
        mask = mask if mask is not None else self._default_mask
        if mask is not None:
            assert mask.size() == scale_product.size(), f"The size of the mask must match the size of the product. " \
                                                        f"{mask.size()} and {scale_product.size()} given."
            scale_product = scale_product.masked_fill(mask == 0, -float("Inf"))

        # Compute the attention scores through softmax
        attention = self._softmax(scale_product)

        # Apply attention to value tensor
        if len(query.size()) == 4:
            output = einops.einsum(attention, value, 'b h i j, b h j d -> b h i d')
        else:
            output = einops.einsum(attention, value, 'b i j, b j d -> b i d')

        # Optionally apply dropout
        if self.dropout_rate > 0:
            output = self._dropout(output)

        return output, attention


class CustomMultiHeadAttention(torch.nn.Module):
    def __init__(self,
                 embed_dim: int,
                 n_heads: int = 8,
                 head_dim_key: Optional[int] = None,
                 head_dim_value: Optional[int] = None,
                 qdim: Optional[int] = None,
                 kdim: Optional[int] = None,
                 vdim: Optional[int] = None,
                 out_dim: Optional[int] = None,
                 use_bias: bool = True,
                 batch_first: bool = False,
                 dropout: float = 0.0,
                 use_with_symbolic_api: bool = False):
        """
        Implements a multi-head attention mechanism using einsum. It allows for customizable dimensions of the query,
        key, and value vectors, as well as the number of heads and the dimension of the key/value head and output.

        Args:
            embed_dim (int): The embedding dimension of the input.
            n_heads (int): The number of attention heads (by default 8).
            head_dim_key (Optional[int]): The dimension of each attention head for keys (if not specified, derived
                from `embed_dim` dividing by `n_heads`).
            head_dim_value (Optional[int]): The dimension of each attention head for values (if not specified, same
                as `head_dim_key`).
            qdim (Optional[int]): Dimension of the query vectors (if not specified, same as `embed_dim`).
            kdim (Optional[int]): Dimension of the key vectors (if not specified, same as `embed_dim`).
            vdim (Optional[int]): Dimension of the value vectors (if not specified, same as `embed_dim`).
            out_dim (Optional[int]): Dimension of the output (if not specified, same as `embed_dim`).
            use_bias (bool): Whether to include bias in linear layers (`True` by default).
            batch_first (bool): If `True`, input and output tensors have shape (batch_size, seq_length, dim),
                otherwise (seq_length, batch_size, dim). By default, this is `False`.
            dropout (float): Dropout probability (default is 0.0).
            use_with_symbolic_api (bool): Whether to ensure compatibility with the symbolic API.

        Attributes:
            `attention` (ScaledDotProductAttention): The scale dot product attention instance.
            `w_q` (torch.nn.Linear): Linear transformation for query vectors.
            `w_k` (torch.nn.Linear): Linear transformation for key vectors.
            `w_v` (torch.nn.Linear): Linear transformation for value vectors.
            `w_out` (torch.nn.Linear): Linear transformation for output.
            `n_heads` (int): Number of attention heads.
            `embed_dim` (int): Embedding dimension.
            `head_dim_key` (int): Dimension of each attention head for keys.
            `head_dim_value` (int): Dimension of each attention head for values.
            `qdim` (int): Dimension of query vectors.
            `kdim` (int): Dimension of key vectors.
            `vdim` (int): Dimension of value vectors.
            `out_dim` (int): Dimension of output vectors.
            `use_bias` (bool): Whether bias is used in linear layers.
            `batch_first` (bool): Whether input and output tensors have shape (batch, length, dim).
            `dropout` (float): Dropout probability.
            `use_with_symbolic_api` (bool): Whether to ensure compatibility with the symbolic API.
        """

        super(CustomMultiHeadAttention, self).__init__()

        # If `head_dim_key` is not specified but `head_dim_value` is, use it for both key and value dimensions
        if head_dim_key is None and head_dim_value is not None:
            head_dim_key = head_dim_value

        # If `head_dim_key` and `head_dim_value` are not specified, head dimension is obtained dividing `embed_dim`
        if head_dim_key is None and embed_dim % n_heads != 0:
            raise ValueError(f"`embed_dim` must be divisible by `n_head` when `head_dim_key` and "
                             f"`head_dim_value` are not specified. {embed_dim} and {n_heads} given.")

        # Initialize attributes
        self._n_heads: int = n_heads
        self._embed_dim: int = embed_dim
        self._head_dim_key: Optional[int] = head_dim_key
        self._head_dim_value: Optional[int] = head_dim_value
        self._qdim: Optional[int] = qdim
        self._kdim: Optional[int] = kdim
        self._vdim: Optional[int] = vdim
        self._out_dim: Optional[int] = out_dim
        self._use_bias: bool = use_bias
        self._batch_first: bool = batch_first
        self._dropout: float = dropout
        self._use_with_symbolic_api = use_with_symbolic_api

        # Initialize layers
        self._to_batch_second = Rearrange("b s e -> s b e")
        self._to_batch_first = Rearrange("s b e -> b s e")
        self._attention = ScaledDotProductAttention(dropout=dropout)
        self._w_q = torch.nn.Linear(self.qdim, self.n_heads * self.head_dim_key, bias=use_bias)
        self._w_k = torch.nn.Linear(self.kdim, self.n_heads * self.head_dim_key, bias=use_bias)
        self._w_v = torch.nn.Linear(self.vdim, self.n_heads * self.head_dim_value, bias=use_bias)
        self._w_out = torch.nn.Linear(self.n_heads * self.head_dim_value, self.out_dim, bias=use_bias)

    @property
    def attention(self) -> ScaledDotProductAttention:
        return self._attention

    @property
    def w_q(self) -> torch.nn.Linear:
        return self._w_q

    @property
    def w_k(self) -> torch.nn.Linear:
        return self._w_k

    @property
    def w_v(self) -> torch.nn.Linear:
        return self._w_v

    @property
    def w_out(self) -> torch.nn.Linear:
        return self._w_out

    @property
    def to_batch_first(self) -> Rearrange:
        return self._to_batch_first

    @property
    def to_batch_second(self) -> Rearrange:
        return self._to_batch_second

    @property
    def n_heads(self) -> int:
        return self._n_heads

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def head_dim_key(self) -> Optional[int]:
        return self._head_dim_key if self._head_dim_key is not None else int(self.embed_dim / self.n_heads)

    @property
    def head_dim_value(self) -> int:
        return self._head_dim_value if self._head_dim_value is not None else self.head_dim_key

    @property
    def qdim(self) -> int:
        return self._qdim if self._qdim is not None else self.embed_dim

    @property
    def kdim(self) -> int:
        return self._kdim if self._kdim is not None else self.embed_dim

    @property
    def vdim(self) -> int:
        return self._vdim if self._vdim is not None else self.embed_dim

    @property
    def out_dim(self) -> int:
        return self._out_dim if self._out_dim else self.embed_dim

    @property
    def use_bias(self) -> bool:
        return self._use_bias

    @property
    def batch_first(self) -> bool:
        return self._batch_first

    @batch_first.setter
    def batch_first(self, batch_first: bool):
        self._batch_first = batch_first

    @property
    def dropout(self) -> float:
        return self._dropout

    @property
    def use_with_symbolic_api(self) -> bool:
        return self._use_with_symbolic_api

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_attn_weights: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the custom multi-head attention layer.

        Args:
            query (torch.Tensor): Query tensor of shape (batch, q_length, qdim).
            key (torch.Tensor): Key tensor of shape (batch, k_length, kdim).
            value (torch.Tensor): Value tensor of shape (batch, k_length, vdim).
            mask (Optional[torch.Tensor]): Mask tensor of shape (batch, q_length, k_length) or
                (batch, n_heads, q_length, k_length).
            return_attn_weights (bool): If True, also return attention weights.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: Output tensor of shape (batch, q_length, out_dim)
            and optionally attention weights tensor of shape (batch, n_heads, q_length, k_length).
        """

        # If mask is provided and not 4D, convert it to 4D
        if mask is not None and len(mask.shape) < 4:

            if len(mask.shape) <= 1:
                raise ValueError("Mask must be at least 2D.")

            # 2D to 3D mask
            if len(mask.shape) == 2:
                # Unsqueeze the mask to shape (1, q_length, k_length)
                mask = mask.unsqueeze(dim=0)

                # Repeat the mask tensor along batch dimension (0), obtaining shape (batch, q_length, k_length)
                batch_size = query.shape[0] if self.batch_first else query.shape[1]
                mask = mask.expand(batch_size, -1, -1)

            # 3D to 4D mask
            if len(mask.shape) == 3:
                # Unsqueeze the mask to shape (batch, 1, q_length, k_length)
                mask = mask.unsqueeze(dim=1)

                # Repeat the mask tensor along head dimension (1), obtaining shape (batch, n_heads, q_length, k_length)
                mask = mask.expand(-1, self.n_heads, -1, -1)

        # If not using batch-first mode, rearrange input tensors
        if not self.batch_first:
            '''pattern = "s b e -> b s e"
            query = einops.rearrange(query, pattern)
            key = einops.rearrange(key, pattern)
            value = einops.rearrange(value, pattern)'''
            query = self._to_batch_first(query)
            key = self._to_batch_first(key)
            value = self._to_batch_first(value)

        # Apply linear transformations to query, key, and value tensors
        query, key, value = self._w_q(query), self._w_k(key), self._w_v(value)

        # Split tensors into multiple heads
        query, key, value = self.split(query), self.split(key), self.split(value)

        # Apply attention mechanism and obtain output
        if self._use_with_symbolic_api and isinstance(query, SymbolicTensor):
            # Temporarily set the default mask to the given one ensures compatibility with the symbolic API, since we
            # cannot explicitly pass the attention mask or the symbolic API will throw an error
            self._attention.default_mask = mask

            # Call the attention layer without passing the mask
            out, attention = self._attention(query, key, value)

            # Set the default mask back to None
            self._attention.default_mask = None
        else:
            out, attention = self._attention(query, key, value, mask=mask)

        # Concatenate heads and apply linear transformation to obtain final output
        out = self.concat(out)
        out = self._w_out(out)

        # If required, convert to to batch-second
        if not self.batch_first:
            # out = einops.rearrange(out, "b s e -> s b e")
            out = self._to_batch_second(out)

        # If requested, return attention weights along with output
        if return_attn_weights:
            return out, attention

        return out

    def split(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Split tensor into multiple heads.

        Args:
            tensor (torch.Tensor): Input tensor of shape (batch, length, d_input).

        Returns:
            torch.Tensor: Output tensor with shape (batch, n_heads, length, d_head).
        """
        batch, length, d_input = tensor.shape
        assert d_input % self.n_heads == 0, f"Input dimension {d_input} is not divisible by the heads {self.n_heads}"
        d_head = d_input // self.n_heads

        # print(d_tensor, d_input, self.n_head)
        # tensor = tensor.view(batch, self.n_heads, length, d_head)
        if self._use_with_symbolic_api:
            tensor = LambdaOpLayer(
                partial(einops.rearrange, pattern="b s (h e) -> b h s e", h=self.n_heads, e=d_head)
            )(tensor)
        else:
            tensor = einops.rearrange(tensor, "b s (h e) -> b h s e", h=self.n_heads, e=d_head)
        return tensor

    def concat(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Concatenate multiple heads into one tensor.

        Args:
            tensor (torch.Tensor): Input tensor of shape (batch, n_heads, length, d_head).

        Returns:
            torch.Tensor: Output tensor with shape (batch, length, n_heads*d_head).
        """

        # batch, n_heads, length, d_head = tensor.size()
        # d_model = n_heads * d_head
        # tensor = tensor.view(batch, length, d_model)
        if self._use_with_symbolic_api:
            tensor = LambdaOpLayer(partial(einops.rearrange, pattern="b h s e -> b s (h e)"))(tensor)
        else:
            tensor = einops.rearrange(tensor, "b h s e -> b s (h e)")
        return tensor
