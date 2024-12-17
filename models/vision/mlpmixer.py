from typing import Optional, Tuple, NamedTuple
import torch
from einops.layers.torch import Rearrange
from pytorch_symbolic.useful_layers import ConcatLayer
from models.layers import SerializableModule
from models.moe.moe import MoE
from utils.utils import SerializableConfig
from models.layers.mlp import FFN
from models.vision.vit import ViTPatchEmbeddings


class MLPMixerOutput(NamedTuple):
    # Network output
    output: torch.Tensor

    # CLS final embedding
    cls_embedding: torch.Tensor

    # Average final embedding
    avg_embedding: torch.Tensor

    # Final embeddings
    final_embeddings: Optional[torch.Tensor] = None

    # MoE loss
    moe_loss: Optional[torch.Tensor] = None


class MLPMixerConfig(SerializableConfig):

    def __init__(self,
                 n_blocks: int,
                 embed_dim: int,
                 in_channels: int,
                 output_dim: int,
                 patch_dim: int = 16,
                 image_dim: int = 224,
                 token_ff_dim: Optional[int] = None,
                 channel_ff_dim: Optional[int] = None,
                 ff_activation: str = 'gelu',
                 ff_gated: Optional[str] = None,
                 dropout: float = 0.0,
                 pre_norm: bool = True,
                 moe_num_experts: int = 1,
                 moe_noisy_gating: bool = True,
                 moe_k: int = 2,
                 final_layer_norm: bool = True,
                 use_cls_pool: bool = False):
        """
        Configuration class for the MLPMixer model.

        Args:
            n_blocks (int): Number of Mixer blocks.
            embed_dim (int): Dimensionality of the embeddings.
            in_channels (int): Number of input channels.
            output_dim (int): Dimensionality of the output.
            patch_dim (int, optional): Dimensionality of the patches. Default is 16.
            image_dim (int, optional): Input image dimension. Default is 224 (standing for 224x224).
            token_ff_dim (Optional[int], optional): Dimensionality of the token feed-forward layer. Defaults to None.
            channel_ff_dim (Optional[int], optional): Dimensionality of the channel feed-forward layer. Defaults to
                None.
            ff_activation (str, optional): Activation function for the feed-forward layers. Default is 'gelu'.
            ff_gated (Optional[str], optional): Gating mechanism for the feed-forward layers. Defaults to None.
            dropout (float, optional): Dropout rate. Default is 0.0.
            pre_norm (bool, optional): Whether to apply layer normalization before feed-forward layers. Default is True.
            moe_num_experts (int, optional): Number of experts in Mixture of Experts. Default is 1.
            moe_noisy_gating (bool, optional): Whether to use noisy gating in MoE. Default is True.
            moe_k (int, optional): Number of experts to use in the gating mechanism of MoE. Default is 2.
            final_layer_norm (bool, optional): Whether to apply layer normalization before pooling the embeddings.
                Defaults to True.
            use_cls_pool (bool, optional): Whether to use classification pooling. Default is False.
        """
        super().__init__()
        self.n_blocks: int = n_blocks
        self.embed_dim: int = embed_dim
        self.in_channels: int = in_channels
        self.output_dim: int = output_dim
        self.patch_dim: int = patch_dim
        self.image_dim: int = image_dim
        self.token_ff_dim: int = token_ff_dim
        self.channel_ff_dim: int = channel_ff_dim
        self.ff_activation: str = ff_activation
        self.ff_gated: str = ff_gated
        self.dropout: float = dropout
        self.pre_norm: bool = pre_norm
        self.moe_num_experts: int = moe_num_experts
        self.moe_noisy_gating: bool = moe_noisy_gating
        self.moe_k: int = moe_k
        self.final_layer_norm: bool = final_layer_norm
        self.use_cls_pool: bool = use_cls_pool


class TokenMixing(FFN):

    def __init__(self,
                 num_patches: int,
                 ff_dim: Optional[int] = None,
                 ff_activation: str = 'gelu',
                 gated: Optional[str] = None,
                 dropout: float = 0.0,
                 use_with_flatten_moe: bool = False):
        # noinspection PyUnresolvedReferences
        """
        TokenMixing module for the MLPMixer model. This module applies a feed-forward network to mix tokens
        (patches) across the input sequence.

        Args:
            num_patches (int): Number of patches (tokens) in the input.
            ff_dim (Optional[int], optional): Dimensionality of the feed-forward layer. Defaults to None.
            ff_activation (str, optional): Activation function to use in the feed-forward layers. Default is
                'gelu'.
            gated (Optional[str], optional): Type of gating to use in the feed-forward layers. Default is None.
            dropout (float, optional): Dropout rate for the feed-forward layers. Default is 0.0.
            use_with_flatten_moe (bool, optional): Whether to use the module with flatten MoE. Default is False.

        Attributes:
            use_with_flatten_moe (bool): Whether the module is used with flatten MoE.
            to_token_last (Rearrange): Transformation to swap sequence and embedding dimensions.
            to_embedding_last (Rearrange): Transformation to revert the swap of sequence and embedding
                dimensions.
            flatten_batch (Optional[Rearrange]): Transformation to flatten batch and sequence dimensions.
            unflatten_batch (Optional[Rearrange]): Transformation to unflatten batch and sequence dimensions.
        """
        super().__init__(
            embed_dim=num_patches,
            ff_dim=ff_dim,
            ff_activation=ff_activation,
            gated=gated,
            inner_dropout=dropout
        )
        self._use_with_flatten_moe: bool = use_with_flatten_moe

        self._to_token_last = Rearrange('b ... s e -> b ... e s')  # the ... were not here
        self._to_embedding_last = Rearrange('b ... e s -> b ... s e')  # the ... were not here

        self._flatten_batch = None
        self._unflatten_batch = None
        if use_with_flatten_moe:
            self._unflatten_batch = Rearrange(
                '(ba s) e -> ba s e',  # ba can contain more aggregated dimensions other than batch_size
                s=num_patches
            )
            self._flatten_batch = Rearrange('ba s e -> (ba s) e')

    @property
    def use_with_flatten_batch(self) -> bool:
        return self._use_with_flatten_moe

    @property
    def to_token_last(self) -> Rearrange:
        return self._to_token_last

    @property
    def to_embedding_last(self) -> Rearrange:
        return self._to_embedding_last

    @property
    def unflatten_batch(self) -> Rearrange:
        return self._unflatten_batch

    @property
    def flatten_batch(self) -> Optional[Rearrange]:
        return self._flatten_batch

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass for the TokenMixing module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_patches, embed_dim).

        Returns:
            torch.Tensor: Output tensor after token mixing and FFN layers.
        """

        # Unflatten the batch and sequence dimensions if needed
        if self.use_with_flatten_batch:
            x = self.unflatten_batch(x)

        # From (batch_size, ..., seq_len, embed_dim) to (batch_size, ..., embed_dim, seq_len)
        x = self.to_token_last(x)

        # Apply FFN
        out = super().forward(x=x, *args, **kwargs)

        # From (batch_size, ..., embed_dim, seq_len) to (batch_size, ..., seq_len, embed_dim)
        out = self.to_embedding_last(out)

        # Flatten the batch and sequence dimension if needed
        if self.use_with_flatten_batch:
            out = self.flatten_batch(out)

        return out


class MLPMixerBlock(torch.nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_patches: int,
                 token_ff_dim: Optional[int] = None,
                 channel_ff_dim: Optional[int] = None,
                 ff_activation: str = 'gelu',
                 ff_gated: Optional[str] = None,
                 dropout: float = 0.0,
                 pre_norm: bool = True,
                 moe_num_experts: int = 1,
                 moe_noisy_gating: bool = True,
                 moe_k: int = 2):
        # noinspection PyUnresolvedReferences
        """
        MLPMixerBlock is a building block of the MLP-Mixer architecture, which leverages token-mixing and
        channel-mixing fully connected layers for image processing tasks. Includes an option for a Mixture of Experts
        (MoE) model for enhanced flexibility and performance.

        Args:
            embed_dim (int): Dimensionality of the token embeddings.
            num_patches (int): Number of patches (tokens) in the input.
            token_ff_dim (Optional[int]): Dimensionality of the token mixing fully connected layer. Defaults to None,
                which means `num_patches * 4`.
            channel_ff_dim (Optional[int]): Dimensionality of the channel mixing fully connected layer. Defaults to
                None, which means `embed_dim * 4`.
            ff_activation (str, optional): Activation function to use in the FFN layers. Default is 'gelu'.
            ff_gated (Optional[str], optional): Type of gating to use in the FFN layers. May be 'add' for additive
                gating, or 'mult' for multiplicative gating. Default is None, meaning no gating.
            dropout (float, optional): Dropout rate for the FFN layers. Default is 0.0.
            pre_norm (bool, optional): Whether to apply LayerNorm before the FFN layers. Default is True.
            moe_num_experts (int, optional): Number of experts in the MoE model. Default is 1, meaning no MoE is used.
            moe_noisy_gating (bool, optional): Whether to use noisy gating in the MoE model. Default is True.
            moe_k (int, optional): Number of experts to use in the MoE gating. Default is 2.

        Attributes:
            pre_norm (bool): Whether LayerNorm is applied before the FFN layers.
            dropout (float): Dropout rate used in the FFN layers.
            moe_num_experts (int): Number of experts in the Mixture of Experts model.
            moe_noisy_gating (bool): Whether noisy gating is used in the MoE model.
            moe_k (int): Number of experts to use in the MoE gating.
            ln0 (torch.nn.Module): LayerNorm applied before token mixing.
            ln1 (torch.nn.Module): LayerNorm applied before channel mixing.
            token_mixing (torch.nn.Module): Container module for token mixing layers or MoE model.
            channel_mixing (torch.nn.Module): Sequential container for channel mixing layers or MoE model.

        Example:
            >>> x_ = torch.rand(2, 16, 512)  # (batch_size, num_patches, embed_dim)
            >>> mixer_block_ = MLPMixerBlock(embed_dim=512, num_patches=16, token_ff_dim=256, channel_ff_dim=1024)
            >>> out_ = mixer_block_(x_)
            >>> print(out_.shape)
            >>> torch.Size([2, 16, 512])
        """

        super().__init__()

        # Store instance variables
        self._embed_dim: int = embed_dim
        self._num_patches: int = num_patches
        self._token_ff_dim: int = token_ff_dim
        self._channel_ff_dim: int = channel_ff_dim
        self._ff_activation: str = ff_activation
        self._ff_gated: str = ff_gated
        self._pre_norm: bool = pre_norm
        self._dropout: float = dropout
        self._moe_num_experts: int = moe_num_experts
        self._moe_noisy_gating: bool = moe_noisy_gating
        self._moe_k: int = moe_k

        # Initialize layer norms & dropouts
        self._ln0 = torch.nn.LayerNorm(embed_dim)
        self._ln1 = torch.nn.LayerNorm(embed_dim)
        self._dropout_layer0 = torch.nn.Dropout(dropout)
        self._dropout_layer1 = torch.nn.Dropout(dropout)

        # Initialize token mixing layer
        if moe_num_experts > 1:
            self._token_mixing = MoE(
                input_size=embed_dim,
                output_size=embed_dim,
                num_experts=moe_num_experts,
                noisy_gating=moe_noisy_gating,
                k=moe_k,
                expert_class=TokenMixing,
                expert_kwargs=dict(
                    num_patches=num_patches,
                    ff_dim=token_ff_dim,
                    ff_activation=ff_activation,
                    gated=ff_gated,
                    dropout=dropout,
                    use_with_flatten_moe=True
                )
            )
        else:
            '''self._token_mixing = torch.nn.Sequential(
                Rearrange('b ... s e -> b ... e s'),  # the ... were not here
                FFN(**token_mixing_config),
                Rearrange('b ... e s -> b ... s e')   # the ... were not here
            )'''
            self._token_mixing = TokenMixing(
                num_patches=num_patches,
                ff_dim=token_ff_dim,
                ff_activation=ff_activation,
                gated=ff_gated,
                dropout=dropout,
                use_with_flatten_moe=False
            )

        # Initialize channel mixing layer
        if moe_num_experts > 1:
            self._token_mixing = MoE(
                input_size=embed_dim,
                output_size=embed_dim,
                num_experts=moe_num_experts,
                noisy_gating=moe_noisy_gating,
                k=moe_k,
                expert_class=FFN,
                expert_kwargs=dict(
                    embed_dim=embed_dim,
                    ff_dim=channel_ff_dim,
                    ff_activation=ff_activation,
                    gated=ff_gated,
                    inner_dropout=dropout
                )
            )
        else:
            self._channel_mixing = FFN(
                embed_dim=embed_dim,
                ff_dim=channel_ff_dim,
                ff_activation=ff_activation,
                gated=ff_gated,
                inner_dropout=dropout
            )

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def num_patches(self) -> int:
        return self._num_patches

    @property
    def token_ff_dim(self) -> Optional[int]:
        return self._token_ff_dim

    @property
    def channel_ff_dim(self) -> Optional[int]:
        return self._channel_ff_dim

    @property
    def ff_activation(self) -> str:
        return self._ff_activation

    @property
    def ff_gated(self) -> Optional[str]:
        return self._ff_gated

    @property
    def pre_norm(self) -> bool:
        return self._pre_norm

    @property
    def dropout(self) -> float:
        return self._dropout

    @property
    def moe_num_experts(self) -> int:
        return self._moe_num_experts

    @property
    def moe_noisy_gating(self) -> bool:
        return self._moe_noisy_gating

    @property
    def moe_k(self) -> int:
        return self._moe_k

    @property
    def ln0(self) -> torch.nn.Module:
        return self._ln0

    @property
    def ln1(self) -> torch.nn.Module:
        return self._ln1

    @property
    def dropout_layer0(self) -> torch.nn.Dropout:
        return self._dropout_layer0

    @property
    def dropout_layer1(self) -> torch.nn.Dropout:
        return self._dropout_layer1

    @property
    def token_mixing(self) -> torch.nn.Module:
        return self._token_mixing

    @property
    def channel_mixing(self) -> torch.nn.Module:
        return self._channel_mixing

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the MLPMixerBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_patches, embed_dim).

        Returns:
            torch.Tensor: Output tensor of the same shape as the input.
            Optional[torch.Tensor]: Output tensor containing the MoE loss, if MoE is used.
        """

        out = x
        if self.pre_norm:
            out = self.ln0(out)

        # Token Mixing
        moe_loss = None
        if self.moe_num_experts > 1:
            ff_out, moe_loss = self.token_mixing(out)
        else:
            ff_out = self.token_mixing(out)
        ff_out = self.dropout_layer0(ff_out)
        out = out + ff_out

        if self.pre_norm:
            out = self.ln1(out)
        else:
            out = self.ln0(out)

        if self.moe_num_experts > 1:
            ff_out, moe_loss1 = self.channel_mixing(out)
            moe_loss += moe_loss1
        else:
            ff_out = self.channel_mixing(out)
        ff_out = self.dropout_layer1(ff_out)
        out = out + ff_out

        if not self.pre_norm:
            out = self.ln1(out)

        return out, moe_loss


class MLPMixerEncoder(torch.nn.Module):

    def __init__(self,
                 n_blocks: int,
                 embed_dim: int,
                 num_patches: int,
                 token_ff_dim: Optional[int] = None,
                 channel_ff_dim: Optional[int] = None,
                 ff_activation: str = 'gelu',
                 ff_gated: Optional[str] = None,
                 dropout: float = 0.0,
                 pre_norm: bool = True,
                 moe_num_experts: int = 1,
                 moe_noisy_gating: bool = True,
                 moe_k: int = 2):
        """
        MLPMixerEncoder is a stack of MLPMixer blocks that processes input token embeddings.

        Args:
            n_blocks (int): Number of MLPMixer blocks.
            embed_dim (int): Dimension of token embeddings.
            num_patches (int): Number of patches/tokens.
            token_ff_dim (Optional[int]): Dimension of the token mixing feed-forward network. Defaults to None.
            channel_ff_dim (Optional[int]): Dimension of the channel mixing feed-forward network. Defaults to None.
            ff_activation (str): Activation function to use in the feed-forward networks. Defaults to 'gelu'.
            ff_gated (Optional[str]): Type of gating mechanism to use, if any. Defaults to None.
            dropout (float): Dropout rate. Defaults to 0.0.
            pre_norm (bool): Whether to apply layer normalization before the feed-forward network. Defaults to True.
            moe_num_experts (int): Number of experts in the MoE layer. Defaults to 1.
            moe_noisy_gating (bool): Whether to use noisy gating in the MoE layer. Defaults to True.
            moe_k (int): Number of experts to use for each token in the MoE layer. Defaults to 2.
        """

        super().__init__()

        self._blocks = torch.nn.ModuleList()

        for _ in range(n_blocks):
            block = MLPMixerBlock(
                embed_dim=embed_dim,
                num_patches=num_patches,
                token_ff_dim=token_ff_dim,
                channel_ff_dim=channel_ff_dim,
                ff_activation=ff_activation,
                ff_gated=ff_gated,
                dropout=dropout,
                pre_norm=pre_norm,
                moe_num_experts=moe_num_experts,
                moe_noisy_gating=moe_noisy_gating,
                moe_k=moe_k
            )
            self._blocks.append(block)

    @property
    def n_block(self) -> int:
        return len(self.blocks)

    @property
    def embed_dim(self) -> int:
        return self.blocks[0].embed_dim

    @property
    def num_patches(self) -> int:
        return self.blocks[0].num_patches

    @property
    def token_ff_dim(self) -> Optional[int]:
        return self.blocks[0].token_ff_dim

    @property
    def channel_ff_dim(self) -> Optional[int]:
        return self.blocks[0].channel_ff_dim

    @property
    def ff_activation(self) -> str:
        return self.blocks[0].ff_activation

    @property
    def ff_gated(self) -> Optional[str]:
        return self.blocks[0].ff_gated

    @property
    def dropout(self) -> float:
        return self.blocks[0].dropout

    @property
    def pre_norm(self) -> bool:
        return self.blocks[0].pre_norm

    @property
    def moe_num_experts(self) -> int:
        return self.blocks[0].moe_num_experts

    @property
    def moe_noisy_gating(self) -> bool:
        return self.blocks[0].moe_noisy_gating

    @property
    def moe_k(self) -> int:
        return self.blocks[0].moe_k

    @property
    def blocks(self) -> torch.nn.ModuleList:
        return self._blocks

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        out = x
        moe_loss = None
        for block in self.blocks:
            out, moe_loss_block = block(out)

            if moe_loss_block is not None:
                moe_loss = 0.0 if moe_loss is None else moe_loss
                moe_loss += moe_loss_block

        return out, moe_loss


class MLPMixer(SerializableModule):
    def __init__(self, config: MLPMixerConfig):
        """
        MLPMixer model consisting of an encoder and various embedding layers.

        Args:
            config (MLPMixerConfig): Configuration object containing model hyperparameters.
        """
        super().__init__()
        self._config = config

        # Initialize MLPMixer blocks
        self._encoder = MLPMixerEncoder(
            n_blocks=config.n_blocks,
            embed_dim=config.embed_dim,
            num_patches=self.num_patches + 1,
            token_ff_dim=config.token_ff_dim,
            channel_ff_dim=config.channel_ff_dim,
            ff_activation=config.ff_activation,
            ff_gated=config.ff_gated,
            dropout=config.dropout,
            pre_norm=config.pre_norm,
            moe_num_experts=config.moe_num_experts,
            moe_noisy_gating=config.moe_noisy_gating,
            moe_k=config.moe_k
        )

        # Initialize patch projection, positional embeddings, CLS token and MLP head
        self._patch_projection = ViTPatchEmbeddings(
            in_channels=config.in_channels,
            patch_dim=config.patch_dim,
            hidden_dim=config.embed_dim
        )
        self._cls_token = torch.nn.Parameter(torch.randn(config.embed_dim))
        self._pos_embedding = torch.nn.Embedding(self.num_patches + 1, config.embed_dim)
        self._mlp_head = torch.nn.Linear(config.embed_dim, config.output_dim)
        self._emb_dropout = torch.nn.Dropout(config.dropout)

        # Initialize final layer norm
        self._ln = None
        if self.config.final_layer_norm:
            self._ln = torch.nn.LayerNorm(config.embed_dim)

    @property
    def config(self) -> MLPMixerConfig:
        return self._config

    @property
    def num_patches(self) -> int:
        return int((self.config.image_dim / self.config.patch_dim)**2)

    @property
    def encoder(self) -> MLPMixerEncoder:
        return self._encoder

    @property
    def cls_token(self) -> torch.Tensor:
        return self._cls_token

    @property
    def pos_embedding(self) -> torch.nn.Module:
        return self._pos_embedding

    @property
    def mlp_head(self) -> torch.nn.Module:
        return self._mlp_head

    @property
    def emb_dropout(self) -> torch.nn.Dropout:
        return self._emb_dropout

    @property
    def patch_projection(self) -> torch.nn.Module:
        return self._patch_projection

    @property
    def ln(self) -> Optional[torch.nn.Module]:
        return self._ln

    def expanded_cls(self, *other_dims) -> torch.Tensor:
        """
        Expands the CLS token for a given batch size.

        Args:
            *other_dims: Additional dimensions.

        Returns:
            torch.Tensor: Expanded CLS token.
        """
        return self.cls_token.expand(*other_dims, 1, -1)

    def serialize_constructor_params(self, *args, **kwargs) -> dict:
        return {"config": self._config}

    def forward(self, x: torch.Tensor, *args, **kwargs) -> MLPMixerOutput:
        """
        Forward pass through the MLPMixer model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            MLPMixerOutput: Output containing the final output, CLS token embedding,
            average embedding, and optional MoE loss.
        """

        embeddings = self.patch_projection(x)
        cls_token = self.expanded_cls(*embeddings.shape[0:-2])
        embeddings = ConcatLayer(dim=-2)(cls_token, embeddings)

        # Apply positional embeddings
        pos_emb = self.pos_embedding(torch.arange(start=0, end=embeddings.shape[-2]).to(embeddings).long())
        embeddings = embeddings + pos_emb

        # Apply embedding dropout
        embeddings = self.emb_dropout(embeddings)

        # Apply encoder
        embeddings, moe_loss = self.encoder(embeddings)

        # Apply last layer normalization if needed
        if self.ln is not None:
            embeddings = self.ln(embeddings)

        # Get CLS token
        cls_token = embeddings[:, 0, :]
        avg_embedding = embeddings.mean(dim=1)

        # Apply MLP
        if self.config.use_cls_pool:
            output = self.mlp_head(cls_token)
        else:
            output = self.mlp_head(avg_embedding)

        # Construct output
        out = MLPMixerOutput(
            output=output,
            cls_embedding=cls_token,
            avg_embedding=avg_embedding,
            moe_loss=moe_loss
        )

        return out
