from enum import Enum
from typing import Optional, NamedTuple, Dict, Any
import torch
from pytorch_symbolic.useful_layers import ConcatLayer
from models.layers import SerializableModule
from models.tensornet import FunctionalTensorNetwork
from models.vision.mlpmixer import MLPMixerEncoder
from utils.utils import SerializableConfig
from models.vision.vit import ViTPatchEmbeddings
from models.elegans_vision.elegans_vision import TensorNetworkHead


class ElegansMLPMixerOutput(NamedTuple):
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
    

class ElegansMLPMixerModelType(str, Enum):
    HEAD = "head"


class ElegansMLPMixerConfig(SerializableConfig):

    def __init__(self,
                 n_blocks: int,
                 embed_dim: int,
                 in_channels: int,
                 output_dim: int,
                 tensornet_config: Dict[str, Any],
                 tensornet_impl: str = "functional",
                 reservoir_tensornet: bool = False,
                 residual_tensornet: bool = False,
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
                 use_cls_pool: bool = False,
                 model_type: ElegansMLPMixerModelType = "HEAD"):
        """
        Configuration class for the ElegansMLPMixer model.

        Args:
            n_blocks (int): Number of Mixer blocks.
            embed_dim (int): Dimensionality of the embeddings.
            in_channels (int): Number of input channels.
            output_dim (int): Dimensionality of the output.
            tensornet_config (Dict[str, Any]): Configuration options for TensorNetwork.
            tensornet_impl (str, optional): Implementation of the TensorNetwork. Defaults to "functional".
            reservoir_tensornet (bool, optional): Whether to use reservoir tensornet. Defaults to False.
            residual_tensornet (bool, optional): Whether to use residual tensornet. Defaults to False.
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
            model_type (ElegansMLPMixerModelType, optional): Type of the MLPMixer model.
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
        self.tensornet_config: Dict[str, Any] = tensornet_config
        self.tensornet_impl: str = tensornet_impl
        self.reservoir_tensornet: bool = reservoir_tensornet
        self.residual_tensornet: bool = residual_tensornet
        self.model_type: ElegansMLPMixerModelType = model_type


class ElegansMLPMixer(SerializableModule):
    def __init__(self, config: ElegansMLPMixerConfig):
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
        self._emb_dropout = torch.nn.Dropout(config.dropout)

        if config.tensornet_impl == "functional":
            tensornet_class = FunctionalTensorNetwork
        else:
            raise NotImplementedError(f"{config.tensornet_impl} not supported.")
        self._tn_head = TensorNetworkHead(
            tensornet_config=config.tensornet_config,
            output_dim=config.output_dim,
            tensornet_class=tensornet_class,
            reservoir_tensornet=config.reservoir_tensornet,
            residual_tensornet=config.residual_tensornet
        )

        # Initialize final layer norm
        self._ln = None
        if self.config.final_layer_norm:
            self._ln = torch.nn.LayerNorm(config.embed_dim)

    @property
    def config(self) -> ElegansMLPMixerConfig:
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
    def tn_head(self) -> torch.nn.Module:
        return self._tn_head

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

    def forward(self, x: torch.Tensor, *args, **kwargs) -> ElegansMLPMixerOutput:
        """
        Forward pass through the ElegansMLPMixer model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            ElegansMLPMixerOutput: Output containing the final output, CLS token embedding,
                average embedding, the tensor network embedding, and optional MoE loss.
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
            output = self.tn_head(cls_token)
        else:
            output = self.tn_head(avg_embedding)

        # Construct output
        out = ElegansMLPMixerOutput(
            output=output,
            cls_embedding=cls_token,
            avg_embedding=avg_embedding,
            moe_loss=moe_loss
        )

        return out
