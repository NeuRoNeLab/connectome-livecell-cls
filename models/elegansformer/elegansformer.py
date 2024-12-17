from enum import Enum
from typing import Type, Union, Tuple, Optional, Dict, Any, List, NamedTuple
import einops
import torch
from pytorch_symbolic.useful_layers import ConcatLayer
from models.moe.tnmoe import TensorNetworkMoE
from models.vision.vit import ViTPatchEmbeddings, CustomViTEncoder
from utils.utils import SerializableConfig
from models.layers.misc import SerializableModule
from models.tensornet import TensorNetwork, FunctionalTensorNetwork, get_reservoir_tensornet_class


class ElegansFormerOutput(NamedTuple):
    # Network output
    output: torch.Tensor

    # CLS final embedding
    cls_embedding: torch.Tensor

    # Average final embedding
    avg_embedding: torch.Tensor

    # Attention weights
    attn_weights: List[Optional[torch.Tensor]]

    # Final embeddings
    final_embeddings: Optional[torch.Tensor] = None

    # MoE loss
    moe_loss: Optional[torch.Tensor] = None


class ElegansFormerModelType(str, Enum):
    FF = "ff"
    FF_SHARED = "ff-shared"
    PATCH_EMB = "patch-emb"
    HEAD = "head"
    POOL = "pool"
    ATTN_EMB = "attn-emb"


class ElegansFormerConfig(SerializableConfig):
    
    def __init__(self,
                 n_blocks: int,
                 embed_dim: int,
                 in_channels: int,
                 output_dim: int,
                 tensornet_config: Dict[str, Any],
                 tensornet_impl: str = "functional",
                 ff_dim: Optional[int] = None,
                 ff_activation: str = "gelu",
                 ff_gated: Optional[str] = None,
                 patch_dim: int = 16,
                 max_seq_len: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 pre_norm: bool = True,
                 use_cls_pool: bool = False,
                 moe_num_experts: int = 1,
                 moe_noisy_gating: bool = True,
                 moe_k: int = 2,
                 # shared_tensornet: bool = True,
                 model_type: ElegansFormerModelType = ElegansFormerModelType.FF_SHARED,
                 reservoir_tensornet_blocks: Optional[List[int]] = None):
        super().__init__()
        self.n_blocks: int = n_blocks
        self.embed_dim: int = embed_dim
        self.in_channels: int = in_channels
        self.output_dim: int = output_dim
        self.tensornet_config: Dict[str, Any] = tensornet_config
        self.tensornet_impl: str = tensornet_impl
        self.ff_dim: int = ff_dim
        self.ff_activation: str = ff_activation
        self.ff_gated: Optional[str] = ff_gated
        self.patch_dim: int = patch_dim
        self.max_seq_len: int = max_seq_len
        self.num_heads: int = num_heads
        self.dropout: float = dropout
        self.pre_norm: bool = pre_norm
        self.use_cls_pool: bool = use_cls_pool
        self.moe_num_experts: int = moe_num_experts
        self.moe_noisy_gating: bool = moe_noisy_gating
        self.moe_k: int = moe_k
        # self.shared_tensornet: bool = shared_tensornet
        self.model_type: ElegansFormerModelType = model_type
        self.reservoir_tensornet_blocks: List[int] = \
            reservoir_tensornet_blocks if reservoir_tensornet_blocks is not None else []


class TensorNetworkPatchEmbeddings(ViTPatchEmbeddings):
    def __init__(self,
                 tensornet_config: Dict[str, Any],
                 tensornet_class: Type[TensorNetwork] = FunctionalTensorNetwork,
                 in_channels: int = 3,
                 patch_dim: Union[int, Tuple[int, int]] = 16,
                 hidden_dim: int = 768):
        super().__init__(patch_dim=patch_dim, in_channels=in_channels, hidden_dim=hidden_dim)

        # Store attributes
        self._tensornet_config: Dict[str, Any] = tensornet_config

        # Delete parent class projection module
        del self._projection

        # Create tensor network
        tensornet = tensornet_class(**tensornet_config)
        self._projection = tensornet

    @property
    def tensornet_config(self) -> Dict[str, Any]:
        return self._tensornet_config

    def forward(self, pixel_values: torch.Tensor,) -> torch.Tensor:
        # interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.in_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.in_channels} but got {num_channels}."
            )
        # if not interpolate_pos_encoding:
        #     if height != self.image_size[0] or width != self.image_size[1]:
        #         raise ValueError(
        #             f"Input image size ({height}*{width}) doesn't match model"
        #             f" ({self.image_size[0]}*{self.image_size[1]})."
        #         )

        # Split in patches, flatten and project
        pixel_values = einops.rearrange(
            tensor=pixel_values,
            pattern="b c (h p0) (w p1) -> b (h w) (p0 p1 c)",
            p0=self.patch_dim[0],
            p1=self.patch_dim[1]
        )
        embeddings = self.projection(pixel_values)
        return embeddings


class TensorNetworkHead(torch.nn.Module):
    def __init__(self,
                 tensornet_config: Dict[str, Any],
                 output_dim: int,
                 tensornet_class: Type[TensorNetwork] = FunctionalTensorNetwork):
        super().__init__()
        self._tensornet_config: Dict[str, Any] = tensornet_config

        # Initialize layers
        self._tensornet = tensornet_class(**tensornet_config)
        self._final_proj = torch.nn.Linear(self.tensornet.embedding_dim, output_dim)

    @property
    def tensornet_config(self) -> Dict[str, Any]:
        return self._tensornet_config

    @property
    def tensornet(self) -> TensorNetwork:
        return self._tensornet

    @property
    def final_proj(self):
        return self._final_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.tensornet(x)
        out = self.final_proj(out)
        return out


class ElegansFormerBlock(torch.nn.Module):

    def __init__(self,
                 embed_dim: int,
                 tensor_network: Union[TensorNetwork, TensorNetworkMoE],
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 pre_norm: bool = True):
        # use_cross_attn: bool = False):
        super().__init__()

        if embed_dim != tensor_network.embedding_dim:
            raise ValueError(f"TensorNetwork embedding dimension mismatch: {embed_dim} != "
                             f"{tensor_network.embedding_dim}")

        if embed_dim != tensor_network.input_dim:
            raise ValueError(f"TensorNetwork embedding dimension mismatch: {embed_dim} != "
                             f"{tensor_network.input_dim}")

        # self._use_cross_att: bool = use_cross_attn
        self._pre_norm: bool = pre_norm

        # Initialize layers
        self._tn: TensorNetwork = tensor_network
        self._ln0 = torch.nn.LayerNorm(embed_dim)
        self._ln1 = torch.nn.LayerNorm(embed_dim)
        self._dropout_layer0 = torch.nn.Dropout(dropout)
        self._dropout_layer1 = torch.nn.Dropout(dropout)
        self._self_attn = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        '''
        self._cross_attn = torch.nn.Identity()
        if self._use_cross_attn:
            self._cross_attn = torch.nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        '''

    # @property
    # def use_cross_attn(self) -> bool:
    #     return self._use_cross_attn

    @property
    def embed_dim(self) -> int:
        return self._self_attn.embed_dim

    @property
    def num_heads(self) -> int:
        return self._self_attn.num_heads

    @property
    def dropout(self) -> float:
        return self._self_attn.dropout

    @property
    def pre_norm(self) -> bool:
        return self._pre_norm

    @property
    def tn(self) -> TensorNetwork:
        return self._tn

    @property
    def ln0(self) -> torch.nn.LayerNorm:
        return self._ln0

    @property
    def ln1(self) -> torch.nn.LayerNorm:
        return self._ln1

    @property
    def dropout_layer0(self) -> torch.nn.Dropout:
        return self._dropout_layer0

    @property
    def dropout_layer1(self) -> torch.nn.Dropout:
        return self._dropout_layer1

    @property
    def self_attn(self) -> torch.nn.MultiheadAttention:
        return self._self_attn

    def forward(self,
                x: torch.Tensor,
                attention_kwargs: Optional[Dict[str, Any]] = None,
                *args, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if attention_kwargs is None:
            attention_kwargs = {}

        # Pre-normalize if pre-norm
        if self.pre_norm:
            x = self.ln0(x)

        # Apply self-attention
        out, attn_weights = self.self_attn(query=x, key=x, value=x, **attention_kwargs)

        # Apply first dropout
        out = self.dropout_layer0(out)

        # Residual connection
        out = out + x

        # Apply the first LN after residual if post-norm, otherwise apply the second LN
        if not self.pre_norm:
            out = self.ln0(out)

        # Otherwise pre-normalize with the second LN
        else:
            out = self.ln1(out)

        # Apply tensornet or tensornet MoE
        moe_loss = None
        if isinstance(self.tn, TensorNetworkMoE):
            tn_out, moe_loss = self.tn(out)
        else:
            tn_out = self.tn(out)

        # Apply second dropout
        tn_out = self.dropout_layer1(tn_out)

        # Second residual connection
        out = tn_out + out

        # Apply second LN after residual if post-norm
        if not self.pre_norm:
            out = self.ln1(out)

        return out, attn_weights, moe_loss


class ElegansFormerEncoder(torch.nn.Module):

    def __init__(self,
                 n_blocks: int,
                 embed_dim: int,
                 tensornet_config: Dict[str, Any],
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 pre_norm: bool = True,
                 moe_num_experts: int = 1,
                 moe_noisy_gating: bool = True,
                 moe_k: int = 2,
                 shared_tensornet: bool = True,
                 reservoir_tensornet_blocks: Optional[List[int]] = None,
                 tensornet_class: Type[TensorNetwork] = FunctionalTensorNetwork):
        super().__init__()

        # TODO: test reservoir tensornet
        self._blocks: torch.nn.ModuleList = torch.nn.ModuleList()
        self._tensornet_config: Dict[str, Any] = tensornet_config
        self._moe_num_experts: int = moe_num_experts
        self._shared_tensornet: bool = shared_tensornet
        self._reservoir_tensornet_blocks: List[int] = \
            reservoir_tensornet_blocks if reservoir_tensornet_blocks is not None else []

        tensornet = None
        if shared_tensornet:
            if moe_num_experts > 1:
                tensornet = TensorNetworkMoE(
                    tensornet_config=tensornet_config,
                    num_experts=moe_num_experts,
                    noisy_gating=moe_noisy_gating,
                    k=moe_k,
                    tensornet_class=tensornet_class
                )
            else:
                tensornet = tensornet_class(**tensornet_config)

        reservoir_tensornet_blocks = set(self._reservoir_tensornet_blocks)
        for b in range(0, n_blocks):

            if not shared_tensornet:
                # Get the reservoir tensornet class if this block has to be made reservoir
                if b in reservoir_tensornet_blocks:
                    tensornet_class = get_reservoir_tensornet_class(tensornet_class)
                if moe_num_experts > 1:
                    tensornet = TensorNetworkMoE(
                        tensornet_config=tensornet_config,
                        num_experts=moe_num_experts,
                        noisy_gating=moe_noisy_gating,
                        k=moe_k,
                        tensornet_class=tensornet_class
                    )
                else:
                    tensornet = tensornet_class(**tensornet_config)

            block = ElegansFormerBlock(
                embed_dim=embed_dim,
                tensor_network=tensornet,
                num_heads=num_heads,
                dropout=dropout,
                pre_norm=pre_norm
            )
            self._blocks.append(block)

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    @property
    def embed_dim(self) -> int:
        return self.blocks[0].embed_dim

    @property
    def tensornet_config(self) -> Dict[str, Any]:
        return self._tensornet_config

    @property
    def num_heads(self) -> int:
        return self.blocks[0].num_heads

    @property
    def dropout(self) -> float:
        return self.blocks[0].dropout

    @property
    def pre_norm(self) -> bool:
        return self.blocks[0].pre_norm

    @property
    def moe_num_experts(self) -> int:
        return self._moe_num_experts

    @property
    def moe_noisy_gating(self) -> Optional[bool]:
        return None if self.moe_num_experts < 2 else self.blocks[0].tn.noisy_gating

    @property
    def moe_k(self) -> Optional[int]:
        return None if self.moe_num_experts < 2 else self.blocks[0].tn.k

    @property
    def shared_tensornet(self) -> bool:
        return self._shared_tensornet

    @property
    def reservoir_tensornet(self) -> Optional[List[int]]:
        return self._reservoir_tensornet_blocks

    @property
    def blocks(self) -> torch.nn.ModuleList:
        return self._blocks

    def forward(self,
                x: torch.Tensor,
                attention_kwargs: Optional[Dict[str, Any]] = None,
                *args, **kwargs) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]], Optional[torch.Tensor]]:
        if attention_kwargs is None:
            attention_kwargs = {}

        out = x
        attention_weights = []
        moe_loss = None

        for block in self.blocks:
            out, attn_weights_block, moe_loss_block = block(out, attention_kwargs=attention_kwargs)

            attention_weights.append(attn_weights_block)
            if moe_loss_block is not None:
                moe_loss = 0.0 if moe_loss is None else moe_loss
                moe_loss += moe_loss_block

        return out, attention_weights, moe_loss


class ElegansFormer(SerializableModule):

    def __init__(self, config: ElegansFormerConfig):
        super().__init__()
        self._config: ElegansFormerConfig = config

        if config.tensornet_impl == "functional":
            tensornet_class = FunctionalTensorNetwork
        else:
            raise NotImplementedError(f"{config.tensornet_impl} not supported.")

        if config.tensornet_config["embedding_dim"] != config.embed_dim:
            raise ValueError(f"TensorNetwork embedding dim {config.tensornet_config['embedding_dim']} != embedding dim "
                             f"{config.embed_dim}")

        # Initialize encoder
        if config.model_type in [ElegansFormerModelType.FF, ElegansFormerModelType.FF_SHARED]:
            self._encoder = ElegansFormerEncoder(
                n_blocks=config.n_blocks,
                embed_dim=config.embed_dim,
                tensornet_config=config.tensornet_config,
                num_heads=config.num_heads,
                dropout=config.dropout,
                pre_norm=config.pre_norm,
                moe_num_experts=config.moe_num_experts,
                moe_noisy_gating=config.moe_noisy_gating,
                moe_k=config.moe_k,
                tensornet_class=tensornet_class,
                shared_tensornet=config.shared_tensornet,
                reservoir_tensornet_blocks=config.reservoir_tensornet_blocks
            )
        else:
            self._encoder = CustomViTEncoder(
                n_blocks=config.n_blocks,
                embed_dim=config.embed_dim,
                ff_dim=config.ff_dim,
                ff_activation=config.ff_activation,
                ff_gated=config.ff_gated,
                num_heads=config.num_heads,
                dropout=config.dropout,
                pre_norm=config.pre_norm,
                moe_num_experts=config.moe_num_experts,
                moe_noisy_gating=config.moe_noisy_gating,
                moe_k=config.moe_k,
                shared_ff=False,
            )

        # Initialize patch embeddings
        # self._patch_projection = torch.nn.Linear(config.patch_dim ** 2 * config.in_channels, config.embed_dim)
        if config.model_type == ElegansFormerModelType.PATCH_EMB:
            self._patch_projection = TensorNetworkPatchEmbeddings(
                tensornet_config=config.tensornet_config,
                tensornet_class=tensornet_class,
                in_channels=config.in_channels,
                patch_dim=config.patch_dim,
                hidden_dim=config.embed_dim
            )
        else:
            self._patch_projection = ViTPatchEmbeddings(
                in_channels=config.in_channels,
                patch_dim=config.patch_dim,
                hidden_dim=config.embed_dim
            )

        # Initialize CLS token & positional embeddings
        self._cls_token = torch.nn.Parameter(torch.randn(config.embed_dim))
        self._pos_embedding = torch.nn.Embedding(config.max_seq_len + 1, config.embed_dim)

        # Initialize head
        if config.model_type == ElegansFormerModelType.HEAD:
            if "input_sequences" in config.tensornet_config and config.tensornet_config["input_sequences"] is not None:
                config.tensornet_config["input_sequences"] = None
            self._head = TensorNetworkHead(
                tensornet_config=config.tensornet_config,
                output_dim=config.output_dim,
                tensornet_class=tensornet_class
            )
        else:
            self._head = torch.nn.Linear(config.embed_dim, config.output_dim)

        # Initialize embedding dropout
        self._emb_dropout = torch.nn.Dropout(config.dropout)

    @property
    def config(self) -> ElegansFormerConfig:
        return self._config

    @property
    def encoder(self) -> Union[ElegansFormerEncoder, CustomViTEncoder]:
        return self._encoder

    @property
    def cls_token(self) -> torch.Tensor:
        return self._cls_token

    @property
    def patch_projection(self) -> torch.nn.Module:
        return self._patch_projection

    @property
    def pos_embedding(self) -> torch.nn.Embedding:
        return self._pos_embedding

    @property
    def head(self) -> torch.nn.Module:
        return self._head

    @property
    def emb_dropout(self) -> torch.nn.Dropout:
        return self._emb_dropout

    def expanded_cls(self, batch_size: int) -> torch.Tensor:
        return self.cls_token.expand(batch_size, 1, -1)

    def serialize_constructor_params(self, *args, **kwargs) -> dict:
        return {
            "config": self.config
        }

    def forward(self,
                x: torch.Tensor,
                attention_kwargs: Optional[Dict[str, Any]] = None,
                *args, **kwargs) -> ElegansFormerOutput:
        if attention_kwargs is None:
            attention_kwargs = {}

        # Split in patches, flatten and project
        # x = einops.rearrange(
        #     tensor=x,
        #     pattern="b c (h p0) (w p1) -> b (h w) (p0 p1 c)",
        #     p0=self.config.patch_dim,
        #     p1=self.config.patch_dim
        # )
        embeddings = self.patch_projection(x)
        cls_token = self.expanded_cls(batch_size=x.shape[0])
        embeddings = ConcatLayer(dim=1)(cls_token, embeddings)

        # Apply positional embeddings
        pos_emb = self.pos_embedding(torch.arange(start=0, end=embeddings.shape[1]).to(embeddings).long())
        embeddings = embeddings + pos_emb

        # Apply embedding dropout
        embeddings = self.emb_dropout(embeddings)

        # Apply encoder
        embeddings, attn_weights, moe_loss = self.encoder(embeddings, attention_kwargs=attention_kwargs)

        # Get CLS token
        cls_token = embeddings[:, 0, :]
        avg_embedding = embeddings.mean(dim=1)

        # Apply MLP
        if self.config.use_cls_pool:
            output = self.head(cls_token)
        else:
            output = self.head(avg_embedding)

        # Construct output
        elegansformer_out = ElegansFormerOutput(
            output=output,
            cls_embedding=cls_token,
            avg_embedding=avg_embedding,
            attn_weights=attn_weights,
            moe_loss=moe_loss
        )

        return elegansformer_out
