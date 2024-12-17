import collections
from typing import Union, Tuple, Optional, Dict, Any, List, NamedTuple
import torch
from pytorch_symbolic.useful_layers import ConcatLayer
from utils.utils import SerializableConfig
from models.layers.misc import SerializableModule
from models.layers.mlp import FFN
from models.moe.moe import MoE


class CustomViTOutput(NamedTuple):
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


class CustomViTConfig(SerializableConfig):

    def __init__(self,
                 n_blocks: int,
                 embed_dim: int,
                 in_channels: int,
                 output_dim: int,
                 ff_dim: Optional[int] = None,
                 ff_activation: str = 'gelu',
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
                 shared_ff: bool = False):
        super().__init__()
        self.n_blocks: int = n_blocks
        self.embed_dim: int = embed_dim
        self.in_channels: int = in_channels
        self.output_dim: int = output_dim
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
        self.shared_ff: bool = shared_ff


class ViTPatchEmbeddings(torch.nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self,
                 in_channels: int = 3,
                 patch_dim: Union[int, Tuple[int, int]] = 16,
                 hidden_dim: int = 768,):
        super().__init__()
        # image_size, patch_size = config.image_size, config.patch_size
        # image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_dim = patch_dim if isinstance(patch_dim, collections.abc.Iterable) else (patch_dim, patch_dim)
        # num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        # self.image_size = image_size

        self._patch_dim: Union[int, Tuple[int, int]] = patch_dim
        self._in_channels: int = in_channels
        # self.num_patches = num_patches

        self._projection = torch.nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_dim, stride=patch_dim)

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def patch_dim(self) -> Union[int, Tuple[int, int]]:
        return self._patch_dim

    @property
    def out_channels(self) -> int:
        return self._projection.out_channels

    @property
    def hidden_dim(self) -> int:
        return self._projection.out_channels

    @property
    def projection(self) -> torch.nn.Module:
        return self._projection

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
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


class CustomViTBlock(torch.nn.Module):

    def __init__(self,
                 embed_dim: int,
                 ff_block: Union[FFN, MoE],
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 pre_norm: bool = True):
        super().__init__()

        self._pre_norm: bool = pre_norm
        self._ff = ff_block
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
    def ff_activation(self) -> str:
        return self._ff_activation

    @property
    def ff_dim(self) -> int:
        if isinstance(self._ff, MoE):
            return self._ff.experts[0].out_features
        return self._ff.ff_dim

    @property
    def ff(self) -> torch.nn.Module:
        return self._ff

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

        # Apply FF or FF-MoE
        moe_loss = None
        if isinstance(self.ff, MoE):
            ff_out, moe_loss = self.ff(out)
        else:
            ff_out = self.ff(out)

        # Apply second dropout
        ff_out = self.dropout_layer1(ff_out)

        # Second residual connection
        out = ff_out + out

        # Apply second LN after residual if post-norm
        if not self.pre_norm:
            out = self.ln1(out)

        return out, attn_weights, moe_loss


class CustomViTEncoder(torch.nn.Module):

    def __init__(self,
                 n_blocks: int,
                 embed_dim: int,
                 ff_dim: Optional[int] = None,
                 ff_activation: str = 'gelu',
                 ff_gated: Optional[str] = None,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 pre_norm: bool = True,
                 moe_num_experts: int = 1,
                 moe_noisy_gating: bool = True,
                 moe_k: int = 2,
                 shared_ff: bool = False):
        super().__init__()

        self._blocks: torch.nn.ModuleList = torch.nn.ModuleList()
        self._moe_num_experts: int = moe_num_experts
        self._shared_ff: bool = shared_ff

        ff = None
        ff_config = {
            "embed_dim": embed_dim,
            "ff_dim": ff_dim,
            "ff_activation": ff_activation,
            "gated": ff_gated
        }
        self._ff_config = ff_config
        if shared_ff:
            if moe_num_experts > 1:
                ff = MoE(
                    input_size=embed_dim,
                    output_size=embed_dim,
                    num_experts=moe_num_experts,
                    noisy_gating=moe_noisy_gating,
                    k=moe_k,
                    expert_class=FFN,
                    expert_kwargs=ff_config
                )
            else:
                ff = FFN(**ff_config)

        for _ in range(0, n_blocks):

            if not shared_ff:
                if moe_num_experts > 1:
                    ff = MoE(
                        input_size=embed_dim,
                        output_size=embed_dim,
                        num_experts=moe_num_experts,
                        noisy_gating=moe_noisy_gating,
                        k=moe_k,
                        expert_class=FFN,
                        expert_kwargs=ff_config
                    )
                else:
                    ff = FFN(**ff_config)

            block = CustomViTBlock(
                embed_dim=embed_dim,
                ff_block=ff,
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
    def ff_config(self) -> Dict[str, Any]:
        return self._ff_config

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
        return None if self.moe_num_experts < 2 else self.blocks[0].ff.noisy_gating

    @property
    def shared_ff(self) -> bool:
        return self._shared_ff

    @property
    def moe_k(self) -> Optional[int]:
        return None if self.moe_num_experts < 2 else self.blocks[0].ff.k

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


class CustomViT(SerializableModule):

    def __init__(self, config: CustomViTConfig):
        super().__init__()
        self._config: CustomViTConfig = config

        # Initialize encoder, cls token, patch projections, positional embeddings and linear head
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
            shared_ff=config.shared_ff
        )
        # self._patch_projection = torch.nn.Linear(config.patch_dim ** 2 * config.in_channels, config.embed_dim)
        self._patch_projection = ViTPatchEmbeddings(
            in_channels=config.in_channels,
            patch_dim=config.patch_dim,
            hidden_dim=config.embed_dim
        )
        self._cls_token = torch.nn.Parameter(torch.randn(config.embed_dim))
        self._pos_embedding = torch.nn.Embedding(config.max_seq_len + 1, config.embed_dim)
        self._mlp_head = torch.nn.Linear(config.embed_dim, config.output_dim)
        self._emb_dropout = torch.nn.Dropout(config.dropout)

    @property
    def config(self) -> CustomViTConfig:
        return self._config

    @property
    def encoder(self) -> CustomViTEncoder:
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
    def mlp_head(self) -> torch.nn.Module:
        return self._mlp_head

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
                *args, **kwargs) -> CustomViTOutput:
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
            output = self.mlp_head(cls_token)
        else:
            output = self.mlp_head(avg_embedding)

        # Construct output
        vit_out = CustomViTOutput(
            output=output,
            cls_embedding=cls_token,
            avg_embedding=avg_embedding,
            attn_weights=attn_weights,
            moe_loss=moe_loss
        )

        return vit_out
