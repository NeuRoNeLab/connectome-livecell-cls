from enum import Enum
from typing import Union, Optional, Dict, Any, NamedTuple
import torch
from models.vision.swin_v2 import SwinTransformerV2, CustomSwinConfig, CustomSwinOutput
from utils.utils import SerializableConfig
from models.layers.misc import SerializableModule
from models.elegans_vision.elegans_vision import TensorNetworkHead
from models.tensornet import TensorNetwork, FunctionalTensorNetwork, get_reservoir_tensornet_class


class ElegansSwinOutput(NamedTuple):
    # Network output
    output: torch.Tensor
    embedding: Optional[torch.Tensor] = None


class ElegansSwinModelType(str, Enum):
    HEAD = "head"


class ElegansSwinConfig(SerializableConfig):

    def __init__(self,
                 in_channels: int,
                 output_dim: int,
                 backbone_config: CustomSwinConfig,
                 tensornet_config: Dict[str, Any],
                 tensornet_impl: str = "functional",
                 dropout: float = 0.0,
                 reservoir_tensornet: bool = False,
                 residual_tensornet: bool = False):
        super().__init__()

        # SwinTransformer should only be used as encoder
        backbone_config.use_as_encoder = True

        self.in_channels: int = in_channels
        self.output_dim: int = output_dim
        self.reservoir_tensornet: bool = reservoir_tensornet
        self.backbone_config: CustomSwinConfig = backbone_config
        self.tensornet_config: Dict[str, Any] = tensornet_config
        self.tensornet_impl: str = tensornet_impl
        self.dropout: float = dropout
        self.reservoir_tensornet: bool = reservoir_tensornet
        self.residual_tensornet: bool = residual_tensornet


class ElegansSwin(SerializableModule):

    def __init__(self, config: ElegansSwinConfig):
        super().__init__()
        self._config: ElegansSwinConfig = config

        if config.tensornet_impl == "functional":
            tensornet_class = FunctionalTensorNetwork
        else:
            raise NotImplementedError(f"{config.tensornet_impl} not supported.")

        # Initialize backbone
        self._backbone = SwinTransformerV2(config.backbone_config)

        # Initialize head
        if "input_sequences" in config.tensornet_config and config.tensornet_config["input_sequences"] is not None:
            config.tensornet_config["input_sequences"] = None
        self._head = TensorNetworkHead(
            tensornet_config=config.tensornet_config,
            output_dim=config.output_dim,
            tensornet_class=tensornet_class,
            reservoir_tensornet=config.reservoir_tensornet,
            residual_tensornet=config.residual_tensornet
        )

        # Initialize embedding dropout
        self._emb_dropout = torch.nn.Dropout(config.dropout)

    @property
    def config(self) -> ElegansSwinConfig:
        return self._config

    @property
    def backbone(self) -> Union[torch.nn.Module]:
        return self._backbone

    @property
    def head(self) -> torch.nn.Module:
        return self._head

    def serialize_constructor_params(self, *args, **kwargs) -> dict:
        return {
            "config": self.config
        }

    def forward(self,
                x: torch.Tensor,
                *args, **kwargs) -> ElegansSwinOutput:

        # Compute the embedding
        swin_out: CustomSwinOutput = self.backbone(x)
        x = swin_out.avg_embedding

        # Apply TN head
        output = self.head(x)

        # Construct output
        elegansswin_out = ElegansSwinOutput(
            output=output,
            embedding=x
        )

        return elegansswin_out
