from enum import Enum
from typing import Type, Union, Optional, Dict, Any, NamedTuple
import torch
from models.vision.backbone_utils import get_torchvision_backbone, TorchVisionBackboneConfig
from utils.utils import SerializableConfig
from models.layers.misc import SerializableModule
from models.tensornet import TensorNetwork, FunctionalTensorNetwork, get_reservoir_tensornet_class


class VisionElegansOutput(NamedTuple):
    # Network output
    output: torch.Tensor

    embedding: Optional[torch.Tensor] = None


class VisionElegansModelType(str, Enum):
    HEAD = "head"


class VisionElegansConfig(SerializableConfig):

    def __init__(self,
                 in_channels: int,
                 output_dim: int,
                 backbone_config: TorchVisionBackboneConfig,
                 tensornet_config: Dict[str, Any],
                 tensornet_impl: str = "functional",
                 dropout: float = 0.0,
                 reservoir_tensornet: bool = False,
                 residual_tensornet: bool = False
                 ):
        super().__init__()
        self.in_channels: int = in_channels
        self.output_dim: int = output_dim
        self.reservoir_tensornet: bool = reservoir_tensornet
        self.backbone_config: TorchVisionBackboneConfig = backbone_config
        self.tensornet_config: Dict[str, Any] = tensornet_config
        self.tensornet_impl: str = tensornet_impl
        self.dropout: float = dropout
        self.reservoir_tensornet: bool = reservoir_tensornet
        self.residual_tensornet: bool = residual_tensornet

class TensorNetworkHead(torch.nn.Module):
    def __init__(self,
                 tensornet_config: Dict[str, Any],
                 output_dim: int,
                 tensornet_class: Type[TensorNetwork] = FunctionalTensorNetwork,
                 reservoir_tensornet: bool = False,
                 residual_tensornet: bool = False,):
        super().__init__()
        self._tensornet_config: Dict[str, Any] = tensornet_config
        self._reservoir_tensornet: bool = reservoir_tensornet
        self._residual_tensornet: bool = residual_tensornet

        # Initialize layers
        if reservoir_tensornet:
            tensornet_class = get_reservoir_tensornet_class(tn_class=tensornet_class)
        self._tensornet = tensornet_class(**tensornet_config)
        self._residual_proj = torch.nn.Identity()
        if self.tensornet.embedding_dim != self.tensornet.input_dim and residual_tensornet:
            self._residual_proj = torch.nn.Linear(self.tensornet.input_dim, self.tensornet.embedding_dim)
        self._final_proj = torch.nn.Linear(self.tensornet.embedding_dim, output_dim)

    @property
    def tensornet_config(self) -> Dict[str, Any]:
        return self._tensornet_config

    @property
    def reservoir_tensornet(self) -> bool:
        return self._reservoir_tensornet

    @property
    def residual_tensornet(self) -> bool:
        return self._residual_tensornet

    @property
    def tensornet(self) -> TensorNetwork:
        return self._tensornet

    @property
    def residual_proj(self) -> torch.nn.Module:
        return self._residual_proj

    @property
    def final_proj(self) -> torch.nn.Module:
        return self._final_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x) if self.residual_tensornet else 0
        out = self.tensornet(x) + residual
        out = self.final_proj(out)
        return out


class VisionElegans(SerializableModule):

    def __init__(self, config: VisionElegansConfig):
        super().__init__()
        self._config: VisionElegansConfig = config

        if config.tensornet_impl == "functional":
            tensornet_class = FunctionalTensorNetwork
        else:
            raise NotImplementedError(f"{config.tensornet_impl} not supported.")

        # Initialize backbone
        self._backbone = get_torchvision_backbone(config.backbone_config)
        self.check_backbone_input()

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
    def config(self) -> VisionElegansConfig:
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

    def check_backbone_input(self):
        pass
        if self.config.in_channels == 1 and isinstance(self.backbone, torch.nn.Sequential) and \
                hasattr(self.backbone[0], "conv1"):
            out_channels = self.backbone[0].conv1.out_channels
            kernel_size = self.backbone[0].conv1.kernel_size
            stride = self.backbone[0].conv1.stride
            padding = self.backbone[0].conv1.padding
            self.backbone[0].conv1 = torch.nn.Conv2d(in_channels=self.config.in_channels,
                                                     out_channels=out_channels,
                                                     kernel_size=kernel_size[0],
                                                     stride=stride[0],
                                                     padding=padding)
            pass
        # self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, self.config.embed_dim)

    def forward(self,
                x: torch.Tensor,
                attention_kwargs: Optional[Dict[str, Any]] = None,
                *args, **kwargs) -> VisionElegansOutput:
        if attention_kwargs is None:
            attention_kwargs = {}

        x = self.backbone(x)
        # x = self.features_projector(x.flatten(1))
        output = self.head(x)
        # Construct output
        elegansformer_out = VisionElegansOutput(
            output=output,
            embedding=x
        )

        return elegansformer_out
