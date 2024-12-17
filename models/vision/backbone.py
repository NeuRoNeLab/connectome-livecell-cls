from typing import Optional, Tuple, NamedTuple, Union, Dict, Any
import torch
import torch.nn

from models.layers import SerializableModule
from models.vision.backbone_utils import TorchVisionBackboneConfig, get_torchvision_backbone


class BackboneOutput(NamedTuple):
    # Network output
    output: torch.Tensor


# class CustomBackboneConfig(SerializableModule):
#     def __init__(self,
#                  in_channels: int,
#                  output_dim: int,
#                  backbone_config:
#                  ):
#         super().__init__()


class TorchVisionBackbone(SerializableModule):

    def __init__(self, config: TorchVisionBackboneConfig):
        super().__init__()
        self._config: TorchVisionBackboneConfig = config

        # Initialize backbone
        self._backbone = get_torchvision_backbone(config)
        self.check_backbone_input()
        if self._config.output_dim is not None:
            self.modify_backbone_classifier()


    @property
    def config(self) -> TorchVisionBackboneConfig:
        return self._config

    @property
    def backbone(self) -> Union[torch.nn.Module]:
        return self._backbone

    def serialize_constructor_params(self, *args, **kwargs) -> dict:
        return {
            "config": self.config
        }

    def modify_backbone_classifier(self):
        if hasattr(self._backbone, "classifier"): #EfficientNet like
            if isinstance(self.backbone.classifier, torch.nn.Sequential):
                in_channels = self.backbone.classifier[-1].in_features
            else:
                in_channels = self.backbone.classifier.in_features
            self._backbone.classifier = torch.nn.Linear(in_channels, self.config.output_dim)

            # self._backbone.classifier = torch.nn.Sequential(
            #     torch.nn.Linear(in_channels, 512),
            #     torch.nn.Sigmoid(),
            #     torch.nn.Linear(512, self._config.backbone_kwargs['output_dim']),
            # )
        elif hasattr(self.backbone, "fc"):
            in_channels = self.backbone.fc.in_features
            self.backbone.fc = torch.nn.Linear(in_channels, self.config.output_dim)

    def check_backbone_input(self):
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
        elif self.config.in_channels == 1 and isinstance(self.backbone, torch.nn.Sequential) and \
                hasattr(self.backbone[0], "features"):
            out_channels = self.backbone.features[0][0].out_channels
            kernel_size = self.backbone.features[0][0].conv1.kernel_size
            stride = self.backbone.features[0][0].stride
            padding = self.backbone.features[0][0].padding
            self.backbone.features[0][0] = torch.nn.Conv2d(in_channels=self.config.in_channels,
                                                     out_channels=out_channels,
                                                     kernel_size=kernel_size[0],
                                                     stride=stride[0],
                                                     padding=padding)

    def forward(self,
                x: torch.Tensor,
                attention_kwargs: Optional[Dict[str, Any]] = None,
                *args, **kwargs) -> BackboneOutput:
        if attention_kwargs is None:
            attention_kwargs = {}

        output = self.backbone(x)


        # Construct output
        elegansformer_out = BackboneOutput(
            output=output,
        )

        return elegansformer_out
