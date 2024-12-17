import abc
from collections import OrderedDict
from typing import Optional, Any, Dict, List, Union
import torch
import torchvision
from torchvision.models import WeightsEnum
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from pytorch_symbolic.useful_layers import StackLayer, LambdaOpLayer
from utils.utils import SerializableConfig
from models.layers.misc import ParallelModules


class BackboneConfig(SerializableConfig, abc.ABC):
    """
    Base abstract configuration class for backbone models.

    Args:
        name (str): The name of the backbone model.

    Attributes:
        name (str): The name of the backbone model.
    """
    def __init__(self, name: str):
        super().__init__()
        self.name: str = name


class TorchVisionBackboneConfig(BackboneConfig):
    """
    Configuration class for TorchVision backbone models.

    Args:
        name (str): The name of the TorchVision backbone model. Default is "resnet50".
        weights (Optional[WeightsEnum]): Pre-trained weights to use for the model. Default is None.
        feature_extractor_return_nodes (Optional[Union[List[str], Dict[str, str]]]): Nodes to return for feature
            extraction. Can be a list of node names or a dictionary mapping original node names to new names. Default is
            None (in this case, the default output of the backbone will be used).
        backbone_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments for the backbone model. Default is
            None.

    Attributes:
        weights (WeightsEnum): Pre-trained weights for the model.
        feature_extractor_return_nodes (Optional[Dict[str, str]]): Nodes to return for feature extraction, as a
            dictionary.
        backbone_kwargs (Dict[str, Any]): Additional keyword arguments for the backbone model.
    """
    def __init__(self,
                 name: str = "resnet50",
                 weights: Optional[WeightsEnum] = None,
                 feature_extractor_return_nodes: Optional[Union[List[str], Dict[str, str]]] = None,
                 backbone_kwargs: Optional[Dict[str, Any]] = None,
                 in_channels: Optional[int] = None,
                 output_dim: Optional[int] = None):
        super().__init__(name=name)
        self.weights: WeightsEnum = weights
        self.backbone_kwargs: Dict[str, Any] = backbone_kwargs if backbone_kwargs is not None else {}
        self.in_channels: int = in_channels
        self.output_dim: int = output_dim

        if feature_extractor_return_nodes is not None:
            if isinstance(feature_extractor_return_nodes, list):
                feature_extractor_return_nodes = {n: n for n in feature_extractor_return_nodes}
            feature_extractor_return_nodes = OrderedDict(feature_extractor_return_nodes)
        self.feature_extractor_return_nodes: Optional[Dict[str, str]] = feature_extractor_return_nodes


def get_torchvision_backbone(backbone_config: TorchVisionBackboneConfig) -> torch.nn.Module:
    """
    Creates a TorchVision backbone model based on the provided configuration. The function performs the following steps:
        1. Loads the model from TorchVision using the specified name and weights.
        2. If feature extraction nodes are specified, creates a feature extractor model.
        3. If there is only one feature extraction node, wraps the model to return a single tensor instead of a
            dictionary.

    Args:
        backbone_config (TorchVisionBackboneConfig): Configuration for the backbone model.

    Returns:
        torch.nn.Module: The backbone model.

    Example usage:
        >>> config_ = TorchVisionBackboneConfig(name="resnet50", weights=None)
        >>> model_ = get_torchvision_backbone(config_)
    """
    model = torchvision.models.get_model(
        name=backbone_config.name,
        weights=backbone_config.weights,
        **backbone_config.backbone_kwargs
    )

    # Create feature extractor if needed
    if backbone_config.feature_extractor_return_nodes is not None and \
            len(backbone_config.feature_extractor_return_nodes) > 0:
        model = create_feature_extractor(model=model, return_nodes=backbone_config.feature_extractor_return_nodes)

        # If there's only 1 return key, don't return a dictionary but a tensor directly
        if len(backbone_config.feature_extractor_return_nodes) == 1:
            lone_key = list(backbone_config.feature_extractor_return_nodes.values())[0]
            model = torch.nn.Sequential(model, LambdaOpLayer(lambda x: x[lone_key]))

    return model


class FeaturePyramidBackboneConfig(TorchVisionBackboneConfig):
    """
    Configuration class for a feature pyramid network (FPN) backbone model.

    Args:
        out_channels (int): The number of output channels for the FPN layers.
        in_channels (int): The number of input channels. Default is 3.
        name (str): The name of the TorchVision backbone model. Default is "resnet50".
        weights (Optional[WeightsEnum]): Pre-trained weights to use for the model. Default is None.
        fp_return_layers (Optional[Union[List[str], Dict[str, str]]]): Nodes to return for feature extraction.
            Can be a list of node names or a dictionary mapping original node names to new names. Default is None.
        extra_block_type (Optional[str]): Type of extra block to add to the FPN. Supported type is "maxpool". Default is
            None.
        backbone_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments for the backbone model. Default is
            None.
        global_pooling (Optional[str]): Type of global pooling to apply. Supported types are "avg", "mean", "sum",
            "add", and "max". Default is None.

    Attributes:
        out_channels (int): The number of output channels for the FPN layers.
        in_channels (int): The number of input channels.
        extra_block_type (Optional[str]): Type of extra block to add to the FPN.
        global_pooling (Optional[str]): Type of global pooling to apply.
    """
    def __init__(self,
                 out_channels: int,
                 in_channels: int = 3,
                 name: str = "resnet50",
                 weights: Optional[WeightsEnum] = None,
                 fp_return_layers: Optional[Union[List[str], Dict[str, str]]] = None,
                 extra_block_type: Optional[str] = None,
                 backbone_kwargs: Optional[Dict[str, Any]] = None,
                 global_pooling: Optional[str] = None):
        if fp_return_layers is None and name == "resnet50":
            fp_return_layers = {f"layer{i}": f"layer{i}" for i in [1, 2, 3, 4]}
        elif fp_return_layers is None:
            raise ValueError(f"fp_layers cannot be None unless the model is 'resnet50', found {name}.")

        super().__init__(
            name=name,
            weights=weights,
            feature_extractor_return_nodes=fp_return_layers,
            backbone_kwargs=backbone_kwargs
        )
        self.out_channels: int = out_channels
        self.in_channels: int = in_channels
        self.extra_block_type: Optional[str] = extra_block_type
        self.global_pooling: Optional[str] = global_pooling


class FPNBackbone(torch.nn.Module):
    # noinspection PyUnresolvedReferences
    """
    Feature Pyramid Network (FPN) backbone model. Combines a standard convolutional neural network backbone with an FPN
    to enhance the multiscale feature representation capability.

    Args:
        config (FeaturePyramidBackboneConfig): Configuration for the FPN backbone model.

    Attributes:
        config (FeaturePyramidBackboneConfig): Configuration for the FPN backbone model.
        body (torch.nn.Module): The backbone model.
        fpn (FeaturePyramidNetwork): The feature pyramid network.
        global_pooling (torch.nn.Module): The global pooling layer.

    Example usage:
        >>> config_ = FeaturePyramidBackboneConfig(out_channels=256, name="resnet50", global_pooling="avg")
        >>> model_ = FPNBackbone(config_)
        >>> inp_ = torch.randn(2, 3, 224, 224)
        >>> out_ = model_(inp_)
        >>> out_.shape
        >>> torch.Size([2, 256])
        """

    def __init__(self, config: FeaturePyramidBackboneConfig):
        super().__init__()

        self._config: FeaturePyramidBackboneConfig = config

        # Initialize the backbone
        m = get_torchvision_backbone(config)
        self._body = m

        # Dry run to get number of channels for FPN
        inp = torch.randn(2, config.in_channels, 224, 224)
        with torch.no_grad():
            out = self._body(inp)
        in_channels_list = [o.shape[-3] for o in out.values()]

        # Get extra block
        extra_block = None
        if config.extra_block_type is not None:
            if config.extra_block_type == 'maxpool':
                extra_block = LastLevelMaxPool()
            else:
                raise ValueError(f'Extra-block type {config.extra_block_type} is not supported.')

        # Build FPN
        self._fpn = FeaturePyramidNetwork(
            in_channels_list,
            out_channels=config.out_channels,
            extra_blocks=extra_block
        )

        # Global pooling
        if self._config.global_pooling is not None:
            if config.global_pooling == "avg" or config.global_pooling == "mean":
                self._global_pooling = LambdaOpLayer(
                    lambda x: x.flatten(-2).mean(dim=-1)
                )
            elif config.global_pooling == "sum" or config.global_pooling == "add":
                self._global_pooling = LambdaOpLayer(
                    lambda x: x.flatten(-2).sum(dim=-1)
                )
            elif config.global_pooling == "max":
                self._global_pooling = LambdaOpLayer(
                    lambda x: x.flatten(-2).max(dim=-1)[0]
                )
            else:
                raise ValueError(f"Unsupported global_pooling {self.config.global_pooling}")
        else:
            self._global_pooling = torch.nn.Identity()

    @property
    def config(self) -> FeaturePyramidBackboneConfig:
        return self._config

    @property
    def body(self) -> torch.nn.Module:
        return self._body

    @property
    def global_pooling(self) -> torch.nn.Module:
        return self._global_pooling

    @property
    def fpn(self) -> torch.nn.Module:
        return self._fpn

    def forward(self,
                x: torch.Tensor,
                return_dict: bool = True,
                *args, **kwargs) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the FPN backbone model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            return_dict (bool): Whether to return a dictionary of feature maps or a concatenated tensor. Default is
                True.
            *args: Additional arguments for the forward pass.
            **kwargs: Additional keyword arguments for the forward pass.

        Returns:
            Union[torch.Tensor, Dict[str, torch.Tensor]]: Output tensor or dictionary of feature maps. Shapes depends on
                the chosen output layers, and on whether global pooling is enabled.

        Raises:
            ValueError: If the input tensor does not have the expected number of channels.
        """

        if x.shape[-3] != self.config.in_channels:
            raise ValueError(f"Expected input shape to be of shape ({self.config.in_channels}, H, W), but got "
                             f"({x.shape[-3]}, H, W)")

        # Apply the backbone
        x = self.body(x, *args, **kwargs)

        # Apply the FP on-top
        x = self.fpn(x)

        # Apply global pooling if needed
        x = OrderedDict({k: self.global_pooling(x[k]) for k in x})

        if not return_dict:
            x = StackLayer(dim=1)(*[x[k] for k in x])

        return x


class EnsembleBackboneConfig(BackboneConfig):
    """
    Configuration class for an ensemble of backbone models.

    This class defines the configuration for combining multiple backbone models into an ensemble, allowing for
    the aggregation of features from different backbones. Each backbone can be configured independently, and their
    outputs are processed together.

    Args:
        backbone_configs (List[BackboneConfig]): List of configurations for the individual backbones in the ensemble.
        out_channels (int): The number of output channels for each projection layer.
        in_channels (int): The number of input channels. Default is 3.
        global_pooling (str): The type of global pooling to apply to each backbone's output. Supported types are
            "avg" (average pooling), "sum" (summation pooling), and "max" (max pooling). Default is "avg".
        name (str): The name of the ensemble backbone. Default is "ensemble_backbone".

    Attributes:
        backbone_configs (List[BackboneConfig]): List of configurations for the individual backbones in the ensemble.
        out_channels (int): The number of output channels for each projection layer.
        global_pooling (str): The type of global pooling to apply.
        in_channels (int): The number of input channels.
    """
    def __init__(self,
                 backbone_configs: List[BackboneConfig],
                 out_channels: int,
                 in_channels: int = 3,
                 global_pooling: str = "avg",
                 name: str = "ensemble_backbone"):
        super().__init__(name)
        self.backbone_configs: List[BackboneConfig] = backbone_configs
        self.out_channels: int = out_channels
        self.global_pooling: str = global_pooling
        self.in_channels: int = in_channels


class EnsembleBackbone(torch.nn.Module):
    # noinspection PyUnresolvedReferences
    """
    Ensemble Backbone model. Combines multiple backbone models into an ensemble, allowing for the extraction and
    aggregation of their features. Each backbone can be configured independently, and their outputs are processed
    together, being projected to a common output space, and global pooling is applied to aggregate the feature maps.

    Args:
        config (EnsembleBackboneConfig): Configuration for the ensemble backbone model.

    Attributes:
        config (EnsembleBackboneConfig): Configuration for the ensemble backbone model.
        backbone_names (List[str]): Names of the individual backbones in the ensemble.
        backbones (torch.nn.Module): Parallelized module containing the individual backbones.
        projections (torch.nn.ModuleDict): Dictionary of projection layers for each backbone's output.
        global_pooling (torch.nn.Module): Global pooling layer applied to the output feature maps.

    Example usage:
        >>> backbone_config1_ = TorchVisionBackboneConfig(name="resnet50")
        >>> backbone_config2_ = TorchVisionBackboneConfig(name="resnet101")
        >>> ensemble_config_ = EnsembleBackboneConfig(
        >>>    backbone_configs=[backbone_config1, backbone_config2],
        >>>    out_channels=1024
        >>> )
        >>> model_ = EnsembleBackbone(ensemble_config_)
        >>> inp_ = torch.randn(2, 3, 224, 224)
        >>> output = model_(inp_)
        >>> output.shape
        >>> torch.Size([2, 1024])
    """
    def __init__(self, config: EnsembleBackboneConfig):
        super().__init__()
        self._config: EnsembleBackboneConfig = config

        # Initialize backbones
        backbone_names = []
        backbones = []
        for i, backbone_config in enumerate(config.backbone_configs):
            backbone_names.append(f"{backbone_config.name}_{i}")
            backbones.append(get_backbone(backbone_config))

        self._backbone_names: List[str] = backbone_names
        self._backbones = ParallelModules(*backbones)

        # Get backbone output shapes
        inp = torch.randn(2, config.in_channels, 224, 224)
        with torch.no_grad():
            out_shapes = [o.shape for o in self._backbones(inp)]

        # Initialize projections
        projections = torch.nn.ModuleDict()
        for i, out_shape in enumerate(out_shapes):
            channels = out_shape[-3] if len(out_shape) > 3 else out_shape[-1]
            proj = torch.nn.Linear(channels, config.out_channels)
            projections[backbone_names[i]] = proj
        self._projections = projections

        # Initialize global pooling
        if config.global_pooling == "avg" or config.global_pooling == "mean":
            self._global_pooling = LambdaOpLayer(
                lambda x: torch.mean(x, dim=-1) if len(x.shape) <= 3 else x.flatten(-2).mean(dim=-1)
            )
        elif config.global_pooling == "sum" or config.global_pooling == "add":
            self._global_pooling = LambdaOpLayer(
                lambda x: torch.sum(x, dim=-1) if len(x.shape) <= 3 else x.flatten(-2).sum(dim=-1)
            )
        elif config.global_pooling == "max":
            self._global_pooling = LambdaOpLayer(
                lambda x: torch.max(x, dim=-1)[0] if len(x.shape) <= 3 else x.flatten(--2).max(dim=-1)[0]
            )
        else:
            raise ValueError(f"Unsupported global_pooling {self.config.global_pooling}")

    @property
    def config(self) -> EnsembleBackboneConfig:
        return self._config

    @property
    def backbone_names(self) -> List[str]:
        return self._backbone_names

    @property
    def global_pooling(self) -> torch.nn.Module:
        return self._global_pooling

    @property
    def backbones(self) -> torch.nn.Module:
        return self._backbones

    @property
    def projections(self) -> torch.nn.ModuleDict:
        return self._projections

    def forward(self,
                x: torch.Tensor,
                return_dict: bool = True,
                *args, **kwargs) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the ensemble backbone model. Processes the input tensor through each backbone in the
        ensemble, applies global pooling to the outputs, projects the pooled outputs to a common output space, and
        optionally combines them into a single tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            return_dict (bool): Whether to return a dictionary of feature maps or a concatenated tensor. Default is
                True.

        Returns:
            Union[torch.Tensor, Dict[str, torch.Tensor]]: Output tensor or dictionary of feature maps.

        Raises:
            ValueError: If the input tensor does not have the expected number of channels.
        """
        if x.shape[-3] != self.config.in_channels:
            raise ValueError(f"Expected input shape to be of shape ({self.config.in_channels}, H, W), but got "
                             f"({x.shape[-3]}, H, W)")

        # Get backbone outputs
        out = OrderedDict({self.backbone_names[i]: o for i, o in enumerate(self.backbones(x, *args, **kwargs))})

        # For each output
        for name, o in out.items():
            # Apply global pooling
            out[name] = self.global_pooling(o)

            # Apply projection
            out[name] = self.projections[name](out[name])

        # Stack outputs if required
        if not return_dict:
            out = StackLayer(dim=1)(*[out[k] for k in out])

        return out


def get_backbone(backbone_config: BackboneConfig) -> Union[torch.nn.Module, FPNBackbone, EnsembleBackbone]:
    if isinstance(backbone_config, FeaturePyramidBackboneConfig):
        return FPNBackbone(backbone_config)
    elif isinstance(backbone_config, TorchVisionBackboneConfig):
        return get_torchvision_backbone(backbone_config)
    elif isinstance(backbone_config, EnsembleBackboneConfig):
        return EnsembleBackbone(backbone_config)
    else:
        raise ValueError(f'Backbone config of type {backbone_config.__class__.__name__} is not supported.')
