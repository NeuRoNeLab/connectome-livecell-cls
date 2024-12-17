from typing import Optional, Dict, Any, NamedTuple, List
import torch
from utils.utils import SerializableConfig
from models.layers.misc import SerializableModule
from models.vision.vit import CustomViT, CustomViTConfig, CustomViTOutput
from models.vision.backbone_utils import TorchVisionBackboneConfig, get_backbone


class BackboneViTConfig(SerializableConfig):

    def __init__(self, backbone_config: TorchVisionBackboneConfig, latent_vit_config: CustomViTConfig):
        super().__init__()
        self.backbone_config: TorchVisionBackboneConfig = backbone_config
        self.latent_vit_config: CustomViTConfig = latent_vit_config


class BackboneViTOutput(NamedTuple):
    vit_out: CustomViTOutput
    feature_map: torch.Tensor = None

    @property
    def output(self) -> torch.Tensor:
        return self.vit_out.output

    @property
    def cls_embedding(self) -> torch.Tensor:
        return self.vit_out.cls_embedding

    @property
    def avg_embedding(self) -> torch.Tensor:
        return self.vit_out.avg_embedding

    @property
    def attn_weights(self) -> List[Optional[torch.Tensor]]:
        return self.vit_out.attn_weights

    @property
    def final_embeddings(self) -> torch.Tensor:
        return self.vit_out.final_embeddings

    @property
    def moe_loss(self) -> Optional[torch.Tensor]:
        return self.vit_out.moe_loss


class BackboneViT(SerializableModule):

    def __init__(self, config: BackboneViTConfig):
        super().__init__()
        self._config: BackboneViTConfig = config
        self._backbone = get_backbone(backbone_config=config.backbone_config)
        self._vit_head: CustomViT = CustomViT(config.latent_vit_config)

    @property
    def config(self) -> BackboneViTConfig:
        return self._config

    @property
    def backbone(self) -> torch.nn.Module:
        return self._backbone

    @property
    def vit_head(self) -> CustomViT:
        return self._vit_head

    def serialize_constructor_params(self, *args, **kwargs) -> dict:
        return {"config": self.config}

    def forward(self,
                x: torch.Tensor,
                attention_kwargs: Optional[Dict[str, Any]] = None,
                *args, **kwargs) -> BackboneViTOutput:
        feature_map = self.backbone(x)
        vit_out = self.vit_head(feature_map, attention_kwargs=attention_kwargs)

        return BackboneViTOutput(vit_out=vit_out, feature_map=feature_map)
