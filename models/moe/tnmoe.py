from typing import Dict, Any, Type, Tuple

import einops
import torch

from models.moe.moe import MoE
from models.tensornet import TensorNetwork, FunctionalTensorNetwork


class TensorNetworkMoE(MoE):

    def __init__(self,
                 tensornet_config: Dict[str, Any],
                 num_experts: int,
                 noisy_gating: bool = True,
                 k: int = 2,
                 tensornet_class: Type[TensorNetwork] = FunctionalTensorNetwork):
        
        if "input_sequences" in tensornet_config:
            tensornet_config["input_sequences"] = None  # no input sequences

        self._tensornet_config = tensornet_config
        super().__init__(
            input_size=tensornet_config["input_dim"],
            output_size=tensornet_config["embedding_dim"],
            num_experts=num_experts,
            noisy_gating=noisy_gating,
            k=k,
            expert_class=tensornet_class,
            expert_kwargs=tensornet_config
        )

    @property
    def tensornet_config(self) -> Dict[str, Any]:
        return self._tensornet_config

    def __getattr__(self, item):
        if item in self._tensornet_config:
            return self.tensornet_config[item]
        return super().__getattr__(item)
    
    def forward(self, x: torch.Tensor, loss_coef: float = 1e-2, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # Shape (B, N, E)
        shape_orig = x.shape
        
        # Reshape to (B*N, E)
        x = einops.rearrange(x, 'b s e -> (b s) e')
        
        # Apply MoE
        out, loss = super().forward(x=x, loss_coef=loss_coef, flatten_batch=False, **kwargs)
        
        # Reshape back to (B, N, E)
        out = einops.rearrange(out, '(b s) e -> b s e', s=shape_orig[1])
        return out, loss
        
