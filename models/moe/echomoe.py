from typing import Type, Tuple
import einops
import torch.nn
from echotorch.nn.Node import Node
from models.esn.esn_custom_cell import EchoStateNet
from models.moe.moe import MoE, MLP


class EchoStateMoE(EchoStateNet):

    def __init__(self,
                 input_dim: int,
                 reservoir_dim: int,
                 readout_dim: int,
                 input_scaling: float = 1.0,
                 nonlin_func=torch.tanh,
                 washout: int = 0,
                 connectivity: float = 0.5,
                 spectral_radius: float = 0.9,
                 bias_scaling: float = 1.0,
                 num_experts: int = 8,
                 k: int = 2,
                 noisy_gating: bool = True,
                 expert_class: Type[torch.nn.Module] = MLP,
                 expert_kwargs: dict = None,
                 noise_generator=None,
                 dtype=torch.float32):
        super().__init__(
            input_dim=input_dim,
            reservoir_dim=reservoir_dim,
            readout_dim=readout_dim,
            input_scaling=input_scaling,
            nonlin_func=nonlin_func,
            washout=washout,
            noise_generator=noise_generator,
            connectivity=connectivity,
            spectral_radius=spectral_radius,
            bias_scaling=bias_scaling,
            dtype=dtype
        )

        expert_kwargs = expert_kwargs if expert_kwargs is not None else {}
        if len(expert_kwargs) == 0 and expert_class == MLP:
            expert_kwargs = {
                "input_size": reservoir_dim,
                "hidden_size": 4 * reservoir_dim,
                "output_size": readout_dim
            }
        self.readout = MoE(
            input_size=reservoir_dim,
            output_size=readout_dim,
            num_experts=num_experts,
            noisy_gating=noisy_gating,
            k=k,
            expert_class=expert_class,
            expert_kwargs=expert_kwargs
        )

    def forward(self, x: torch.Tensor, moe_loss_coef: float = 1e-2, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input has shape (B, seq_len, Emb)

        # Apply ESN Cell
        out = self.esn_cell(x)

        # Apply MoE MLP layer as readout
        out = einops.rearrange(out, "b s e -> (b s) e", s=x.shape[1])
        out, moe_loss = self.readout(out, loss_coef=moe_loss_coef)
        out = einops.rearrange(out, "(b s) e -> b s e", s=x.shape[1])

        return out, moe_loss



