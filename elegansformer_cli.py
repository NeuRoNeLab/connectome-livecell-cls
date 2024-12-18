import torch
from lightning import Trainer  # noqa: F401
from lightning.pytorch.cli import LightningCLI
from models.elegansformer.elegansformer_lit_wrapper import ElegansFormerWrapper


def cli_main():
    torch.set_float32_matmul_precision('medium')
    LightningCLI(model_class=ElegansFormerWrapper,
                 save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    cli_main()
