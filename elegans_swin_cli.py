import torch
from lightning import Trainer  # noqa: F401
from lightning.pytorch.cli import LightningCLI
from models.elegansformer.elegans_swin_lit_wrapper import ElegansSwinWrapper

def cli_main():
    torch.set_float32_matmul_precision('medium')
    LightningCLI(
        model_class=ElegansSwinWrapper,
        save_config_kwargs={"overwrite": True}
    )


if __name__ == "__main__":
    cli_main()
