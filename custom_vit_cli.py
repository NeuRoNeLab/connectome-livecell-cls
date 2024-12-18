import torch
from lightning import Trainer  # noqa: F401
from lightning.pytorch.cli import LightningCLI
from models.vision.vit_lit_wrapper import CustomViTWrapper


def cli_main():
    torch.set_float32_matmul_precision('medium')
    LightningCLI(model_class=CustomViTWrapper,
                 # datamodule_class=M1DataModule,
                 save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    cli_main()
