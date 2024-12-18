import torch
from lightning import Trainer  # noqa: F401
from lightning.pytorch.cli import LightningCLI
from models.elegans_vision.elegans_vision_lit_wrapper import VisionElegansWrapper


def cli_main():
    torch.set_float32_matmul_precision('medium')
    LightningCLI(model_class=VisionElegansWrapper,
                 save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    cli_main()
