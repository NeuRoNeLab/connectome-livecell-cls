import functools
import os
import argparse
import typing
import enum
import numpy as np
import yaml
from torchvision.transforms import ToPILImage
from tqdm import tqdm
import torch
from PIL import Image
from ds.livecell_datamodule import LivecellDataModule
from ds.livecell_dataset import CellDataset
from explainability.gradcam_utils import METHODS, apply_cam, resolve_target_layers, reshape_transform_vit, \
    get_img_paths, reshape_transform_general
from explainability.attentionmap import attention_rollout
from models.elegans_vision.elegans_convnext import ElegansConvNextConfig
from models.elegans_vision.elegans_convnext_lit_wrapper import ElegansConvNextWrapper
from models.elegans_vision.elegans_vision import VisionElegansConfig
from models.elegans_vision.elegans_vision_lit_wrapper import VisionElegansWrapper
from models.elegansformer.elegans_swin import ElegansSwinConfig
from models.elegansformer.elegans_swin_lit_wrapper import ElegansSwinWrapper
from models.elegansformer.elegansformer import ElegansFormerConfig
from models.elegansformer.elegansformer_lit_wrapper import ElegansFormerWrapper
from models.elegansmixer.elegansmlpmixer import ElegansMLPMixerConfig
from models.elegansmixer.elegansmlpmixer_lit_wrapper import ElegansMLPMixerWrapper
from models.fpelegans.fpelegans import FPElegansConfig
from models.fpelegans.fpelegans_lit_wrapper import FPElegansWrapper
from models.vision.backbone_vit import BackboneViTConfig, BackboneViT
from models.vision.mlpmixer import MLPMixerConfig
from models.vision.mlpmixer_lit_wrapper import MLPMixerWrapper
from models.vision.swin_v2 import CustomSwinConfig, SwinTransformerV2
from models.vision.swin_v2_lit_wrapper import CustomSwinWrapper
from models.vision.vit import CustomViTConfig
from models.vision.vit_lit_wrapper import CustomViTWrapper
from models.vision.backbone import TorchVisionBackboneConfig
from models.vision.convnext import CustomConvNextConfig
from models.vision.convnext_lit_wrapper import CustomConvNextWrapper

from models.vision.backbone_lit_wrapper import BackboneWrapper
from explainability.lime_utils import lime_explain
from lime import lime_image

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    model_type = args.model_type
    ckpt_path = args.ckpt_path
    target_layer_names = args.target_layer_names
    model_name = args.model_name
    config_path = args.config

    # Create the model
    with open(config_path, 'r') as fp:
        config = yaml.safe_load(fp)
    config_args = config['model']['config']['init_args']

    # Get the main config class
    config_classname = config['model']['config']['class_path'].split(".")[-1]
    config_module_name = ".".join(config['model']['config']['class_path'].split(".")[:-1])
    config_mod = __import__(config_module_name, fromlist=[config_classname])
    config_class = getattr(config_mod, config_classname)
    conf = config_class(**config['model']['config']['init_args'])

    # Get the backbone config class and updating the config class
    if args.model_type == 'visionelegans':
        backbone_classname = config['model']['config']['init_args']['backbone_config']['class_path'].split(".")[-1]
        backbone_module_name = ".".join(
            config['model']['config']['init_args']['backbone_config']['class_path'].split(".")[:-1])
        config_backbone_mod = __import__(backbone_module_name, fromlist=[backbone_classname])
        config_backbone_class = getattr(config_backbone_mod, backbone_classname)
        backbone_conf = config_backbone_class(**config['model']['config']['init_args']['backbone_config']['init_args'])
        conf.backbone_config = backbone_conf

    if args.model_type == "vit":
        type_hints = typing.get_type_hints(CustomViTConfig.__init__)
        config_args = {
            k: config_args[k] if not (isinstance(type_hints[k], type) and issubclass(type_hints[k], enum.Enum))
            else getattr(type_hints[k], config_args[k])
            for k in config_args
        }
        model = CustomViTWrapper(CustomViTConfig(**config_args))
    elif args.model_type == "swin":
        type_hints = typing.get_type_hints(CustomSwinConfig.__init__)
        config_args = {
            k: config_args[k] if not (isinstance(type_hints[k], type) and issubclass(type_hints[k], enum.Enum))
            else getattr(type_hints[k], config_args[k])
            for k in config_args
        }
        model = CustomSwinWrapper(CustomSwinConfig(**config_args))
    elif args.model_type == "elegans_swin":
        type_hints = typing.get_type_hints(ElegansSwinConfig.__init__)
        config_args = {
            k: config_args[k] if not (isinstance(type_hints[k], type) and issubclass(type_hints[k], enum.Enum))
            else getattr(type_hints[k], config_args[k])
            for k in config_args
        }
        model = ElegansSwinWrapper(ElegansSwinConfig(**config_args))
    elif args.model_type == "visionbackbone":
        type_hints = typing.get_type_hints(TorchVisionBackboneConfig.__init__)
        config_args = {
            k: config_args[k] if not (isinstance(type_hints[k], type) and issubclass(type_hints[k], enum.Enum))
            else getattr(type_hints[k], config_args[k])
            for k in config_args
        }
        model = BackboneWrapper(TorchVisionBackboneConfig(**config_args))
    elif args.model_type == "convnext":
        type_hints = typing.get_type_hints(CustomConvNextConfig.__init__)
        config_args = {
            k: config_args[k] if not (isinstance(type_hints[k], type) and issubclass(type_hints[k], enum.Enum))
            else getattr(type_hints[k], config_args[k])
            for k in config_args
        }
        model = CustomConvNextWrapper(CustomConvNextConfig(**config_args))
    elif args.model_type == "elegansformer":
        type_hints = typing.get_type_hints(ElegansFormerConfig.__init__)
        config_args = {
            k: config_args[k] if not (isinstance(type_hints[k], type) and issubclass(type_hints[k], enum.Enum))
            else getattr(type_hints[k], config_args[k])
            for k in config_args
        }
        model = ElegansFormerWrapper(ElegansFormerConfig(**config_args))
    elif args.model_type == "visionelegans":
        type_hints = typing.get_type_hints(VisionElegansConfig.__init__)
        conf = {
            k: conf[k] if not (isinstance(type_hints[k], type) and issubclass(type_hints[k], enum.Enum))
            else getattr(type_hints[k], conf[k])
            for k in conf
        }
        model = VisionElegansWrapper(VisionElegansConfig(**conf))
    elif args.model_type == "elegans_convnext":
        type_hints = typing.get_type_hints(ElegansConvNextConfig.__init__)
        conf = {
            k: conf[k] if not (isinstance(type_hints[k], type) and issubclass(type_hints[k], enum.Enum))
            else getattr(type_hints[k], conf[k])
            for k in conf
        }
        model = ElegansConvNextWrapper(ElegansConvNextConfig(**conf))
    elif args.model_type == "mlpmixer":
        type_hints = typing.get_type_hints(MLPMixerConfig.__init__)
        config_args = {
            k: config_args[k] if not (isinstance(type_hints[k], type) and issubclass(type_hints[k], enum.Enum))
            else getattr(type_hints[k], config_args[k])
            for k in config_args
        }
        model = MLPMixerWrapper(MLPMixerConfig(**config_args))
    elif args.model_type =="elegansmlpmixer":
        type_hints = typing.get_type_hints(ElegansMLPMixerConfig.__init__)
        config_args = {
            k: config_args[k] if not (isinstance(type_hints[k], type) and issubclass(type_hints[k], enum.Enum))
            else getattr(type_hints[k], config_args[k])
            for k in config_args
        }
        model = ElegansMLPMixerWrapper(ElegansMLPMixerConfig(**config_args))
    elif args.model_type == "backbonevit":
        type_hints = typing.get_type_hints(BackboneViTConfig.__init__)
        config_args = {
            k: config_args[k] if not (isinstance(type_hints[k], type) and issubclass(type_hints[k], enum.Enum))
            else getattr(type_hints[k], config_args[k])
            for k in config_args
        }
        backbone_confg = TorchVisionBackboneConfig(**config_args['backbone_config']['init_args'])
        latent_vit_confg = CustomViTConfig(**config_args['latent_vit_config']['init_args'])
        model = BackboneViT(BackboneViTConfig(backbone_confg, latent_vit_confg))
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    if "attentionmap" in args.methods and model_type not in ["vit", "elegansformer", "backbonevit", "swin", "elegans_swin"]:
        raise ValueError(f"Unsupported model type {args.model_type} for method {args.method}")

    # Load state dict
    ckpt = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model = model.to(args.device)
    # print(model._vit.stages[0])
    # exit()
    print("Model loaded.")

    # Create the processing pipeline
    classname = config['data']['class_path'].split(".")[-1]
    module_name = ".".join(config['data']['class_path'].split(".")[:-1])
    mod = __import__(module_name, fromlist=[classname])
    datamodule_class = getattr(mod, classname)

    if not issubclass(datamodule_class, LivecellDataModule):
        raise ValueError(f"Unsupported data module: {datamodule_class}")

    datamodule: LivecellDataModule = datamodule_class(**config['data']['init_args'])
    datamodule.setup('test')
    ds: CellDataset = datamodule.test_dataloader().dataset
    transform = ds.preprocess_image
    reshape_transform = functools.partial(
        reshape_transform_vit,
        height=args.reshape_height_vit,
        width=args.reshape_width_vit,
    ) if model_type in ["vit", "elegansformer", "mlpmixer", "elegansmlpmixer", "swin", "elegans_swin"] else functools.partial(reshape_transform_general)

    img_paths = get_img_paths(args.img_paths)[0]

    for img_path in tqdm(img_paths, desc="Applying explainability on images..."):
        # Get the image name
        img_name = os.path.basename(img_path)
        label = ds.get_label(img_name)

        # Read the image and convert it to a torch tensor and a numpy array
        image = Image.open(img_path).convert('L')  # avoiding color bias
        img_tensor = transform(image).to(args.device).unsqueeze(0)

        # img_array_rgb_scaled = np.asarray(img_tensor.detach().cpu().numpy(),
        #                                   dtype=np.float32)[..., None].repeat(3, 1)
        image_rgb_pil = ToPILImage()(img_tensor.squeeze(0)).convert("RGB")
        img_array_rgb_scaled = np.asarray(image_rgb_pil, dtype=np.float32) / 255.0

        # Get the model prediction
        output_key = "output"
        out = model(img_tensor)
        if output_key is not None and hasattr(out, output_key):
            logits = getattr(out, output_key)
        elif output_key is not None and hasattr(out, "__getitem__"):
            logits = out[output_key]
        else:
            logits = out
        prediction = logits.argmax(dim=-1).squeeze().item()

        for method in args.methods:
            target_path = str(os.path.join(args.out_dir, model_type, model_name, img_name))
            os.makedirs(target_path, exist_ok=True)

            if method == 'attentionmap':
                attention_rollout(
                    attn_weights=out.attn_weights,
                    img=image_rgb_pil,
                    label=label,
                    logits=logits,
                    return_masked_img=args.return_masked_img,
                    plot_img=args.plot_imgs,
                    figsize=(16, 16),
                    out_path=os.path.join(target_path, f'lab_{label}_pred_{prediction}_attentionmap.png'),
                )
            elif method == 'lime':
                lime_explain(
                    model=model,
                    img=image_rgb_pil.convert('L'),
                    explainer=lime_image.LimeImageExplainer(),
                    out_path=os.path.join(target_path, f'lab_{label}_pred_{prediction}_lime.png'),
                    device=args.device
                )
            else:
                apply_cam(
                    cam_used=METHODS[method],
                    cam_name=method,
                    model=model,
                    input_tensor=img_tensor,
                    scaled_rgb_img=img_array_rgb_scaled,
                    target_layers=[] if method == 'fullgradcam' else resolve_target_layers(model=model, target_layers_ids=target_layer_names),
                    prediction=prediction,
                    out_dir=target_path,
                    reshape_transform=functools.partial(reshape_transform, method=method)
                                      if model_type in ["vit", "elegansformer", "mlpmixer", "elegansmlpmixer", "swin", "elegans_swin"] else None,
                    eigen_smooth=args.eigen_smooth if method != "hirescam" else False,
                    aug_smooth=args.aug_smooth if method != "hirescam" else False,
                    category=label,
                    use_rgb=True,
                    output_key="output" if model_type in ["backbonevit"] else None,
                    plot_imgs=args.plot_imgs,
                    image_weight=args.image_weight
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config.yaml", help="lightning CLI yaml config file")
    parser.add_argument("--model_type", type=str, default="vit", help="model type",
                        choices=["vit", "visionbackbone", "elegansformer", "visionelegans", "mlpmixer", "elegansmlpmixer",
                                 "backbonevit", "convnext", "elegans_convnext", "swin", "elegans_swin"])
    parser.add_argument("--model_name", type=str, default="model", help="model name")
    parser.add_argument("--ckpt_path", type=str, default=None, help="path to checkpoint file")
    parser.add_argument("--target_layer_names", type=str, default=None, help="target layer names",
                        nargs="+")
    # parser.add_argument("--target_layer_idx", type=int, default=None, help="target layer index", nargs="+")
    parser.add_argument("--methods", type=str, default="gradcam", help="the explainability technique",
                        choices=[*list(METHODS.keys()), "attentionmap", "lime"], nargs="+")
    parser.add_argument("--eigen_smooth", action=argparse.BooleanOptionalAction, default=True,
                        help="whether to use eigen smoothed method", dest="eigen_smooth")
    parser.add_argument("--aug_smooth", action=argparse.BooleanOptionalAction, default=True,
                        help="whether to use aug smoothed method", dest="aug_smooth")
    parser.add_argument("--reshape_height_vit", type=int, default=16,
                        help="the height of the latent image to use with ViT-like models in GradCAM-based methods")
    parser.add_argument("--reshape_width_vit", type=int, default=16,
                        help="the width of the latent image to use with ViT-like models in GradCAM-based methods")
    parser.add_argument("--plot_imgs", action=argparse.BooleanOptionalAction, default=False,
                        help="whether to plot the images", dest="plot_imgs")
    parser.add_argument("--return_masked_img", type=str, default="all",
                        help="attention-masked images to return", choices=["all", "last", "mean"])
    parser.add_argument("--img_paths", type=str, default="example.png", help="image paths", nargs="+")
    parser.add_argument("--out_dir", type=str, default="_explainability", help="output dir path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="device to use")
    parser.add_argument("--image_weight", type=float, default=0.5, help="weight for overlay map")
    main(parser.parse_args())


