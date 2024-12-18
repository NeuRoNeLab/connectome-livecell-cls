import enum
import json
import os
import argparse
import typing
import yaml
from tqdm import tqdm
import torch
from ds.livecell_datamodule import LivecellDataModule
from explainability.covariance import compute_cov, plot_spectral_values
from models.elegans_vision.elegans_vision import VisionElegansConfig
from models.elegans_vision.elegans_vision_lit_wrapper import VisionElegansWrapper
from models.elegansformer.elegansformer import ElegansFormerConfig
from models.elegansformer.elegansformer_lit_wrapper import ElegansFormerWrapper
from models.vision.mlpmixer import MLPMixerConfig
from models.vision.mlpmixer_lit_wrapper import MLPMixerWrapper
from models.vision.vit import CustomViTConfig
from models.vision.vit_lit_wrapper import CustomViTWrapper


def main(args):
    model_name = args.model_name
    out_dir = str(os.path.join(args.out_dir, model_name))
    os.makedirs(out_dir, exist_ok=True)

    model_type = args.model_type
    ckpt_path = args.ckpt_path
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
    elif args.model_type == "mlpmixer":
        type_hints = typing.get_type_hints(MLPMixerConfig.__init__)
        config_args = {
            k: config_args[k] if not (isinstance(type_hints[k], type) and issubclass(type_hints[k], enum.Enum))
            else getattr(type_hints[k], config_args[k])
            for k in config_args
        }
        model = MLPMixerWrapper(MLPMixerConfig(**config_args))
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    # Load state dict
    ckpt = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(args.device)
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
    dl = datamodule.test_dataloader()
    running_avg_cov = None

    for i, batch in tqdm(enumerate(dl), total=len(dl), desc="Getting covariances for the embedding space"):

        if i < args.max_batches:
            x, y = batch
            out = model(x)
            if model_type in ["vit", "elegansformer", "mlpmixer"]:
                embedding = out.cls_embedding if model.config.use_cls_pool else out.avg_embedding
            elif model_type == "visionelegans":
                embedding = out.embedding
            else:
                raise ValueError(f"Unsupported model type: {args.model_type}")
            cov = compute_cov(embedding, batched=False)

            if running_avg_cov is None:
                running_avg_cov = cov
            else:
                running_avg_cov = running_avg_cov + (cov - running_avg_cov) / (i + 1)

    method = args.method
    value_type = 'eigenvalues' if method in ['eig', 'eigh', 'eigvals', 'eigvalsh'] else 'singular values'

    print(f"Computing {value_type}...")
    values = plot_spectral_values(
        tensor=running_avg_cov,
        method=method,
        absolute=args.absolute,
        path=os.path.join(out_dir, f"{method}_{value_type}_bar.svg"),
        plot_type="bar",
        show_plot=args.show_plot
    )
    plot_spectral_values(
        tensor=running_avg_cov,
        method=method,
        absolute=args.absolute,
        path=os.path.join(out_dir, f"{method}_{value_type}_hist.svg"),
        plot_type="hist",
        show_plot=args.show_plot
    )
    plot_spectral_values(
        tensor=running_avg_cov,
        method=method,
        absolute=args.absolute,
        path=os.path.join(out_dir, f"{method}_{value_type}_box.svg"),
        plot_type="box",
        show_plot=args.show_plot
    )
    print(f"Computing of {value_type} completed.")

    print(f"Computing {value_type} statistics...")
    top_k_vals, top_k_idxs = values.topk(k=args.topk, sorted=True)
    top_k_vals = top_k_vals.tolist()[1:]  # remove the max
    top_k_idxs = top_k_idxs.tolist()[1:]  # remove the max
    stats_dict = {
        "mean": values.mean().item(),
        "std": values.std().item(),
        "min": values.min().item(),
        "min_idx": values.argmin().item(),
        **{f"quantile{q}": values.quantile(q / 100.0).item() for q in range(5, 100, 5)},
        "max": values.max().item(),
        "max_idx": values.argmax().item(),
        "l2norm": values.norm(p=2).item(),
        "l1norm": values.norm(p=1).item(),
        **{f"top_{k + 2}":  top_k_val for k, top_k_val in enumerate(top_k_vals)},
        **{f"top_{k + 2}_idx": top_k_idx for k, top_k_idx in enumerate(top_k_idxs)}
    }

    with open(os.path.join(out_dir, f"{method}_{value_type}_stats.json"), "w") as fp:
        json.dump(stats_dict, fp, indent=4)
    print(f"Computation of {value_type} statistics completed.")
    print(stats_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config.yaml", help="lightning CLI yaml config file")
    parser.add_argument("--model_type", type=str, default="vit", help="model type",
                        choices=["vit", "visionbackbone", "elegansformer", "visionelegans", "mlpmixer"])
    parser.add_argument("--model_name", type=str, default="model", help="model name")
    parser.add_argument("--ckpt_path", type=str, default=None, help="path to checkpoint file")
    parser.add_argument("--method", type=str, default="eigh",
                        help="the technique used to compute spectral values",
                        choices=["svd", "eig", "eigh", "eigvals", "eigvalsh"])
    parser.add_argument("--out_dir", type=str, default="_explainability/covariance",
                        help="output dir path")
    parser.add_argument("--show_plot", action=argparse.BooleanOptionalAction, default=True,
                        dest="show_plot", help="whether to show plot")
    parser.add_argument("--absolute", action=argparse.BooleanOptionalAction, default=True, dest="absolute",
                        help="whether to get the absolute of the spectral values (module in the complex case)")
    parser.add_argument("--topk", type=int, default=10, dest="topk",
                        help="the top-k spectral values to save in the statistics")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="device to use")
    parser.add_argument("--max_batches", type=int, default=-1, help="maximum number of batches to test")
    main(parser.parse_args())
