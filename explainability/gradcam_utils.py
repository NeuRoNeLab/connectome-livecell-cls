import os
from typing import Type, Final, Union, Tuple, List, Dict, Optional, Callable, Any
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image

from explainability.gradcam import GradCAMV2, XGradCAMV2, GradCAMPlusPlusV2, GradCAMElementWiseV2, EigenGradCAMV2, \
    EigenCAMV2, FullGradV2, HiResCAMV2, LayerCAMV2, ScoreCAMV2, RandomCAMV2, AblationCAMV2, BaseCAMV2
import glob



METHODS: Final[Dict[str, Type[BaseCAMV2]]] = {
    "gradcam": GradCAMV2,
    "xgradcam": XGradCAMV2,
    "gradcampp": GradCAMPlusPlusV2,
    "gradcamelementwise": GradCAMElementWiseV2,
    "eigengradcam": EigenGradCAMV2,
    "eigencam": EigenCAMV2,
    "fullgradcam": FullGradV2,
    "hirescam": HiResCAMV2,
    "layercam": LayerCAMV2,
    "scorecam": ScoreCAMV2,
    "randomcam": RandomCAMV2,
    "ablationcam": AblationCAMV2
}


def get_img_paths(img_paths):
    paths = []
    if isinstance(img_paths, list):
        for img_path in img_paths:
            if os.path.isdir(img_path):
                paths.append(glob.glob(img_path + '/*/*.png'))
            else:
                paths.append(img_path)
        return paths
    else:
        return img_paths





def reshape_transform_vit(tensor: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor], Dict[str, torch.Tensor]],
                          height: int = 16,
                          width: int = 16,
                          key: Union[int, str] = 0,
                          method: str = 'gradcam'):
    if isinstance(tensor, (tuple, list, dict)):
        tensor = tensor[key]
    t_size = tensor.size()
    if method == 'fullgradcam':
        return tensor
    else:
        result = tensor[:, 1:, :].reshape(t_size[0], height, width, t_size[-1])  # remove cls token

    # Bring the channels to the first dimension, like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def reshape_transform_general(tensor: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor], Dict[str, torch.Tensor]],
                          key: Union[int, str] = 0,
                          method: str = 'gradcam'):
    if isinstance(tensor, (tuple, list, dict)):
        tensor = tensor[key]
    return tensor


def _parse_model_layer(target_layer_name: str, model_name: str = 'model') -> str:
    parts = target_layer_name.split('.')
    converted_parts = []

    for part in parts:
        if part.isdigit():
            converted_parts.append(f'[{int(part)}]')
        else:
            converted_parts.append('.' + part)

    out = f'{model_name}.' + ''.join(converted_parts)[1:]
    return out


def _get_layers_list(model: torch.nn.Module) -> List[str]:
    layers_of_model = []
    # Iterate over named modules
    for i, (name, module) in enumerate(model.named_modules()):
        # print(str(i), name)
        layers_of_model.append(name)
    return layers_of_model

#If you can't pass the full module, use this function to filter layers in the 'resolve_target_layers' function
def filtra_stringhe(lista):
    # Filtra stringhe con almeno 6 punti e contenenti '_backbone.0'
    return [s for s in lista if s.count('.') >= 3 and '_backbone.0' in s and 'conv' in s]

def resolve_target_layers(model: torch.nn.Module, target_layers_ids: List[Union[str, int]]) -> List[torch.nn.Module]:
    layer_list = _get_layers_list(model)
    target_layers = []
    # for l in layer_list:
    #     print(l)
    # target_layers_ids = filtra_stringhe(layer_list)
    for target_layer_id in target_layers_ids:
        if isinstance(target_layer_id, int):
            target_layer_id = layer_list[target_layer_id]
        target_layers.append(model.get_submodule(target_layer_id))
    return target_layers


class ClassifierOutputSoftmaxTargetV2(torch.nn.Module):
    def __init__(self, category: Optional[int] = None, output_key: Optional[Union[str, int]] = None):
        super().__init__()
        self.category: Optional[int] = category
        self.output_key: Optional[str] = output_key

    def forward(self, model_output):
        if self.output_key is not None and hasattr(model_output, self.output_key):
            model_output = getattr(model_output, self.output_key)
        elif self.output_key is not None and hasattr(model_output, "__getitem__"):
            model_output = model_output[self.output_key]

        category = self.category if self.category is not None else model_output.argmax()
        if len(model_output.shape) == 1:
            return torch.softmax(model_output, dim=-1)[category]
        return torch.softmax(model_output, dim=-1)[:, category]


def apply_cam(cam_used: Type[BaseCAMV2],
              cam_name: str,
              model: torch.nn.Module,
              input_tensor: torch.Tensor,
              scaled_rgb_img: np.ndarray,
              target_layers: List[torch.nn.Module],
              prediction: int,
              out_dir: Optional[str] = None,
              reshape_transform: Optional[Callable[[Any], torch.Tensor]] = None,
              eigen_smooth: bool = False,
              aug_smooth: bool = False,
              category: Optional[int] = None,
              use_rgb: bool = True,
              output_key: Optional[Union[str, int]] = None,
              plot_imgs: bool = False,
              image_weight: float = 0.5):
    with cam_used(model=model, target_layers=target_layers, reshape_transform=reshape_transform) as cam:
        if cam_name in ['fullgradcam', 'gradcam'] and out_dir is not None:
            # After a crash, skim images done before
            filenames = os.listdir(out_dir)
            if any(cam_name in filename for filename in filenames):
                print(f"Esiste già l'immagine {cam_name}")
                return


        targets = [ClassifierOutputSoftmaxTargetV2(category, output_key=output_key)]
        grayscale_cams = cam(
            input_tensor=input_tensor,
            targets=targets,
            eigen_smooth=eigen_smooth,
            aug_smooth=aug_smooth
        )

        cam_image = show_cam_on_image(scaled_rgb_img, grayscale_cams[0, :], use_rgb=use_rgb, image_weight=image_weight)

        img1 = Image.fromarray(cam_image)
        img2 = Image.fromarray(np.uint8(255 * scaled_rgb_img))

        if out_dir is not None:
            p1 = os.path.join(out_dir, f'lab_{category if category is not None else ""}_pred_{prediction}'
                                       f'_{cam_name}.png')
            p2 = os.path.join(out_dir, f'lab_{category if category is not None else ""}_pred_{prediction}_input.png')
            img1.save(p1)
            img2.save(p2)

        if plot_imgs:
            img1.show()
            img2.show()

