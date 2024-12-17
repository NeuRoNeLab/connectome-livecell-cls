import torch
import os
import functools
import torchvision.transforms
from torch.utils.data import Dataset
from glob import glob
from PIL import Image


# Temp change just for windows
def maybe_to_rgb(x: torch.Tensor, to_rgb: bool = False) -> torch.Tensor:
    return x.repeat(3, 1, 1) if to_rgb and x.shape[-3] == 1 else x


class CellDataset(torch.utils.data.Dataset):
    def __init__(self, root_path: str, mode: str = 'train', task: str = 'mask', resize_to: tuple = (224, 224),
                 to_rgb: bool = False):
        self.root_path = root_path
        assert mode.lower() in ['train', 'val', 'valid', 'test']
        self.mode = mode
        self.task = task

        # self.images_dir = os.path.join(self.root_path, mode)
        self.images_paths = sorted(glob(self.root_path + f'/{self.mode}/*/*.png'))
        print(len(self.images_paths))
        # print(self.images_paths)
        self.preprocess_only_train = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                # lambda x: x.repeat(3, 1, 1) if to_rgb and x.shape[-3] == 1 else x,
                functools.partial(maybe_to_rgb, to_rgb=to_rgb),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(10),
                torchvision.transforms.Resize(resize_to, antialias=True),
                torchvision.transforms.RandomAffine(degrees=180, translate=(0.35, 0.35)),
            ]
        )

        self.preprocess_image = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                # lambda x: x.repeat(3, 1, 1) if to_rgb and x.shape[-3] == 1 else x,
                functools.partial(maybe_to_rgb, to_rgb=to_rgb),
                # torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                # torchvision.transforms.Grayscale(),
                torchvision.transforms.Resize(resize_to, antialias=True),
                # torchvision.transforms.RandomAffine(degrees=10,translate=(0.8, 0.8)),
            ]
        )

    def __len__(self):
        return len(self.images_paths)

    def get_label(self, img_name):
        if 'A172' in img_name:
            label = 0
        elif 'BT474' in img_name:
            label = 1
        elif 'BV2' in img_name:
            label = 2
        elif 'Huh7' in img_name:
            label = 3
        elif 'MCF7' in img_name:
            label = 4
        elif 'SHSY5Y' in img_name:
            label = 5
        elif 'SkBr3' in img_name:
            label = 6
        elif 'SKOV3' in img_name:
            label = 7
        else:
            raise ValueError('Class not recognized')
        return label

    def __getitem__(self, idx):
        img_path = self.images_paths[idx]
        img_name = os.path.basename(img_path)
        label = self.get_label(img_name)
        image = Image.open(img_path).convert('L')  # avoiding color bias
        if self.mode == 'train':
            tensor_image = self.preprocess_only_train(image)
        elif self.mode == 'test' or self.mode == 'val' or self.mode == 'valid':
            tensor_image = self.preprocess_image(image)
        else:
            raise ValueError("Mode not implemented")
        # return {
        #     'pixel_values': tensor_image,
        #     'labels': torch.tensor(label)
        # }
        return tensor_image, torch.tensor(label)
