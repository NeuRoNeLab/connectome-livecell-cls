import os
from typing import Optional, Callable, Final, Union, Iterable
import pandas as pd
import torch
import lightning as lit
from torch.utils.data import random_split, DataLoader, Sampler
from ds.livecell_dataset import CellDataset
from torch.utils.data import WeightedRandomSampler


DRUG_VS_NO_DRUG: Final[str] = "drug_vs_nodrug"
BCCD: Final[str] = "bccd"
SUPPORTED_DATASETS: Final = {
    DRUG_VS_NO_DRUG,
    BCCD
}


class LivecellDataModule(lit.LightningDataModule):
    def __init__(self,
                 root_images: Optional[str] = None,
                 task: str = 'mask',
                 dataset_name: str = DRUG_VS_NO_DRUG,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 shuffle: bool = True,
                 custom_collate_fn: Optional[Callable] = None,
                 custom_sampler: Optional[Union[torch.utils.data.Sampler, Iterable]] = None,
                 use_weighted_random_sampler: bool = False,
                 resize_to: tuple[int, int] = (224, 224),
                 to_rgb: bool = False):
        super().__init__()

        # Store attributes
        self._root_path: str = root_images
        self._dataset_name: str = dataset_name
        self._batch_size: int = batch_size
        self._num_workers: int = num_workers
        self._shuffle: bool = shuffle
        self._custom_collate_fn: Optional[Callable] = custom_collate_fn
        self._custom_sampler: Optional[Union[torch.utils.data.Sampler, Iterable]] = custom_sampler
        self._resize_to: tuple[int, int] = resize_to
        self._to_rgb: bool = to_rgb
        if use_weighted_random_sampler:
            print('Using WeightedRandomSampler')
            self._custom_sampler_training: Optional[Union[torch.utils.data.WeightedRandomSampler, Iterable]] = None
            self._use_weighted_random_sampler: bool = use_weighted_random_sampler
        else:
            self._custom_sampler_training = self._custom_sampler
            self._use_weighted_random_sampler = False
        self._task = task
        if dataset_name not in SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset {dataset_name}. Must be one of {SUPPORTED_DATASETS}.")

        # Initialize train, validation and test sets
        self._train_set = None
        self._val_set = None
        self._test_set = None
        self._predict_set = None

    def _get_label(self, img_name):
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

    def get_sample_weights(self, images_paths):
        images = []
        labels = []
        for img_file in images_paths:
            image_name = os.path.basename(img_file)
            label = self._get_label(image_name)
            images.append(img_file)
            labels.append(label)
        df = pd.DataFrame({'images': images, 'labels': labels})
        class_counts = df.labels.value_counts()
        print(f'Class counts: {class_counts}')
        sample_weights = [1 / class_counts[i] for i in df.labels.values]
        return sample_weights

    @property
    def use_weighted_random_sampler(self) -> bool:
        return self._use_weighted_random_sampler

    @property
    def task(self) -> str:
        return self._task

    @property
    def root_path(self) -> str:
        return self._root_path

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def resize_to(self) -> tuple[int, int]:
        return self._resize_to

    @property
    def to_rgb(self) -> bool:
        return self._to_rgb

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def num_workers(self) -> int:
        return self._num_workers

    @property
    def shuffle(self) -> bool:
        return self._shuffle

    @property
    def custom_collate_fn(self) -> Optional[Callable]:
        return self._custom_collate_fn

    @property
    def custom_sampler(self) -> Optional[Union[Sampler, Iterable]]:
        return self._custom_sampler

    @property
    def custom_sampler_training(self):
        return self._custom_sampler_training

    def prepare_data(self):
        # Download data if needed
        pass

    def setup(self, stage: str):
        if stage == "fit":
            train_ds, tmp_ds, val_ds, test_ds = self._train_set, None, self._val_set, self._test_set
            if self.dataset_name == DRUG_VS_NO_DRUG or self.dataset_name == BCCD:
                # Get training dataset
                train_ds = CellDataset(
                    root_path=self.root_path,
                    mode='train',
                    task=self.task,
                    resize_to=self.resize_to,
                    to_rgb=self.to_rgb
                )
                sample_weights = self.get_sample_weights(train_ds.images_paths)
                if self.use_weighted_random_sampler:
                    self._custom_sampler_training = WeightedRandomSampler(
                        weights=sample_weights,
                        num_samples=len(train_ds),
                        replacement=True
                    )
                else:
                    self._custom_sampler_training = self._custom_sampler

                # Get validation dataset
                val_ds = CellDataset(
                    root_path=self.root_path,
                    mode='val',
                    task=self.task,
                    resize_to=self.resize_to,
                    to_rgb=self.to_rgb
                )

            # Set up the instance variables
            if train_ds is not None:
                self._train_set = train_ds
            if val_ds is not None:
                self._val_set = val_ds

        if stage == "test":
            test_ds = CellDataset(
                root_path=self.root_path,
                mode='test',
                task=self.task,
                resize_to=self.resize_to,
                to_rgb=self.to_rgb
            )
            if test_ds is not None:
                self._test_set = test_ds

        if stage == "predict":
            pass

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            dataset=self._train_set,
            batch_size=self._batch_size,
            shuffle=self.shuffle,
            sampler=self.custom_sampler_training,
            num_workers=self.num_workers,
            collate_fn=self.custom_collate_fn,
            drop_last=False,
            persistent_workers=True,
            pin_memory=True
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            dataset=self._val_set,
            batch_size=self._batch_size,
            shuffle=False,
            sampler=self.custom_sampler,
            num_workers=self.num_workers,
            collate_fn=self.custom_collate_fn,
            drop_last=False,
            persistent_workers=True,
            pin_memory=True
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            dataset=self._test_set,
            batch_size=self._batch_size,
            shuffle=False,
            sampler=self.custom_sampler,
            num_workers=self.num_workers,
            collate_fn=self.custom_collate_fn,
            drop_last=False
        )

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            dataset=self._predict_set,
            batch_size=self._batch_size,
            shuffle=False,
            sampler=self.custom_sampler,
            num_workers=self.num_workers,
            collate_fn=self.custom_collate_fn,
            drop_last=False
        )
