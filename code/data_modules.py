# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24, 2026

@author: Matthias Schmal
@tui: Data Intensive Systems and Visualization Group (dAI.SY)


"""

import os

import h5py
import torch
import torchvision

torchvision.disable_beta_transforms_warning()
import glob
import math
import time
import warnings
from multiprocessing import Pool
from zipfile import ZipFile

import datasets
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torchvision
from datasets import load_dataset
from PIL import Image
from sklearn.model_selection import train_test_split

# for transforms
from torchvision.transforms import v2
from tqdm import tqdm

# paths on cluster
cifar10_path = "../data/CIFAR/cifar-10"
cifar10_c_path = "../data"
cifar100_path = "../data"
cifar100_c_path = "../data"
svhn_path = "../data"
tinyImagenet_path = "../data"
tinyImagenetC_path = "../data"
imagenet100_path = "../data"
imagenetA_path = "../data"
imagenetO_path = "../data"
imagenetR_path = "../data"
imagenetC_path = "../data"


def get_datamodule_by_name(name):
    if name == "Cifar10":
        return Cifar10DataModule, 10
    elif name == "Cifar100":
        return Cifar100DataModule, 100
    elif name == "TinyImagenet":
        return TinyImagenetDataModule, 200
    elif name == "Imagenet100":
        return Imagenet100DataModule, 100
    else:
        raise NotImplementedError(
            "This dataset is not supported: Check spelling; available methods are in doc-str"
        )


# ---------------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------------


CORRUPTION_ORGA = {
    "noise": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather": ["frost", "snow", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
    "extra": ["spatter", "speckle_noise", "gaussian_blur", "saturate"],
}


class NpyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, transform=None, target_transform=None):
        self.images = x
        self.labels = torch.from_numpy(y).long()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def compute_ds_stats(dataloader, dataset_name=""):
    """Using the following convention to compute mean and std over one epoch
    mean = np.mean(train_images, axis=(0, 1, 2))  # axis are ordered as (batch, height, width, channels)
    and std = np.std(train_images, axis=(0, 1, 2)).
    """
    means = []
    stds = []
    for x, y in tqdm(dataloader, desc=f"Computing stats for {dataset_name}"):
        means.append(x.mean(dim=(0, 2, 3)))
        stds.append(x.std(dim=(0, 2, 3)))
    result = {
        "mean": torch.stack(means).mean(dim=0),
        "std": torch.stack(stds).mean(dim=0),
    }
    print(result)
    return result


def compute_all_ds_stats(num_workers=8):
    dataset_list = ["Cifar10", "Cifar100", "TinyImagenet", "Imagenet100"]
    for dataset in dataset_list:
        print("Statistics of ", dataset)
        datamodule_F, n_classes = get_datamodule_by_name(dataset)
        datamodule = datamodule_F(
            batch_size=100,
            num_workers_train=num_workers,
            normalize=False,
            cluster=True,
            transform="Base",
        )
        datamodule.setup("fit")
        dataloader = datamodule.train_dataloader()
        compute_ds_stats(dataloader, dataset)


def stratified_train_test_split(dataset_train, dataset_val, ratios):
    """Stratified train test split for Pytorch datasets
    Parameters:
        dataset_train: Pytorch dataset with train transform
        dataset_val: Pytorch dataset with val transform
        ratios: list of ratios for train and test split
    Returns:
        train_set: Pytorch dataset
        test_set: Pytorch dataset
    """
    labels = dataset_train.targets
    train_idx, test_idx = train_test_split(
        list(range(len(labels))), test_size=ratios[1], stratify=labels
    )
    train_set = torch.utils.data.Subset(dataset_train, train_idx)
    test_set = torch.utils.data.Subset(dataset_val, test_idx)
    return train_set, test_set


# ---------------------------------------------------------------------------------
# Cifar Datasets
# ---------------------------------------------------------------------------------


def get_transform_for_cifar(transform, base_transform, normalize, mean, std, new_size):
    if len(transform.split("_")) > 1:
        transform = get_transform_for_cifar(
            transform.split("_")[0],
            base_transform,
            normalize,
            mean,
            std,
            new_size,
        )
        return TwoCropTransform(transform)
    if transform is None:
        # transforms.Pad(4),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(32),
        aug_transform = base_transform
    elif transform == "Base":
        aug_transform = v2.Compose(
            [
                v2.ToPILImage(),
                v2.Resize(new_size, interpolation=Image.BICUBIC),
                v2.RandomCrop(32, padding=4),
                v2.RandomHorizontalFlip(),
                v2.PILToTensor(),
                v2.ConvertImageDtype(torch.float32),
            ]
        )
        if normalize:
            aug_transform = v2.Compose(
                [aug_transform, v2.Normalize(mean=mean, std=std)]
            )
    elif transform.__contains__("TwoCrop"):
        aug_transform = v2.Compose(
            [
                v2.ToPILImage(),
                v2.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
                v2.RandomHorizontalFlip(),
                v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                v2.RandomGrayscale(p=0.2),
                v2.PILToTensor(),
                v2.ConvertImageDtype(torch.float32),
            ]
        )
        if normalize:
            aug_transform = v2.Compose(
                [aug_transform, v2.Normalize(mean=mean, std=std)]
            )
    else:
        raise NotImplementedError

    return aug_transform


class Cifar10DataModule(pl.LightningDataModule):
    """Pytorch-Lightning data module for serving CIFAR-10 dataloaders with train/val split of 95 % / 5 %
    and corrupted test_dataloader with 5 severities available via dataloader_idx in [1, ..., 5]
    """

    def __init__(
        self,
        batch_size: int,
        num_workers_train: int = 2,
        num_workers_test: int = 2,
        normalize: bool = False,
        transform="Base",
        insert_corrupted_data: bool = False,
        test_batch_size=None,
        image_size=32,
    ):
        super().__init__()
        self.batch_size = batch_size
        if test_batch_size is None:
            self.test_batch_size = batch_size
        else:
            self.test_batch_size = int(test_batch_size)
        self.num_workers_train = num_workers_train
        self.num_workers_test = num_workers_test
        self.trainSet = None
        self.testSet = None
        self.valSet = None
        self.corrupted_testSets = []
        self.corrupted_testSets_names = []
        self.insert_corrupted_data = insert_corrupted_data
        # Statistics of Cifar10
        self.cifar10_mean = [0.4914, 0.4822, 0.4465]  # [0.4248, 0.4152, 0.3842]
        self.cifar10_std = [0.2470, 0.2435, 0.2616]  # [0.2828, 0.2778, 0.2845]
        self.new_size = image_size

        if normalize:
            self.base_transform = v2.Compose(
                [
                    v2.ToPILImage(),
                    v2.Resize(self.new_size, interpolation=Image.BICUBIC),
                    v2.PILToTensor(),
                    v2.ConvertImageDtype(torch.float32),
                    v2.Normalize(mean=self.cifar10_mean, std=self.cifar10_std),
                ]
            )
        else:
            self.base_transform = v2.Compose(
                [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
            )

        self.aug_transform = get_transform_for_cifar(
            transform,
            self.base_transform,
            normalize,
            self.cifar10_mean,
            self.cifar10_std,
            self.new_size,
        )

        self.data_path = cifar10_path
        self.corrupted_data = cifar10_c_path
        self.save_hyperparameters()

    def setup(self, stage: str):
        if stage == "fit":
            cifar_big_t = torchvision.datasets.CIFAR10(
                root=self.data_path,
                train=True,
                transform=self.aug_transform,
                download=False,
            )

            cifar_big_v = torchvision.datasets.CIFAR10(
                root=self.data_path,
                train=True,
                transform=self.base_transform,
                download=False,
            )

            self.trainSet, self.valSet = stratified_train_test_split(
                cifar_big_t, cifar_big_v, [0.95, 0.05]
            )

        if stage == "test":
            if not len(self.corrupted_testSets_names) > 0:
                self.testSet = torchvision.datasets.CIFAR10(
                    root=self.data_path, train=False, transform=self.base_transform
                )

                self.oodSet = torchvision.datasets.SVHN(
                    root=svhn_path,
                    split="test",
                    transform=self.base_transform,
                    download=False,
                )

                if self.insert_corrupted_data:
                    # Load corrupted CIFAR-10
                    for severity in list(range(1, 6)):
                        for dist_group in CORRUPTION_ORGA.keys():
                            numpy_files = [
                                self.corrupted_data + "/CIFAR-10-C/" + k + ".npy"
                                for k in CORRUPTION_ORGA[dist_group]
                            ]
                            numpy_files.append(
                                self.corrupted_data + "/CIFAR-10-C/labels.npy"
                            )
                            data_list = []
                            for file in numpy_files:
                                if file.__contains__("labels"):
                                    labels = np.load(file)[
                                        (severity - 1) * 10000 : severity * 10000
                                    ]
                                else:
                                    data_list.append(
                                        np.load(file)[
                                            (severity - 1) * 10000 : severity * 10000
                                        ]
                                    )

                            data = np.vstack(data_list)
                            labels = np.tile(labels, 19)
                            self.corrupted_testSets.append(
                                NpyDataset(data, labels, transform=self.base_transform)
                            )
                            self.corrupted_testSets_names.append(
                                dist_group + str(severity)
                            )

    def n_train_points(self):
        return len(self.trainSet)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.trainSet,
            self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.num_workers_train,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valSet,
            self.test_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers_train,
        )

    def test_dataloader(self):
        if not self.insert_corrupted_data:
            return torch.utils.data.DataLoader(
                self.testSet, self.test_batch_size, shuffle=False, drop_last=False
            )
        else:
            return [
                torch.utils.data.DataLoader(
                    self.testSet, self.test_batch_size, shuffle=False, drop_last=False
                )
            ] + [
                torch.utils.data.DataLoader(
                    self.corrupted_testSets[i],
                    self.test_batch_size,
                    shuffle=False,
                    drop_last=False,
                )
                for i in list(range(len(self.corrupted_testSets)))
            ]

    def ood_dataloader(self):
        return torch.utils.data.DataLoader(
            self.oodSet,
            self.test_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers_train,
        )


class Cifar100DataModule(pl.LightningDataModule):
    """Pytorch-Lightning data module for serving CIFAR-100 dataloaders with train/val split of 95 % / 5 %
    and corrupted test_dataloader with 5 severities available via dataloader_idx in [1, ..., 5]
    """

    def __init__(
        self,
        batch_size: int,
        num_workers_train: int = 2,
        num_workers_test: int = 1,
        normalize: bool = False,
        transform="Base",
        insert_corrupted_data: bool = False,
        test_batch_size=None,
        image_size=32,
    ):
        super().__init__()
        self.batch_size = batch_size
        if test_batch_size is None:
            self.test_batch_size = batch_size
        else:
            self.test_batch_size = int(test_batch_size)
        self.num_workers_train = num_workers_train
        self.num_workers_test = num_workers_test
        self.trainSet = None
        self.testSet = None
        self.valSet = None
        self.corrupted_testSets = []
        self.corrupted_testSets_names = []
        self.insert_corrupted_data = insert_corrupted_data
        self.normalize = normalize
        # Statistics of Cifar10
        self.cifar100_mean = [0.4381, 0.4179, 0.3775]
        self.cifar100_std = [0.3003, 0.2872, 0.2936]
        self.new_size = image_size

        if normalize:
            self.base_transform = v2.Compose(
                [
                    v2.ToPILImage(),
                    v2.Resize(self.new_size, interpolation=Image.BICUBIC),
                    v2.PILToTensor(),
                    v2.ConvertImageDtype(torch.float32),
                    v2.Normalize(mean=self.cifar100_mean, std=self.cifar100_std),
                ]
            )
        else:
            self.base_transform = v2.Compose(
                [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
            )

        self.aug_transform = get_transform_for_cifar(
            transform,
            self.base_transform,
            normalize,
            self.cifar100_mean,
            self.cifar100_std,
            self.new_size,
        )

        self.data_path = cifar100_path
        self.corrupted_data = cifar100_c_path
        self.save_hyperparameters()

    def setup(self, stage: str):
        if stage == "fit":
            cifar_big_t = torchvision.datasets.CIFAR100(
                root=self.data_path,
                train=True,
                transform=self.aug_transform,
                download=False,
            )

            cifar_big_v = torchvision.datasets.CIFAR100(
                root=self.data_path,
                train=True,
                transform=self.base_transform,
                download=False,
            )

            self.trainSet, self.valSet = stratified_train_test_split(
                cifar_big_t, cifar_big_v, [0.95, 0.05]
            )

        if stage == "test":
            if not len(self.corrupted_testSets_names) > 0:
                self.testSet = torchvision.datasets.CIFAR100(
                    root=self.data_path, train=False, transform=self.base_transform
                )

                self.oodSet = torchvision.datasets.SVHN(
                    root=svhn_path,
                    split="test",
                    transform=self.base_transform,
                    download=False,
                )

                if self.insert_corrupted_data:
                    # Load corrupted CIFAR-10
                    for severity in list(range(1, 6)):
                        for dist_group in CORRUPTION_ORGA.keys():
                            numpy_files = [
                                self.corrupted_data + "/CIFAR-100-C/" + k + ".npy"
                                for k in CORRUPTION_ORGA[dist_group]
                            ]
                            numpy_files.append(
                                self.corrupted_data + "/CIFAR-100-C/labels.npy"
                            )
                            data_list = []
                            for file in numpy_files:
                                if file.__contains__("labels"):
                                    labels = np.load(file)[
                                        (severity - 1) * 10000 : severity * 10000
                                    ]
                                else:
                                    data_list.append(
                                        np.load(file)[
                                            (severity - 1) * 10000 : severity * 10000
                                        ]
                                    )

                            data = np.vstack(data_list)
                            labels = np.tile(labels, 19)
                            self.corrupted_testSets.append(
                                NpyDataset(data, labels, transform=self.base_transform)
                            )
                            self.corrupted_testSets_names.append(
                                dist_group + str(severity)
                            )

    def n_train_points(self):
        return len(self.trainSet)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.trainSet,
            self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.num_workers_train,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valSet,
            self.test_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers_test,
        )

    def test_dataloader(self):
        if not self.insert_corrupted_data:
            return torch.utils.data.DataLoader(
                self.testSet, self.test_batch_size, shuffle=False, drop_last=False
            )
        else:
            return [
                torch.utils.data.DataLoader(
                    self.testSet, self.test_batch_size, shuffle=False, drop_last=False
                )
            ] + [
                torch.utils.data.DataLoader(
                    self.corrupted_testSets[i],
                    self.test_batch_size,
                    shuffle=False,
                    drop_last=False,
                )
                for i in list(range(len(self.corrupted_testSets)))
            ]

    def ood_dataloader(self):
        return torch.utils.data.DataLoader(
            self.oodSet,
            self.test_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers_train,
        )


# ---------------------------------------------------------------------------------
# Imagenet-based datasets
# ---------------------------------------------------------------------------------


class FilteredImageFolderDataset(torch.utils.data.Dataset):
    """Huggingface datasets do not support autoaugment in pytorch and multiprocessing seems to be broken."""

    def __init__(self, base_path, filter_file, transform=None):
        super(FilteredImageFolderDataset, self).__init__()

        self.base_path = base_path
        self.filter_classes = list(pd.read_csv(filter_file, header=None)[0])

        self.image_paths = []
        self.targets = []

        self.transform = transform

    def prepare_images(self):
        folders = glob.glob(self.base_path + "/*/")
        for folder in folders:
            class_identifier = os.path.basename(os.path.dirname(folder))
            if class_identifier in self.filter_classes:
                images = glob.glob(folder + "*")
                label = self.filter_classes.index(class_identifier)
                for image in images:
                    self.image_paths.append(image)
                    self.targets.append(label)

    def get_samples_per_class(self):
        return {
            i: len(np.where(np.array(self.targets) == i)[0])
            for i in np.unique(np.array(self.targets))
        }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = self.image_paths[index]
        label = self.targets[index]
        img = Image.open(img).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class TinyImagenetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers_train: int = 2,
        num_workers_test: int = 2,
        normalize: bool = False,
        transform="Base",
        insert_corrupted_data: bool = False,
        image_size: int = 64,
        test_batch_size=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        if test_batch_size is None:
            self.test_batch_size = batch_size
        else:
            self.test_batch_size = int(test_batch_size)

        self.num_workers_train = num_workers_train
        self.num_workers_test = num_workers_test
        self.transform = transform
        self.corrupted_testSets = []
        self.corrupted_testSets_names = []
        self.insert_corrupted_data = insert_corrupted_data
        self.corrupted_data = tinyImagenetC_path
        self.normalize = normalize
        self.new_size = int(image_size)

        # Statistics of EMNIST
        if normalize:
            self.tinyimagenet_mean = [0.4461, 0.4231, 0.3827]
            self.tinyimagenet_std = [0.2894, 0.2811, 0.2885]
        else:
            self.tinyimagenet_mean = [0.0, 0.0, 0.0]
            self.tinyimagenet_std = [1.0, 1.0, 1.0]

        self.trainSet = None
        self.testSet = None
        self.valSet = None
        self.oodSet = None

        self.save_hyperparameters()

    def get_transforms(self, description="", n=2, m=14):
        if len(description.split("_")) > 1:
            transform = self.get_transforms(description.split("_")[0])
            return TwoCropTransform(transform)
        if description == "Base":
            return v2.Compose(
                [
                    v2.ToPILImage(),
                    v2.Resize(self.new_size, interpolation=Image.BICUBIC),
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.RandomCrop(self.new_size, padding=6),
                    v2.RandAugment(magnitude=9),
                    # v2.RandAugment(magnitude=14),
                    v2.PILToTensor(),
                    v2.ConvertImageDtype(torch.float32),
                    v2.Normalize(
                        mean=self.tinyimagenet_mean, std=self.tinyimagenet_std
                    ),
                ]
            )
        elif description == "RandAugJitter":
            return v2.Compose(
                [
                    v2.ToPILImage(),
                    # v2.RandomResizedCrop(size=self.new_size, scale=(0.2, 1.)),
                    v2.Resize(self.new_size, interpolation=Image.BICUBIC),
                    v2.RandomHorizontalFlip(),
                    v2.RandAugment(n, m),
                    # v2.RandomApply([
                    #     v2.ColorJitter(0.2, 0.2, 0.2, 0.1)
                    # ], p=0.8),
                    v2.PILToTensor(),
                    v2.ConvertImageDtype(torch.float32),
                    v2.RandomApply(
                        [
                            v2.RandomChoice(
                                [
                                    # v2.GaussianBlur(kernel_size=9, sigma=(1., 8.)),
                                    # v2.ElasticTransform(),
                                    v2.GaussianNoise(),
                                    v2.RandomErasing(0.1),
                                ]
                            )
                        ],
                        p=0.2,
                    ),
                    v2.Normalize(
                        mean=self.tinyimagenet_mean, std=self.tinyimagenet_std
                    ),
                ]
            )
        elif description == "test":
            return v2.Compose(
                [
                    v2.ToPILImage(),
                    v2.Resize(self.new_size, interpolation=Image.BICUBIC),
                    v2.PILToTensor(),
                    v2.ConvertImageDtype(torch.float32),
                    v2.Normalize(
                        mean=self.tinyimagenet_mean, std=self.tinyimagenet_std
                    ),
                ]
            )
        else:
            raise NotImplementedError

    def setup(self, stage: str) -> None:
        if stage == "fit":
            tiny_imagenet = torchvision.datasets.ImageFolder(
                tinyImagenet_path + "/train",
                transform=self.get_transforms(self.transform),
            )
            tiny_imagenet_val = torchvision.datasets.ImageFolder(
                tinyImagenet_path + "/train", transform=self.get_transforms("test")
            )

            self.trainSet, self.valSet = stratified_train_test_split(
                tiny_imagenet, tiny_imagenet_val, [0.95, 0.05]
            )

        if stage == "test":
            if not len(self.corrupted_testSets_names) > 0:
                self.testSet = torchvision.datasets.ImageFolder(
                    tinyImagenet_path + "/val", transform=self.get_transforms("test")
                )

                self.oodSet = FilteredImageFolderDataset(
                    imagenetO_path,
                    os.path.join(self.corrupted_data, "tiny-imagenet-200.txt"),
                    self.get_transforms("croptest"),
                )
                self.oodSet.prepare_images()

                if self.insert_corrupted_data:
                    for severity in list(range(1, 6)):
                        for dist_group in CORRUPTION_ORGA.keys():
                            if dist_group == "extra":
                                continue
                            image_folders = [
                                os.path.join(self.corrupted_data, k, str(severity))
                                for k in CORRUPTION_ORGA[dist_group]
                            ]
                            ds_list = []
                            for fold in image_folders:
                                # print(fold, os.path.exists(fold))
                                ds = FilteredImageFolderDataset(
                                    fold,
                                    os.path.join(
                                        self.corrupted_data, "tiny-imagenet-200.txt"
                                    ),
                                    self.get_transforms("test"),
                                )
                                ds.prepare_images()
                                ds_list.append(ds)

                            self.corrupted_testSets.append(
                                torch.utils.data.ConcatDataset(ds_list)
                            )
                            self.corrupted_testSets_names.append(
                                dist_group + str(severity)
                            )
                    im_a = FilteredImageFolderDataset(
                        imagenetA_path,
                        os.path.join(self.corrupted_data, "tiny-imagenet-200.txt"),
                        self.get_transforms("croptest"),
                    )
                    im_a.prepare_images()
                    im_r = FilteredImageFolderDataset(
                        imagenetR_path,
                        os.path.join(self.corrupted_data, "tiny-imagenet-200.txt"),
                        self.get_transforms("croptest"),
                    )
                    im_r.prepare_images()

                    self.corrupted_testSets.append(im_a)
                    self.corrupted_testSets_names.append("imagenetA")
                    self.corrupted_testSets.append(im_r)
                    self.corrupted_testSets_names.append("imagenetR")

    def n_train_points(self):
        return len(self.trainSet)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.trainSet,
            self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers_train,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valSet,
            self.test_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers_test,
        )

    def test_dataloader(self):
        if not self.insert_corrupted_data:
            return torch.utils.data.DataLoader(
                self.testSet, self.test_batch_size, shuffle=False, drop_last=False
            )
        else:
            return [
                torch.utils.data.DataLoader(
                    self.testSet, self.test_batch_size, shuffle=False, drop_last=False
                )
            ] + [
                torch.utils.data.DataLoader(
                    self.corrupted_testSets[i],
                    self.test_batch_size,
                    shuffle=False,
                    drop_last=False,
                )
                for i in list(range(len(self.corrupted_testSets)))
            ]

    def ood_dataloader(self):
        return torch.utils.data.DataLoader(
            self.oodSet,
            self.test_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers_train,
        )


class CachedHuggingfaceImageDataset(torch.utils.data.Dataset):
    """Huggingface datasets do not support autoaugment in pytorch and multiprocessing seems to be broken."""

    def __init__(
        self,
        huggingface_dataset,
        transform_before_cache=None,
        transform_after_cache=None,
        caching_worker=2,
    ):
        super(CachedHuggingfaceImageDataset, self).__init__()

        self.hf_dataset = huggingface_dataset

        self.transform_before = transform_before_cache
        self.transform_after = transform_after_cache

        self.caching_worker = caching_worker
        self.cached_images = []
        self.targets = []

    def cache_samples(self, image_idxes):
        res_img = []
        res_label = []
        for image_idx in image_idxes:
            try:
                sample = self.hf_dataset[image_idx]
                label = np.array(sample["label"])
                label = torch.from_numpy(label)
                img = sample["image"].convert("RGB").copy()
                if self.transform_before:
                    img = self.transform_before(img)
                res_img.append(img)
                res_label.append(label)
            except:
                print(image_idx)
                continue
        return res_img, res_label

    def cache_all_samples(self, bs=100):
        tic = time.time()
        work = [
            list(range(i, min(i + bs, self.__len__())))
            for i in range(0, self.__len__(), bs)
        ]
        # with Pool(processes=self.caching_worker) as pool:
        #     out = pool.map(self.cache_samples, work)
        out = []
        for w in tqdm(work):
            out.append(self.cache_samples(w))
        self.cached_images = sum([o[0] for o in out], [])
        self.targets = sum([o[1] for o in out], [])
        toc = time.time()
        print("All images cached in [min]: ", (toc - tic) / 60)
        print(
            "Cached Images:",
            len(self.cached_images),
            "File length:",
            len(self.hf_dataset),
        )

    def get_samples_per_class(self):
        return {
            i: len(np.where(np.array(self.targets) == i)[0])
            for i in np.unique(np.array(self.targets))
        }

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, index):
        img = self.cached_images[index]
        label = self.targets[index]
        if self.transform_after:
            img = self.transform_after(img)
        return img, label


class H5pyDataset(torch.utils.data.IterableDataset):
    def __init__(self, base_path, h5_file, transform_after_cache=None):
        super(H5pyDataset, self).__init__()

        self.h5file = os.path.join(base_path, h5_file)

        self.transform_after = transform_after_cache
        self.start = 0
        self.end = self.__len__()

    def __len__(self):
        with h5py.File(self.h5file, "r", libver="latest", swmr=True) as file:
            return len(file["targets"])

    # def __getitem__(self, index):
    #     with h5py.File(self.h5file, 'r', libver='latest', swmr=True) as file:
    #         img = torch.from_numpy(np.moveaxis(file['images'][index], 2, 0))
    #         label = file['targets'][index]
    #     if self.transform_after:
    #         img = self.transform_after(img)
    #     return img, label

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(
                math.ceil((self.end - self.start) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        with h5py.File(self.h5file, "r", libver="latest", swmr=True) as file:
            ims = torch.from_numpy(
                np.moveaxis(file["images"][iter_start:iter_end], 3, 1)
            )
            labels = file["targets"][iter_start:iter_end]
            for im, l in zip(ims, labels):
                if self.transform_after:
                    im = self.transform_after(im)
                yield im, l


class Imagenet100DataModule(pl.LightningDataModule):
    """This is build as a general class for Huggingface image datasets"""

    def __init__(
        self,
        batch_size: int,
        num_workers_train: int = 2,
        num_workers_test: int = 2,
        normalize: bool = True,
        transform="Base",
        insert_corrupted_data=False,
        image_size_train=224,
        image_size_test=224,
        test_batch_size=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        if test_batch_size is None:
            self.test_batch_size = batch_size
        else:
            self.test_batch_size = int(test_batch_size)

        self.num_workers_train = num_workers_train
        self.num_workers_test = num_workers_test
        self.transform = transform
        self.train_image_size = int(image_size_train)
        self.test_image_size = int(image_size_test)
        self.insert_corrupted_data = insert_corrupted_data
        self.corrupted_testSets = []
        self.corrupted_testSets_names = []
        self.corrupted_data = imagenetC_path

        # Requires changes, depending on the dataset
        self.data_files = {
            "train": "data/train*.parquet",
            "validation": "data/validation*.parquet",
        }

        if normalize:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        else:
            self.mean = [0.0, 0.0, 0.0]
            self.std = [1.0, 1.0, 1.0]

        self.trainSet = None
        self.testSet = None
        self.valSet = None

        self.save_hyperparameters()

    def get_transforms(self, description="", n=2, m=9):
        if len(description.split("_")) > 1:
            transform = self.get_transforms(description.split("_")[0])
            return TwoCropTransform(transform)
        if description == "beforeCaching":
            return v2.Compose(
                [
                    v2.ToPILImage(),
                    v2.Resize(self.train_image_size),
                ]
            )
        elif description == "train" or description == "Base":
            return v2.Compose(
                [
                    v2.CenterCrop(size=self.train_image_size),
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.RandAugment(magnitude=5),
                    v2.PILToTensor(),
                    v2.ConvertImageDtype(torch.float32),
                    v2.Normalize(mean=self.mean, std=self.std),
                    v2.RandomErasing(0.1),
                ]
            )
        elif description == "RandAugJitter":
            # derived from https://albumentations.ai/docs/3-basic-usage/image-classification/
            # https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/data.py
            # https://github.com/kiimmm/GenSCL/blob/main/data/data.py
            aug_transform = v2.Compose(
                [
                    v2.RandomResizedCrop(size=self.train_image_size, scale=(0.2, 1.0)),
                    v2.RandAugment(n, m),
                    v2.RandomHorizontalFlip(),
                    v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    v2.PILToTensor(),
                    v2.ConvertImageDtype(torch.float32),
                    v2.RandomApply(
                        [
                            v2.RandomChoice(
                                [
                                    v2.GaussianBlur(),
                                    v2.ElasticTransform(),
                                    v2.GaussianNoise(),
                                    v2.RandomErasing(0.1),
                                ]
                            )
                        ],
                        p=0.3,
                    ),
                    v2.Normalize(mean=mean, std=std),
                ]
            )
            return aug_transform
        elif description == "beforeCachingTest":
            return v2.Compose(
                [
                    v2.ToPILImage(),
                    v2.Resize(self.test_image_size),
                    v2.CenterCrop(size=self.test_image_size),
                ]
            )
        elif description == "test":
            return v2.Compose(
                [
                    v2.PILToTensor(),
                    v2.ConvertImageDtype(torch.float32),
                    v2.Normalize(mean=self.mean, std=self.std),
                ]
            )
        elif description == "corrtest":
            return v2.Compose(
                [
                    v2.ToPILImage(),
                    v2.Resize(self.test_image_size),
                    v2.CenterCrop(size=self.test_image_size),
                    v2.PILToTensor(),
                    v2.ConvertImageDtype(torch.float32),
                    v2.Normalize(mean=self.mean, std=self.std),
                ]
            )
        elif description == "h5AfterCache":
            return v2.Compose(
                [
                    v2.ConvertImageDtype(torch.float32),
                    v2.Normalize(mean=self.mean, std=self.std),
                ]
            )
        else:
            raise NotImplementedError

    def setup(self, stage: str) -> None:
        if stage == "fit":
            ds_train = load_dataset(
                imagenet100_path,
                data_files=self.data_files,
                split="train",
                keep_in_memory=False,
            )
            self.trainSet = CachedHuggingfaceImageDataset(
                ds_train,
                self.get_transforms("beforeCaching"),
                self.get_transforms(self.transform),
                caching_worker=self.num_workers_train,
            )
            self.trainSet.cache_all_samples()

            ds_val = load_dataset(
                imagenet100_path,
                data_files=self.data_files,
                split="validation",
                keep_in_memory=False,
            )
            self.valSet = CachedHuggingfaceImageDataset(
                ds_val,
                self.get_transforms("beforeCachingTest"),
                self.get_transforms("test"),
                caching_worker=self.num_workers_train,
            )
            self.valSet.cache_all_samples()

        if stage == "test":
            if self.valSet is None:
                ds_val = load_dataset(
                    imagenet100_path,
                    data_files=self.data_files,
                    split="validation",
                    keep_in_memory=False,
                )
                self.valSet = CachedHuggingfaceImageDataset(
                    ds_val,
                    self.get_transforms("beforeCachingTest"),
                    self.get_transforms("test"),
                    caching_worker=self.num_workers_train,
                )
                self.valSet.cache_all_samples()
            if not len(self.corrupted_testSets_names) > 0:
                self.testSet = self.valSet

                self.oodSet = FilteredImageFolderDataset(
                    imagenetO_path,
                    os.path.join(imagenet100_path, "imagenet100.txt"),
                    self.get_transforms("corrtest"),
                )
                self.oodSet.prepare_images()

                if self.insert_corrupted_data:
                    for severity in list(range(1, 6)):
                        for dist_group in CORRUPTION_ORGA.keys():
                            ds = H5pyDataset(
                                os.path.join(self.corrupted_data, "imagenet_c_100_h5"),
                                dist_group + "_" + str(severity) + ".h5",
                                self.get_transforms("h5AfterCache"),
                            )
                            self.corrupted_testSets.append(ds)
                            self.corrupted_testSets_names.append(
                                dist_group + str(severity)
                            )
                    im_a = FilteredImageFolderDataset(
                        imagenetA_path,
                        os.path.join(imagenet100_path, "imagenet100.txt"),
                        self.get_transforms("corrtest"),
                    )
                    im_a.prepare_images()
                    im_r = FilteredImageFolderDataset(
                        imagenetR_path,
                        os.path.join(imagenet100_path, "imagenet100.txt"),
                        self.get_transforms("corrtest"),
                    )
                    im_r.prepare_images()

                    self.corrupted_testSets.append(im_a)
                    self.corrupted_testSets_names.append("imagenetA")
                    self.corrupted_testSets.append(im_r)
                    self.corrupted_testSets_names.append("imagenetR")

    def n_train_points(self):
        return len(self.trainSet)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.trainSet,
            self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers_train,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valSet,
            self.test_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers_test,
        )

    def test_dataloader(self):
        if not self.insert_corrupted_data:
            return self.val_dataloader()
        else:
            return [self.val_dataloader()] + [
                torch.utils.data.DataLoader(
                    self.corrupted_testSets[i],
                    self.test_batch_size,
                    shuffle=False,
                    drop_last=False,
                )
                for i in list(range(len(self.corrupted_testSets)))
            ]

    def ood_dataloader(self):
        return torch.utils.data.DataLoader(
            self.oodSet,
            self.test_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers_train,
        )
