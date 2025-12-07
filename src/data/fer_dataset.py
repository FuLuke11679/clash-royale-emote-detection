# src/data/fer_dataset.py

import os
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


# If your folders are exactly: angry, disgust, fear, happy, neutral, sad
# this will match ImageFolder's alphabetical ordering.
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(CLASS_NAMES)


def get_transforms(split: str):
    """
    Returns transforms for train/val/test.
    Uses ImageNet-style normalization to match pretrained ResNet.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )
    else:
        # val/test – no heavy augmentation
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )


class EmotionSubset(Dataset):
    """
    A subset of an ImageFolder with its own transform.
    This lets us use different transforms for train vs val
    while sharing the same underlying images.
    """

    def __init__(self, base_dataset: ImageFolder, indices, transform=None):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

        # Expose classes and targets for convenience
        self.classes = base_dataset.classes
        # targets: labels (ints) for just this subset
        all_targets = np.array(base_dataset.targets)
        self.targets = all_targets[self.indices].tolist()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        # base_dataset returns (PIL image, label) with its own transform,
        # but we ignore its transform and load the raw image manually.
        path, label = self.base_dataset.samples[base_idx]
        img = self.base_dataset.loader(path)  # PIL Image

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def get_dataloaders(
    train_dir: str = "data/train",
    test_dir: str = "data/test",
    batch_size: int = 64,
    num_workers: int = 4,
    val_ratio: float = 0.2,
    train_augment: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, val, test dataloaders from folder-structured FER-2013.

    Expected layout:
      data/train/angry/*.jpg
      data/train/disgust/*.jpg
      ...
      data/test/angry/*.jpg
      ...

    We create a validation split from the train folder.
    """

    # Base dataset with no transform (for splitting)
    base_train = ImageFolder(root=train_dir)

    # Sanity check class names
    print("Found classes:", base_train.classes)

    indices = np.arange(len(base_train))
    targets = np.array(base_train.targets)

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        stratify=targets,
        random_state=42,
    )

    train_ds = EmotionSubset(
        base_dataset=base_train,
        indices=train_idx,
        transform=get_transforms("train") if train_augment else get_transforms("val"),
    )
    val_ds = EmotionSubset(
        base_dataset=base_train,
        indices=val_idx,
        transform=get_transforms("val"),
    )

    # Test dataset uses its own folder
    test_ds = ImageFolder(
        root=test_dir,
        transform=get_transforms("test"),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
