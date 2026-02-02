import os
import random
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from library.config import load_data


def get_transforms(mode="train"):
    """
    Returns a transformation function for the dataset.
    Strictly limits augmentation to RandomHorizontalFlip and RandomVerticalFlip for training.
    Normalizes images to [0, 1] range.

    Args:
        mode (str): 'train' for augmentation, 'val' or 'test' for deterministic.

    Returns:
        callable: A function that takes a numpy image and returns a tensor.
    """

    def train_transform(image):
        # image is HWC, uint8, 32x32
        # Random Horizontal Flip
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
        # Random Vertical Flip
        if random.random() > 0.5:
            image = cv2.flip(image, 0)

        # Convert HWC to CHW
        image = image.transpose((2, 0, 1))
        # Normalize to [0, 1]
        image = torch.from_numpy(image.copy()).float() / 255.0
        return image

    def eval_transform(image):
        # Convert HWC to CHW
        image = image.transpose((2, 0, 1))
        # Normalize to [0, 1]
        image = torch.from_numpy(image.copy()).float() / 255.0
        return image

    if mode == "train":
        return train_transform
    else:
        return eval_transform


class CactusDataset(Dataset):
    """
    Dataset class for Cactus identification.
    Wraps numpy arrays of images and labels.
    """

    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        if self.transform:
            img = self.transform(img)
        else:
            # Fallback if no transform provided
            img = img.transpose((2, 0, 1))
            img = torch.from_numpy(img.copy()).float() / 255.0

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img, label
        return img


def get_data_arrays(load_cached_data=True):
    """
    Wrapper to load data arrays using the config library.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing train/val/test images and labels/ids.
    """
    return load_data(load_cached_data=load_cached_data)
