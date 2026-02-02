import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library import config, data_processing


class MGMTDataset(Dataset):
    """
    PyTorch Dataset for MGMT Promoter Methylation Prediction.
    Accepts pre-processed 9-channel volumetric stacks.
    """

    def __init__(self, ids, images, targets=None, transform=None):
        """
        Args:
            ids (np.array): Array of Subject IDs.
            images (np.array): Array of images with shape (N, C, H, W).
            targets (np.array, optional): Array of target labels.
            transform (A.Compose, optional): Albumentations pipeline.
        """
        self.ids = ids
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Retrieve image: (C, H, W) -> (9, 224, 224)
        img = self.images[idx]

        # Transpose to (H, W, C) for Albumentations
        img = np.transpose(img, (1, 2, 0))

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]
        else:
            # Fallback if no transform provided (e.g. raw inference)
            # Convert to tensor manually: (H, W, C) -> (C, H, W)
            img = torch.from_numpy(np.transpose(img, (2, 0, 1)))

        # Return dictionary
        sample = {"BraTS21ID": self.ids[idx], "image": img}

        # Add target if available
        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            sample["target"] = target

        return sample


def get_transforms(phase: str):
    """
    Returns the Albumentations transformation pipeline.
    Implements Spatially-Preserved Augmentation for training.

    Args:
        phase (str): 'train', 'valid', or 'test'.
    """
    if phase == "train":
        return A.Compose(
            [
                # Spatially-Preserved Augmentations
                # Rotations and Flips (Anatomical anchoring preserved)
                A.Rotate(limit=15, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Deformations (Texture/Shape variance without translation)
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                A.GridDistortion(p=0.5),
                # Convert to Tensor (HWC -> CHW)
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: No augmentation, just Tensor conversion
        return A.Compose([ToTensorV2()])


def get_dataloader(
    metadata_df: pd.DataFrame,
    phase: str,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    load_cached_data: bool = True,
):
    """
    Creates a DataLoader for the given phase.
    Handles data processing, caching, and dataset instantiation.

    Args:
        metadata_df (pd.DataFrame): Metadata dataframe containing paths and labels.
        phase (str): 'train', 'val', or 'test'. Used for cache naming and transform selection.
        batch_size (int): Batch size.
        num_workers (int): Number of workers for DataLoader.
        load_cached_data (bool): Whether to attempt loading from .npy cache.

    Returns:
        DataLoader: Configured DataLoader instance.
    """
    # 1. Process or Load Data from Cache
    # data_processing.process_dataset returns tuple: (ids, images, [labels])
    data_tuple = data_processing.process_dataset(
        metadata_df, dataset_name=phase, load_cached_data=load_cached_data
    )

    # Unpack based on return length (labels are optional)
    if len(data_tuple) == 3:
        ids, images, targets = data_tuple
    else:
        ids, images = data_tuple
        targets = None

    # 2. Define Transforms
    # Map 'val' or 'test' to validation transforms (no aug)
    transform_phase = "train" if phase == "train" else "valid"
    transforms = get_transforms(transform_phase)

    # 3. Create Dataset
    dataset = MGMTDataset(ids=ids, images=images, targets=targets, transform=transforms)

    # 4. Create DataLoader
    # Shuffle only for training
    shuffle = phase == "train"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=(phase == "train"),  # Drop last incomplete batch during training
    )

    return loader
