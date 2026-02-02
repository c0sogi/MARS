import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import (
    IMG_SIZE,
    AUG_ROTATION_LIMIT,
    AUG_ELASTIC_ALPHA,
    AUG_GRID_DISTORT_LIMIT,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)
from library.data_processing import load_data
from library.utils import set_seed


class SICAVDataset(Dataset):
    """
    PyTorch Dataset for the SICAV Network strategy.

    Expects input images to be pre-processed into (H, W, 9) tensors representing:
    - Channels 0-2: FLAIR, T1wCE, T2w at 40% Relative Depth
    - Channels 3-5: FLAIR, T1wCE, T2w at 50% Relative Depth (Centroid)
    - Channels 6-8: FLAIR, T1wCE, T2w at 60% Relative Depth

    Strictly preserves spatial alignment by avoiding translation/scaling augmentations.
    """

    def __init__(self, ids, images, labels, transforms=None):
        """
        Args:
            ids (np.array): Array of BraTS21IDs.
            images (np.array): Array of shape (N, H, W, 9).
            labels (np.array): Array of shape (N,).
            transforms (albumentations.Compose, optional): Augmentation pipeline.
        """
        self.ids = ids
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Retrieve data
        # Image is (224, 224, 9), float32, range [0, 1]
        image = self.images[idx]
        label = self.labels[idx]

        # Apply Augmentations
        if self.transforms:
            # Albumentations works with (H, W, C)
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # If no transforms provided, just convert to tensor manually
            # Transpose to (C, H, W)
            image = torch.from_numpy(image.transpose(2, 0, 1))

        # Ensure label is float tensor for BCEWithLogitsLoss
        label = torch.tensor(label, dtype=torch.float32)

        return image, label


def get_transforms(phase: str):
    """
    Generates the augmentation pipeline based on the SICAV strategy.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        albumentations.Compose: The transform pipeline.
    """
    if phase == "train":
        return A.Compose(
            [
                # Spatially-Preserved Augmentations
                # We strictly EXCLUDE ShiftScaleRotate with shift/scale > 0
                # to preserve the centroid alignment.
                # Elastic Transform
                # Note: sigma=50, alpha_affine=50 are standard defaults for medical imaging
                # when alpha is small. Using config value for alpha.
                A.ElasticTransform(
                    alpha=AUG_ELASTIC_ALPHA, sigma=50, alpha_affine=50, p=0.5
                ),
                # Grid Distortion
                A.GridDistortion(
                    num_steps=5, distort_limit=AUG_GRID_DISTORT_LIMIT, p=0.5
                ),
                # Rotation (without shifting/scaling)
                A.Rotate(
                    limit=AUG_ROTATION_LIMIT,
                    p=0.5,
                    border_mode=0,  # Constant padding (0)
                ),
                # Flips
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Convert to Tensor and Transpose to (C, H, W)
                ToTensorV2(),
            ]
        )
    else:
        # For validation/test, just convert to tensor
        return A.Compose([ToTensorV2()])


def get_dataloaders(
    load_cached_data=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
):
    """
    Loads data using the data_processing module and creates PyTorch DataLoaders.

    Args:
        load_cached_data (bool): Whether to use cached numpy arrays.
        batch_size (int): Batch size for training/inference.
        num_workers (int): Number of worker threads.

    Returns:
        dict: Dictionary containing 'train', 'val', 'test' DataLoaders.
    """
    # Ensure reproducibility
    set_seed(SEED)

    loaders = {}

    # Define splits to process
    splits = ["train", "val", "test"]

    for split in splits:
        # Load raw numpy arrays (cached or processed from scratch)
        ids, images, labels = load_data(split, load_cached_data=load_cached_data)

        # Determine transform phase
        phase = split if split == "train" else "val"
        transforms = get_transforms(phase)

        # Create Dataset
        dataset = SICAVDataset(ids, images, labels, transforms=transforms)

        # Create DataLoader
        shuffle = split == "train"

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),  # Drop last incomplete batch only in training
        )

        loaders[split] = loader

        print(f"Created {split} DataLoader with {len(dataset)} samples.")

    return loaders
