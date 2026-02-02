import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library import config, utils


class SaltDataset(Dataset):
    """
    PyTorch Dataset for Salt Segmentation Task.
    Handles images, masks (optional), and depth values.
    """

    def __init__(
        self,
        images,
        masks=None,
        depths=None,
        ids=None,
        transform=None,
        depth_mean=0.0,
        depth_std=1.0,
    ):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, 1).
            masks (np.ndarray, optional): Array of masks (N, H, W, 1).
            depths (np.ndarray, optional): Array of depths (N,).
            ids (np.ndarray, optional): Array of IDs.
            transform (albumentations.Compose): Augmentation pipeline.
            depth_mean (float): Mean depth for standardization.
            depth_std (float): Std dev depth for standardization.
        """
        self.images = images
        self.masks = masks
        self.depths = depths
        self.ids = ids
        self.transform = transform
        self.depth_mean = depth_mean
        self.depth_std = depth_std

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image
        image = self.images[idx]  # (H, W, 1)

        # Prepare data dictionary for Albumentations
        data = {"image": image}
        if self.masks is not None:
            data["mask"] = self.masks[idx]  # (H, W, 1)

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(**data)
            image = augmented["image"]
            if self.masks is not None:
                mask = augmented["mask"]
        else:
            # Fallback conversion to tensor if no transform provided
            # (Though get_dataloaders should always provide one)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            if self.masks is not None:
                mask = torch.from_numpy(self.masks[idx].transpose(2, 0, 1)).float()

        # Handle Depth (Standardization)
        # Always return a tensor for depth, even if dummy
        if self.depths is not None:
            z = self.depths[idx]
            z = (z - self.depth_mean) / self.depth_std
            z_tensor = torch.tensor([z], dtype=torch.float32)
        else:
            z_tensor = torch.tensor([0.0], dtype=torch.float32)

        # Handle ID
        img_id = self.ids[idx] if self.ids is not None else ""

        if self.masks is not None:
            return image, mask, z_tensor, img_id

        return image, z_tensor, img_id


def get_transforms(phase):
    """
    Returns Albumentations transforms for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.
    """
    # Standard ImageNet statistics for normalization
    # Since input is 1-channel (grayscale), we use the mean of the RGB means
    # or just the first channel stats. Using standard RGB stats is common practice
    # even for grayscale when using pretrained backbones.
    mean = (0.485,)
    std = (0.229,)

    if phase == "train":
        return A.Compose(
            [
                # Non-Rigid Augmentation: Elastic Transform
                A.ElasticTransform(
                    alpha=config.ELASTIC_ALPHA,
                    sigma=config.ELASTIC_SIGMA,
                    alpha_affine=config.ELASTIC_ALPHA_AFFINE,
                    p=config.AUG_PROB,
                ),
                # Rigid Augmentation: Shift, Scale, Rotate
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=config.AUG_PROB,
                ),
                # Horizontal Flip
                A.HorizontalFlip(p=0.5),
                # Normalization and Tensor Conversion
                A.Normalize(mean=mean, std=std),
                ToTensorV2(transpose_mask=True),
            ]
        )
    else:
        # Validation / Test: Only Normalize and ToTensor
        return A.Compose(
            [A.Normalize(mean=mean, std=std), ToTensorV2(transpose_mask=True)]
        )


def get_dataloaders(
    load_cached_data=True, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS
):
    """
    Loads data, creates datasets, and returns dataloaders for train, val, and test.

    Args:
        load_cached_data (bool): Whether to use cached .npy files.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Load Data
    # Train
    train_data = utils.load_dataset_data(
        config.TRAIN_METADATA_PATH,
        config.CACHE_TRAIN_IMAGES,
        config.CACHE_TRAIN_MASKS,
        config.CACHE_TRAIN_DEPTHS,
        config.CACHE_TRAIN_IDS,
        load_cached_data=load_cached_data,
    )

    # Val
    val_data = utils.load_dataset_data(
        config.VAL_METADATA_PATH,
        config.CACHE_VAL_IMAGES,
        config.CACHE_VAL_MASKS,
        config.CACHE_VAL_DEPTHS,
        config.CACHE_VAL_IDS,
        load_cached_data=load_cached_data,
    )

    # Test
    test_data = utils.load_dataset_data(
        config.TEST_METADATA_PATH,
        config.CACHE_TEST_IMAGES,
        None,  # No masks for test
        config.CACHE_TEST_DEPTHS,
        config.CACHE_TEST_IDS,
        load_cached_data=load_cached_data,
    )

    # 2. Calculate Depth Statistics from Training Set
    # We use these to normalize depths across all splits
    train_depths = train_data["depths"]
    depth_mean = np.mean(train_depths)
    depth_std = np.std(train_depths)

    # Avoid division by zero
    if depth_std == 0:
        depth_std = 1.0

    # 3. Create Datasets
    train_dataset = SaltDataset(
        images=train_data["images"],
        masks=train_data["masks"],
        depths=train_data["depths"],
        ids=train_data["ids"],
        transform=get_transforms("train"),
        depth_mean=depth_mean,
        depth_std=depth_std,
    )

    val_dataset = SaltDataset(
        images=val_data["images"],
        masks=val_data["masks"],
        depths=val_data["depths"],
        ids=val_data["ids"],
        transform=get_transforms("val"),
        depth_mean=depth_mean,
        depth_std=depth_std,
    )

    test_dataset = SaltDataset(
        images=test_data["images"],
        masks=None,
        depths=test_data["depths"],
        ids=test_data["ids"],
        transform=get_transforms("test"),
        depth_mean=depth_mean,
        depth_std=depth_std,
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
