import torch
import numpy as np
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import load_data


class SaltDataset(Dataset):
    """
    Dataset class for Salt Segmentation.

    Implements the Input Channel Multiplexing strategy:
    Constructs a 3-channel input tensor [Seismic, Seismic, Depth] to leverage
    ImageNet-pretrained encoders while incorporating depth information.
    """

    def __init__(self, mode="train", transform=None):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transforms. If None, uses defaults for train.
        """
        self.mode = mode

        # Load data using the library utility
        # This handles caching, reading from disk, and padding images/masks to 128x128.
        # images: (N, 128, 128) uint8
        # masks: (N, 128, 128) float32 (0.0 or 1.0) or None
        # depths: (N,) float32
        # ids: (N,) str
        self.images, self.masks, self.depths, self.ids = load_data(mode=mode)

        self.transform = transform
        if self.transform is None and mode == "train":
            # Conservative augmentation strategy for seismic data
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                ]
            )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # 1. Retrieve Raw Data
        img = self.images[idx]  # (128, 128) uint8
        depth = self.depths[idx]  # scalar float32
        id_ = self.ids[idx]  # str

        mask = None
        if self.masks is not None:
            mask = self.masks[idx]  # (128, 128) float32

        # 2. Apply Augmentations
        # Note: Augmentations are applied before channel multiplexing and normalization
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=img)
                img = augmented["image"]

        # 3. Normalization & Channel Multiplexing
        # Normalize Image: [0, 255] -> [0, 1]
        img = img.astype(np.float32) / 255.0

        # Normalize Depth: [0, ~1000] -> [0, 1]
        # We use a fixed scale of 1000.0 based on dataset analysis (max depth ~960)
        # to ensure consistency across train/val/test splits.
        d = (depth - 0.0) / 1000.0

        # Create Tensors
        # Image Channel: (1, H, W)
        img_t = torch.tensor(img).float().unsqueeze(0)

        # Depth Channel: (1, H, W)
        # Create a constant spatial channel filled with the normalized depth value
        depth_t = torch.full_like(img_t, d)

        # Stack Channels: [Seismic, Seismic, Depth] -> (3, H, W)
        # This 3-channel input is compatible with standard ResNet/ResNeXt encoders
        input_tensor = torch.cat([img_t, img_t, depth_t], dim=0)

        # 4. Return Data
        if self.mode == "test":
            return input_tensor, id_

        # Mask: (H, W)
        mask_t = torch.tensor(mask).float()

        return input_tensor, mask_t, id_


def get_dataloader(
    mode, batch_size=Config.BATCH_SIZE, shuffle=None, num_workers=Config.NUM_WORKERS
):
    """
    Factory function to create DataLoaders with appropriate configuration.

    Args:
        mode (str): 'train', 'val', or 'test'.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle. Defaults to True for train, False otherwise.
        num_workers (int): Number of worker threads.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    if shuffle is None:
        shuffle = mode == "train"

    ds = SaltDataset(mode=mode)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        # Drop last incomplete batch during training to maintain stable batch statistics
        drop_last=(mode == "train"),
    )
