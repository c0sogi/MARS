import os
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config
from library.utils import read_bson_images


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline.

    Args:
        mode (str): 'train' or 'val'/'test'.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class CdiscountDataset(Dataset):
    """
    PyTorch Dataset for the Cdiscount Product Categorization task.
    Reads images dynamically from BSON files using metadata offsets.
    """

    def __init__(self, metadata_path, mode="train", transform=None):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file (train.csv, val.csv, or test.csv).
            mode (str): 'train', 'val', or 'test'. Used to determine label handling and default transforms.
            transform (A.Compose, optional): Custom transform pipeline. If None, uses default based on mode.
        """
        self.metadata_path = metadata_path
        self.mode = mode

        # Load Metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        self.meta = pd.read_csv(metadata_path)

        # Load Category Mapping
        # Ensure consistent mapping from category_id (int) to class index (0..N-1)
        if not os.path.exists(Config.CATEGORY_NAMES):
            raise FileNotFoundError(
                f"Category names file not found: {Config.CATEGORY_NAMES}"
            )

        cats = pd.read_csv(Config.CATEGORY_NAMES)
        # Sort to ensure deterministic order
        cats = cats.sort_values("category_id").reset_index(drop=True)

        # Create mappings
        self.id_to_idx = {
            cat_id: idx for idx, cat_id in enumerate(cats["category_id"].values)
        }

        # Set Transforms
        if transform is None:
            self.transform = get_transforms(mode)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]

        # Construct absolute path to BSON file
        # row['file_path'] is relative to input dir (e.g., "train.bson")
        bson_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read images from BSON
        # Returns a list of RGB numpy arrays
        images = read_bson_images(bson_path, row["bson_offset"], row["bson_length"])

        # Robustness: Handle empty image list (though unlikely with valid metadata)
        if not images:
            # Create a black placeholder image
            images = [np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)]

        # Apply Transforms
        processed_images = []
        for img in images:
            # Albumentations requires named argument 'image'
            augmented = self.transform(image=img)
            processed_images.append(augmented["image"])

        # Stack images into a single tensor: (N_images, C, H, W)
        images_tensor = torch.stack(processed_images)

        # Process Label
        target = -1
        category_id = row["category_id"]

        if self.mode != "test":
            if not pd.isna(category_id):
                # Map original category_id to 0-indexed class label
                cat_id_int = int(category_id)
                target = self.id_to_idx.get(cat_id_int, -1)

        # Get Product ID (useful for aggregation during inference)
        product_id = int(row["product_id"])

        return images_tensor, target, product_id


def collate_flatten(batch):
    """
    Custom collate function to flatten the variable number of images per product.

    Args:
        batch: List of tuples (images_tensor, target, product_id)
               - images_tensor: (N, C, H, W)
               - target: int
               - product_id: int

    Returns:
        flat_images: Tensor of shape (Total_Images, C, H, W)
        flat_targets: Tensor of shape (Total_Images,)
        flat_product_ids: Tensor of shape (Total_Images,)
    """
    images_list = []
    targets_list = []
    product_ids_list = []

    for images, target, pid in batch:
        num_imgs = images.shape[0]

        # Append images
        images_list.append(images)

        # Repeat target and product_id for each image
        targets_list.extend([target] * num_imgs)
        product_ids_list.extend([pid] * num_imgs)

    # Concatenate all images along the batch dimension
    flat_images = torch.cat(images_list, dim=0)

    # Convert lists to tensors
    flat_targets = torch.tensor(targets_list, dtype=torch.long)
    flat_product_ids = torch.tensor(product_ids_list, dtype=torch.long)

    return flat_images, flat_targets, flat_product_ids
