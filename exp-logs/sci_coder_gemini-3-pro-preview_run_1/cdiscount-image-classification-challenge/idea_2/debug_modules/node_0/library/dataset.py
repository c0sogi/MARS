import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from library.config import Config
from library.utils import read_bson_images, CategoryHierarchy


class BSONProductDataset(Dataset):
    """
    Custom Dataset for loading product images from BSON files using metadata indices.
    Returns images padded to a fixed sequence length and hierarchical labels.
    """

    def __init__(self, mode="train", transform=None):
        """
        Args:
            mode (str): One of 'train', 'val', 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.mode = mode
        self.transform = transform

        # Load Metadata
        if self.mode == "train":
            self.metadata_path = Config.TRAIN_METADATA
            self.bson_path = Config.TRAIN_BSON
            self.has_labels = True
        elif self.mode == "val":
            self.metadata_path = Config.VAL_METADATA
            self.bson_path = Config.TRAIN_BSON
            self.has_labels = True
        elif self.mode == "test":
            self.metadata_path = Config.TEST_METADATA
            self.bson_path = Config.TEST_BSON
            self.has_labels = False
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Load the metadata CSV
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        self.df = pd.read_csv(self.metadata_path)

        # Debugging: Limit size if configured
        if Config.DEBUG:
            self.df = self.df.head(1000)

        # Initialize Hierarchy Mapper if needed
        if self.has_labels:
            self.hierarchy = CategoryHierarchy(load_cached_data=True)

        # Default Transforms if none provided
        # Using statistics from Data Analysis:
        # Mean: R=0.7837, G=0.7692, B=0.7583
        # Std:  R=0.3131, G=0.3212, B=0.3300
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.7837, 0.7692, 0.7583], std=[0.3131, 0.3212, 0.3300]
                    ),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Images
        offset = row["bson_offset"]
        length = row["bson_length"]

        # Use utility to read from disk
        # Note: read_bson_images returns a list of PIL Images
        pil_images = read_bson_images(self.bson_path, offset, length)

        # Handle case where no images found (should be rare/impossible with valid data)
        if not pil_images:
            # Create a black image as fallback
            pil_images = [Image.new("RGB", (Config.IMG_SIZE, Config.IMG_SIZE))]

        # 2. Apply Transforms
        img_tensors = []
        for img in pil_images:
            if self.transform:
                img_tensors.append(self.transform(img))
            else:
                img_tensors.append(transforms.ToTensor()(img))

        # 3. Get Identifiers and Labels
        sample_id = int(row["sample_id"])

        labels = {}
        if self.has_labels:
            cat_id = int(row["category_id"])
            l1, l2, l3 = self.hierarchy.get_hierarchy_indices(cat_id)
            labels = {
                "target": torch.tensor(l3, dtype=torch.long),
                "l1": torch.tensor(l1, dtype=torch.long),
                "l2": torch.tensor(l2, dtype=torch.long),
                "original_category_id": cat_id,
            }
        else:
            # Dummy labels for test set
            labels = {
                "target": torch.tensor(-1, dtype=torch.long),
                "l1": torch.tensor(-1, dtype=torch.long),
                "l2": torch.tensor(-1, dtype=torch.long),
                "original_category_id": -1,
            }

        return {
            "id": sample_id,
            "images": img_tensors,  # List of 3D tensors
            "labels": labels,
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable number of images per product.
    Pads the image list to a fixed size (MAX_IMAGES=4).

    Returns:
        dict: {
            'ids': LongTensor (B,),
            'images': FloatTensor (B, 4, 3, H, W),
            'mask': BoolTensor (B, 4), # True if image is real, False if padding
            'labels': {
                'target': LongTensor (B,),
                'l1': LongTensor (B,),
                'l2': LongTensor (B,)
            }
        }
    """
    MAX_IMAGES = 4

    batch_size = len(batch)

    # Prepare containers
    ids = []

    # Image tensor: (B, MAX_IMAGES, C, H, W)
    # We assume C=3, H=180, W=180 based on Config and Transforms
    c, h, w = batch[0]["images"][0].shape
    batched_images = torch.zeros(batch_size, MAX_IMAGES, c, h, w)
    batched_mask = torch.zeros(batch_size, MAX_IMAGES, dtype=torch.bool)

    # Label containers
    targets = []
    l1s = []
    l2s = []

    for i, item in enumerate(batch):
        ids.append(item["id"])

        # Process Images
        imgs = item["images"]
        num_imgs = min(len(imgs), MAX_IMAGES)

        for j in range(num_imgs):
            batched_images[i, j] = imgs[j]
            batched_mask[i, j] = True

        # If fewer than MAX_IMAGES, the rest remains 0 (padding)
        # If more than MAX_IMAGES, we truncated (though dataset says max 4)

        # Process Labels
        targets.append(item["labels"]["target"])
        l1s.append(item["labels"]["l1"])
        l2s.append(item["labels"]["l2"])

    return {
        "ids": torch.tensor(ids, dtype=torch.long),
        "images": batched_images,
        "mask": batched_mask,
        "labels": {
            "target": torch.stack(targets),
            "l1": torch.stack(l1s),
            "l2": torch.stack(l2s),
        },
    }
