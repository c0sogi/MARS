import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import io
import pandas as pd
import numpy as np
import torchvision.transforms as T
from library.config import Config
from library.utils import read_bson_images, HierarchyManager


class BSONDataset(Dataset):
    """
    PyTorch Dataset for the Cdiscount Image Categorization task.
    Reads images directly from BSON files using a metadata index.
    Handles variable number of images (1-4) by padding to a fixed size.
    """

    def __init__(
        self, metadata_path, bson_path, split="train", transform=None, limit_size=None
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            bson_path (str): Path to the binary BSON file.
            split (str): 'train', 'val', or 'test'. Determines augmentations and label handling.
            transform (callable, optional): Optional transform to be applied on a sample.
            limit_size (int, optional): If provided, limits the dataset to this many samples.
        """
        self.metadata_path = metadata_path
        self.bson_path = bson_path
        self.split = split
        self.limit_size = limit_size

        # Load Metadata
        self.metadata = pd.read_csv(metadata_path)

        # Apply limit if requested
        if self.limit_size is not None:
            self.metadata = self.metadata.iloc[: self.limit_size].reset_index(drop=True)

        # Initialize HierarchyManager for label mapping (only needed for train/val usually,
        # but useful to have consistent state)
        self.hierarchy_manager = HierarchyManager(load_cached_data=True)

        # File handle (lazy loading)
        self.file_handle = None

        # Define Transforms
        # Analysis Stats: Mean=[0.7837, 0.7692, 0.7583], Std=[0.3131, 0.3212, 0.3300]
        mean = [0.7837, 0.7692, 0.7583]
        std = [0.3131, 0.3212, 0.3300]

        if transform is None:
            if self.split == "train":
                self.transform = T.Compose(
                    [
                        T.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize(mean=mean, std=std),
                    ]
                )
            else:
                self.transform = T.Compose(
                    [
                        T.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
                        T.ToTensor(),
                        T.Normalize(mean=mean, std=std),
                    ]
                )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Lazy open file handle to support multiprocessing
        if self.file_handle is None:
            self.file_handle = open(self.bson_path, "rb")

        row = self.metadata.iloc[idx]

        # Read raw image bytes
        # row keys: sample_id, bson_file_path, bson_offset, bson_length, [category_id]
        offset = int(row["bson_offset"])
        length = int(row["bson_length"])

        img_bytes_list = read_bson_images(self.file_handle, offset, length)

        # Process images
        processed_imgs = []
        for b_data in img_bytes_list:
            try:
                img = Image.open(io.BytesIO(b_data))
                img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img)
                processed_imgs.append(img)
            except Exception:
                # Handle corrupt images by creating a blank one (rare case)
                blank = torch.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE))
                processed_imgs.append(blank)

        # Handle variable number of images (Pad to 4)
        # Max images per product is 4 based on dataset description
        max_imgs = 4
        current_count = len(processed_imgs)

        # Stack valid images
        if current_count > 0:
            imgs_tensor = torch.stack(processed_imgs)
        else:
            # Fallback for empty product (should not happen)
            imgs_tensor = torch.zeros((1, 3, Config.IMG_SIZE, Config.IMG_SIZE))
            current_count = 1

        # Pad if necessary
        if current_count < max_imgs:
            pad_size = max_imgs - current_count
            pad_tensor = torch.zeros((pad_size, 3, Config.IMG_SIZE, Config.IMG_SIZE))
            imgs_tensor = torch.cat([imgs_tensor, pad_tensor], dim=0)
        elif current_count > max_imgs:
            # Truncate if somehow we get more than 4
            imgs_tensor = imgs_tensor[:max_imgs]
            current_count = max_imgs

        # Create mask (1 for valid image, 0 for padding)
        mask = torch.zeros(max_imgs, dtype=torch.float32)
        mask[:current_count] = 1.0

        # Prepare output dict
        sample_id = int(row["sample_id"])
        output = {
            "images": imgs_tensor,  # Shape: (4, 3, 180, 180)
            "mask": mask,  # Shape: (4,)
            "sample_id": sample_id,
        }

        # Get Label if available
        if self.split != "test" and "category_id" in row:
            raw_cat_id = int(row["category_id"])
            class_idx = self.hierarchy_manager.category_id_to_class_idx(raw_cat_id)
            output["target"] = torch.tensor(class_idx, dtype=torch.long)

        return output

    def __del__(self):
        # Close file handle when dataset is destroyed
        if self.file_handle is not None:
            self.file_handle.close()
