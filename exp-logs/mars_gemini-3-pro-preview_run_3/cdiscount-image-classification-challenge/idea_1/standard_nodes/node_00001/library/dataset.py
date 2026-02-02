import os
import torch
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from library.config import IMG_SIZE, SEED
from library.utils import extract_images_from_bson, get_category_mapping

# Set seeds for reproducibility where possible
torch.manual_seed(SEED)
np.random.seed(SEED)


class CdiscountDataset(Dataset):
    """
    Custom Dataset for Cdiscount Product Categorization.
    Reads directly from BSON files using metadata offsets to minimize memory usage.
    """

    def __init__(self, metadata_path, bson_path, mode="train", transform=None):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file (train.csv, val.csv, or test.csv).
            bson_path (str): Path to the source BSON file (train.bson or test.bson).
            mode (str): Operation mode - 'train', 'val', or 'test'.
            transform (callable, optional): PyTorch transforms to be applied on the image.
        """
        self.metadata = pd.read_csv(metadata_path)
        self.bson_path = bson_path
        self.mode = mode
        self.transform = transform
        self.file_handle = None

        # Load category mapping for training and validation to convert global IDs to 0-N indices
        if self.mode in ["train", "val"]:
            self.id_to_idx, _ = get_category_mapping(load_cached_data=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Lazy initialization of file handle ensures safety with multiple DataLoader workers
        if self.file_handle is None:
            self.file_handle = open(self.bson_path, "rb")

        # Retrieve record metadata
        row = self.metadata.iloc[idx]
        offset = row["bson_offset"]
        length = row["bson_length"]

        # Seek to the specific record and read raw BSON data
        self.file_handle.seek(offset)
        bson_data = self.file_handle.read(length)

        # Parse BSON to extract raw image byte strings
        img_bytes_list = extract_images_from_bson(bson_data)

        # Decode images from bytes to numpy arrays
        images = []
        for img_bytes in img_bytes_list:
            # Decode using OpenCV
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is not None:
                # Convert BGR (OpenCV default) to RGB (PyTorch/PIL default)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)

        # Handle rare edge case where no images are found or decoding fails
        if len(images) == 0:
            # Return a blank black image
            images = [np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)]

        # --- Mode Specific Logic ---

        if self.mode == "train":
            # Instance-level flattening: Randomly select ONE image from the product's list
            # This allows the model to see different views across epochs
            img = images[np.random.randint(len(images))]

            if self.transform:
                img = self.transform(img)

            # Map category_id to class index
            cat_id = int(row["category_id"])
            label = self.id_to_idx.get(
                cat_id, 0
            )  # Default to 0 if not found (should not happen)

            return img, label

        elif self.mode == "val":
            # Deterministic selection: Always select the FIRST image
            # Ensures validation metrics are consistent
            img = images[0]

            if self.transform:
                img = self.transform(img)

            cat_id = int(row["category_id"])
            label = self.id_to_idx.get(cat_id, 0)

            return img, label

        elif self.mode == "test":
            # Inference: Return ALL images as a stack
            # This allows the model to average predictions (logits) across all views
            img_stack = []
            for img in images:
                if self.transform:
                    img_t = self.transform(img)
                    img_stack.append(img_t)
                else:
                    # Ensure it is a tensor even if no transform provided
                    img_stack.append(transforms.functional.to_tensor(img))

            # Stack shape: (Num_Images, Channels, Height, Width)
            img_stack = torch.stack(img_stack)

            # Return product ID to map predictions back to the submission format
            product_id = int(row["_id"])

            return img_stack, product_id

        else:
            raise ValueError(f"Unknown mode: {self.mode}")


def get_transforms(mode="train", img_size=IMG_SIZE):
    """
    Factory function to get the appropriate transform pipeline.
    """
    # Standard ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if mode == "train":
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        # Validation and Test (Deterministic)
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )
