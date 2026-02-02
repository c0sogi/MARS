import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from library.config import Config
from library.utils import get_label_map, parse_labels, get_transforms


class KuzushijiDataset(Dataset):
    """
    PyTorch Dataset for Kuzushiji Character Recognition.
    Handles loading images, parsing labels, and applying transformations.
    """

    def __init__(self, split, config=None, transforms=None, load_cached_data=True):
        """
        Args:
            split (str): One of 'train', 'val', 'test'.
            config (Config): Configuration object.
            transforms (callable, optional): Optional transform to be applied on a sample.
            load_cached_data (bool): Whether to load metadata from cache (Parquet) if available.
        """
        self.split = split
        self.config = config if config is not None else Config()
        self.transforms = transforms

        # Ensure working directory exists for caching
        os.makedirs(self.config.WORKING_DIR, exist_ok=True)

        # Load metadata with caching mechanism
        self.df = self._load_data(split, load_cached_data)

        # Load label map
        self.label_map, _ = get_label_map(self.config)

    def _load_data(self, split, load_cached_data):
        """
        Loads metadata from CSV or Parquet cache.
        """
        cache_path = os.path.join(self.config.WORKING_DIR, f"{split}_data.parquet")

        if load_cached_data and os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                return df
            except Exception as e:
                print(
                    f"Failed to load cache from {cache_path}: {e}. Reloading from CSV."
                )

        # Determine CSV path based on split
        if split == "train":
            csv_path = self.config.TRAIN_CSV
        elif split == "val":
            csv_path = self.config.VAL_CSV
        elif split == "test":
            csv_path = self.config.TEST_CSV
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train', 'val', or 'test'."
            )

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Save to cache
        try:
            df.to_parquet(cache_path, index=False)
        except Exception as e:
            print(f"Warning: Failed to save cache to {cache_path}: {e}")

        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]

        # Construct full image path
        # Metadata file_path is relative to input dir (e.g., "train_images/id.jpg")
        # Config INPUT_DIR is "./input"
        # We need to handle potential double joining if paths are already absolute or strictly relative
        # Based on metadata generation: file_path is "train_images/..."
        image_path = os.path.join(self.config.INPUT_DIR, row["file_path"])

        # Load Image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image or handle error appropriately
            # For strict training, we might raise, but here we'll try to fail gracefully if possible
            # or just raise to stop bad training.
            raise e

        w, h = image.size

        # Parse Labels
        # parse_labels returns (boxes, labels) tensors
        # It handles empty strings for test set or unlabeled images
        label_str = row.get("labels", "")
        boxes, labels = parse_labels(label_str, self.label_map)

        target = {}
        target["image_id"] = image_id  # Keep as string for submission mapping

        # If we have boxes (Train/Val), we need to process them
        if boxes.shape[0] > 0:
            # Geometric Sanitization: Clip boxes to image boundaries
            # boxes format is [x1, y1, x2, y2]
            boxes[:, 0] = boxes[:, 0].clamp(min=0, max=w)
            boxes[:, 1] = boxes[:, 1].clamp(min=0, max=h)
            boxes[:, 2] = boxes[:, 2].clamp(min=0, max=w)
            boxes[:, 3] = boxes[:, 3].clamp(min=0, max=h)

            # Filter out invalid boxes (area <= 0)
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[keep]
            labels = labels[keep]

            target["boxes"] = boxes
            target["labels"] = labels
            target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            target["iscrowd"] = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        else:
            # Handle images with no labels (or test set)
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        # Apply Transforms
        if self.transforms:
            # transforms usually expect PIL image and return Tensor
            image = self.transforms(image)

        return image, target
