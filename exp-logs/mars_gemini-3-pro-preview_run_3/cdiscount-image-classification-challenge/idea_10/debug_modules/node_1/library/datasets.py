import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

from library.config import Config
from library.utils import BSONLoader, HierarchyMapper


def get_default_transform(image_size=224):
    """
    Returns the standard image transformation pipeline for pre-trained models.
    Converts images to RGB, Resizes, Normalizes (ImageNet stats), and converts to Tensor.
    """
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


class RawImageDataset(Dataset):
    """
    Dataset for reading raw images from BSON files based on metadata indices.
    Used primarily for the Feature Extraction phase.
    """

    def __init__(self, metadata_path, bson_path, transform=None, subset_size=None):
        """
        Args:
            metadata_path (str): Path to the CSV file containing _id, bson_offset, etc.
            bson_path (str): Path to the .bson file containing raw image data.
            transform (albumentations.Compose, optional): Image transformations.
            subset_size (int, optional): If provided, limits the dataset size for debugging.
        """
        self.meta = pd.read_csv(metadata_path)
        if subset_size is not None:
            self.meta = self.meta.iloc[:subset_size]

        self.bson_loader = BSONLoader(bson_path)
        self.transform = transform if transform is not None else get_default_transform()

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        offset = row["bson_offset"]
        length = row["bson_length"]
        _id = int(row["_id"])

        # Get label if available (Train/Val), else -1 (Test)
        category_id = int(row["category_id"]) if "category_id" in row else -1

        # Read images from BSON
        # BSONLoader returns a list of BGR numpy images
        images = self.bson_loader.read_images(offset, length)

        # Handle edge case: Product with no valid images
        if len(images) == 0:
            # Create a black placeholder image (180x180 is approx mean size)
            images = [np.zeros((180, 180, 3), dtype=np.uint8)]

        transformed_images = []
        for img in images:
            # Convert BGR (cv2 default) to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Apply transforms
            if self.transform:
                augmented = self.transform(image=img_rgb)
                transformed_images.append(augmented["image"])
            else:
                # Fallback to simple tensor conversion
                transformed_images.append(
                    torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
                )

        # Stack images into a single tensor: (Num_Images, Channels, Height, Width)
        # Note: Num_Images varies between 1 and 4.
        # The DataLoader collate_fn must handle this if batch_size > 1.
        images_tensor = torch.stack(transformed_images)

        return images_tensor, _id, category_id


class FeatureDataset(Dataset):
    """
    Dataset for loading pre-computed feature vectors and hierarchical labels.
    Used for Training the MLP head and generating Inference predictions.
    """

    def __init__(
        self,
        features_path,
        labels_path=None,
        ids_path=None,
        hierarchy_mapper=None,
        mode="train",
        mmap_mode="r",
        subset_size=None,
    ):
        """
        Args:
            features_path (str): Path to the .npy file containing feature vectors.
            labels_path (str, optional): Path to .npy file with category_ids (for train/val).
            ids_path (str, optional): Path to .npy file with product _ids (for test).
            hierarchy_mapper (HierarchyMapper, optional): Instance to map category_id to levels.
            mode (str): 'train', 'val', or 'test'. Determines return values.
            mmap_mode (str): Numpy mmap mode ('r' for read-only) to save RAM.
            subset_size (int, optional): Limit dataset size for debugging.
        """
        self.mode = mode

        # Load features with memory mapping to handle large files (>RAM)
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")

        self.features = np.load(features_path, mmap_mode=mmap_mode)

        self.labels = None
        self.ids = None
        self.hierarchy_mapper = hierarchy_mapper

        # Handle Labels (Train/Val)
        if mode in ["train", "val"]:
            if labels_path is None:
                raise ValueError("labels_path is required for train/val mode")
            if not os.path.exists(labels_path):
                raise FileNotFoundError(f"Labels file not found: {labels_path}")

            self.labels = np.load(labels_path, mmap_mode=mmap_mode)

            # Initialize mapper if not provided
            if self.hierarchy_mapper is None:
                self.hierarchy_mapper = HierarchyMapper()
                self.hierarchy_mapper.process()

        # Handle IDs (Test)
        elif mode == "test":
            if ids_path is None:
                raise ValueError("ids_path is required for test mode")
            if not os.path.exists(ids_path):
                raise FileNotFoundError(f"IDs file not found: {ids_path}")

            self.ids = np.load(ids_path, mmap_mode=mmap_mode)

        # Apply subsetting if requested (e.g. for debugging)
        if subset_size is not None:
            self.features = self.features[:subset_size]
            if self.labels is not None:
                self.labels = self.labels[:subset_size]
            if self.ids is not None:
                self.ids = self.ids[:subset_size]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Load feature vector
        # .copy() is essential here when using mmap to decouple from the file handle
        # and ensure we have a writable, contiguous array in memory for PyTorch.
        feature = torch.from_numpy(self.features[idx].copy())

        if self.mode in ["train", "val"]:
            # Get raw category ID
            category_id = int(self.labels[idx])

            # Map to hierarchical targets
            l1, l2, l3 = self.hierarchy_mapper.get_labels(category_id)

            # Safety check for unknown categories (should not happen in clean data)
            if l3 is None:
                l1, l2, l3 = 0, 0, 0

            # Return feature and all 3 levels of labels
            return (
                feature,
                torch.tensor(l1, dtype=torch.long),
                torch.tensor(l2, dtype=torch.long),
                torch.tensor(l3, dtype=torch.long),
            )

        else:
            # Test mode: Return feature and product ID for submission mapping
            _id = int(self.ids[idx])
            return feature, _id
