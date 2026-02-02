import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library import config, utils
from library.bbox_handler import BBoxHandler


class WildCamDataset(Dataset):
    """
    PyTorch Dataset for iWildCam 2020.
    Handles loading images, applying context-aware cropping based on MegaDetector results,
    and applying data augmentations.
    """

    def __init__(self, metadata_path, mode="train", bbox_handler=None):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            mode (str): One of 'train', 'val', 'test'.
            bbox_handler (BBoxHandler, optional): Instance of BBoxHandler. If None, a new one is created.
        """
        self.mode = mode
        self.df = pd.read_csv(metadata_path)

        # Initialize BBoxHandler to get cached bounding boxes
        if bbox_handler:
            self.bbox_handler = bbox_handler
        else:
            self.bbox_handler = BBoxHandler(load_cached_data=True)

        # Build Label Mapping
        # We must ensure that the mapping from category_id to model output index (0..N-1)
        # is consistent across Train, Val, and Test. We derive this from the Training set.
        if os.path.exists(config.TRAIN_METADATA_PATH):
            train_df = pd.read_csv(config.TRAIN_METADATA_PATH)
            # Sort unique category IDs to ensure deterministic mapping
            self.unique_classes = sorted(train_df["category_id"].unique())
            self.class_to_idx = {
                cls_id: idx for idx, cls_id in enumerate(self.unique_classes)
            }
        else:
            # Fallback if training metadata is not found (should not happen in this pipeline)
            self.class_to_idx = {}

        # Setup Augmentations
        self.transform = self.get_transforms(mode)

    def get_transforms(self, mode):
        """
        Returns the Albumentations transform pipeline based on the mode.
        """
        if mode == "train":
            return A.Compose(
                [
                    # Resize is applied after cropping in __getitem__
                    A.Resize(height=config.IMAGE_SIZE, width=config.IMAGE_SIZE),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                    ),
                    A.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                    ),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        else:
            # Validation and Test
            return A.Compose(
                [
                    A.Resize(height=config.IMAGE_SIZE, width=config.IMAGE_SIZE),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]

        # 1. Construct Image Path
        # The file_path in metadata is relative to the input directory (e.g., "train/xyz.jpg")
        img_path = os.path.join(config.INPUT_DIR, row["file_path"])

        # 2. Load Image
        image = cv2.imread(img_path)
        if image is None:
            # Handle potentially missing or corrupt images by returning a blank image
            # This ensures the dataloader doesn't crash
            image = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 3. Context-Aware Cropping
        # Retrieve normalized bbox coordinates (x, y, w, h) with margin applied
        norm_x, norm_y, norm_w, norm_h = self.bbox_handler.get_expanded_bbox(image_id)

        h_img, w_img, _ = image.shape

        # Convert normalized coordinates to pixel coordinates
        x = int(norm_x * w_img)
        y = int(norm_y * h_img)
        w = int(norm_w * w_img)
        h = int(norm_h * h_img)

        # Perform the crop
        # Note: get_expanded_bbox already clamps coordinates to [0, 1], so indices are safe
        crop = image[y : y + h, x : x + w]

        # Fallback if crop is empty (rare edge case)
        if crop.size == 0:
            crop = image

        # 4. Apply Augmentations (Resize -> Augment -> Normalize -> Tensor)
        if self.transform:
            augmented = self.transform(image=crop)
            image_tensor = augmented["image"]
        else:
            image_tensor = ToTensorV2()(image=crop)["image"]

        # 5. Return Data
        if self.mode == "test":
            # For test set, we need the image_id to create the submission file
            return image_tensor, image_id
        else:
            # For train/val, we return the image and the mapped label index
            category_id = row["category_id"]
            # Map the raw category_id to the continuous index (0..184)
            label_idx = self.class_to_idx.get(category_id, 0)
            return image_tensor, torch.tensor(label_idx, dtype=torch.long)
