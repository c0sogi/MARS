import os
import cv2
import pydicom
import torch
import numpy as np
import pandas as pd
import ast
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple

from library.config import Config

# Define class mapping for study-level labels
STUDY_CLASSES = [
    "Negative for Pneumonia",
    "Typical Appearance",
    "Indeterminate Appearance",
    "Atypical Appearance",
]


def get_transforms(split: str, img_size: int = 1024):
    """
    Returns the Albumentations transforms for the given split.

    Args:
        split: 'train', 'val', or 'test'.
        img_size: Target image size.
    """
    if split == "train":
        return A.Compose(
            [
                # LSJ: Resize (0.1 to 2.0) -> Pad -> Crop
                # 1. Resize longest edge to target size (base scale)
                A.LongestMaxSize(max_size=img_size, always_apply=True),
                # 2. Randomly scale between 0.1x and 2.0x of the base size
                # scale_limit of (-0.9, 1.0) maps to factors (1-0.9)=0.1 to (1+1.0)=2.0
                A.RandomScale(scale_limit=(-0.9, 1.0), p=1.0),
                # 3. Pad if the resulting image is smaller than target size
                A.PadIfNeeded(
                    min_height=img_size,
                    min_width=img_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    always_apply=True,
                ),
                # 4. Random crop to target size
                A.RandomCrop(height=img_size, width=img_size, always_apply=True),
                # 5. Augmentations
                A.HorizontalFlip(p=0.5),
                # 6. Normalize and Convert
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="coco", min_visibility=0.0, label_fields=["class_labels"]
            ),
        )
    else:
        # Val/Test: Letterbox Resize (Resize Longest + Pad)
        return A.Compose(
            [
                # Resize longest edge to target size
                A.LongestMaxSize(max_size=img_size, always_apply=True),
                # Pad shorter edge to make it square
                A.PadIfNeeded(
                    min_height=img_size,
                    min_width=img_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    always_apply=True,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="coco", min_visibility=0.0, label_fields=["class_labels"]
            ),
        )


class SIIMDataset(Dataset):
    def __init__(
        self,
        split: str,
        load_cached_data: bool = True,
        debug: bool = False,
    ):
        """
        Args:
            split: 'train', 'val', or 'test'.
            load_cached_data: Whether to load cached dataframe.
            debug: If True, subsample dataset for debugging.
        """
        self.split = split
        self.img_size = Config.IMG_SIZE
        self.input_dir = Config.INPUT_DIR

        # Determine metadata path
        if split == "train":
            self.meta_path = Config.TRAIN_METADATA_PATH
        elif split == "val":
            self.meta_path = Config.VAL_METADATA_PATH
        else:
            self.meta_path = Config.TEST_METADATA_PATH

        # Caching logic for Dataframe
        cache_name = f"cached_{split}_df.parquet"
        cache_path = os.path.join(Config.CACHE_DIR, cache_name)

        # Ensure cache dir exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        if load_cached_data and os.path.exists(cache_path):
            print(f"[{split}] Loading cached dataframe from {cache_path}")
            self.df = pd.read_parquet(cache_path)
        else:
            print(f"[{split}] Processing metadata from {self.meta_path}")
            self.df = pd.read_csv(self.meta_path)

            # Preprocess bounding boxes if they exist
            if "boxes" in self.df.columns:
                # Convert string representation to list of dicts
                # Handle NaNs (no boxes)
                self.df["boxes_list"] = self.df["boxes"].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else []
                )
            else:
                self.df["boxes_list"] = [[] for _ in range(len(self.df))]

            # Preprocess study labels
            if split != "test":
                # Create a single label column (0-3)
                # We assume one-hot columns exist in the metadata
                def get_study_label(row):
                    for idx, cls in enumerate(STUDY_CLASSES):
                        if row.get(cls, 0) == 1:
                            return idx
                    return 0  # Default to negative if none found

                self.df["study_label_idx"] = self.df.apply(get_study_label, axis=1)

            # Save to cache
            print(f"[{split}] Saving dataframe to cache {cache_path}")
            self.df.to_parquet(cache_path)

        if debug:
            self.df = self.df.iloc[:50]
            print(f"[{split}] Debug mode: sampled 50 items.")

        self.transforms = get_transforms(split, self.img_size)
        print(f"[{split}] Dataset initialized. Size: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load DICOM
        dicom_path = os.path.join(self.input_dir, row["file_path"])
        try:
            dcm = pydicom.dcmread(dicom_path)
            image = dcm.pixel_array.astype(np.float32)

            # Handle Photometric Interpretation (Monochrome1 means 0 is white)
            if (
                hasattr(dcm, "PhotometricInterpretation")
                and dcm.PhotometricInterpretation == "MONOCHROME1"
            ):
                image = np.max(image) - image

            # Normalize to [0, 1]
            img_min = image.min()
            img_max = image.max()
            if img_max > img_min:
                image = (image - img_min) / (img_max - img_min)
            else:
                image = np.zeros_like(image)

            # Convert to 3 channels (Backbone expects RGB)
            image = np.stack([image, image, image], axis=-1)

            orig_h, orig_w = dcm.Rows, dcm.Columns

        except Exception as e:
            print(f"Error loading {dicom_path}: {e}")
            # Return a dummy image in case of read failure
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
            orig_h, orig_w = self.img_size, self.img_size

        # 2. Prepare Boxes
        # Format: [x, y, w, h] (COCO format for Albumentations)
        boxes = []
        if "boxes_list" in row:
            for box in row["boxes_list"]:
                # box is dict {'x': ..., 'y': ..., 'width': ..., 'height': ...}
                x, y, w, h = box["x"], box["y"], box["width"], box["height"]
                boxes.append([x, y, w, h])

        # Labels: 0 for "opacity" (Config.NUM_CLASSES_DETECTION = 1)
        labels = [0] * len(boxes)

        # 3. Apply Transforms
        if self.transforms:
            transformed = self.transforms(
                image=image, bboxes=boxes, class_labels=labels
            )
            image = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["class_labels"]

        # 4. Post-process Boxes for DINO
        # DINO expects (cx, cy, w, h) normalized to [0, 1] relative to the TRANSFORMED image size.
        # Albumentations with 'coco' returns (x, y, w, h) absolute.

        # Get dimensions of the transformed image (should be IMG_SIZE x IMG_SIZE)
        _, h_new, w_new = image.shape

        boxes_norm = []
        for box in boxes:
            x, y, w, h = box
            # Convert to center coordinates
            cx = x + w / 2
            cy = y + h / 2
            # Normalize
            cx /= w_new
            cy /= h_new
            w /= w_new
            h /= h_new
            boxes_norm.append([cx, cy, w, h])

        boxes_tensor = torch.tensor(boxes_norm, dtype=torch.float32)
        # Ensure shape (N, 4) even if empty
        if boxes_tensor.shape[0] == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)

        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        # 5. Study Label
        if self.split != "test":
            study_label = torch.tensor(row["study_label_idx"], dtype=torch.int64)
        else:
            # Dummy label for test set
            study_label = torch.tensor(-1, dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "study_label": study_label,
            "image_id": row["image_id"],
            "study_id": row["study_id"],
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.int64),
            "img_size": torch.tensor([h_new, w_new], dtype=torch.int64),
        }

        return image, target


def collate_fn(batch):
    """
    Custom collate function for object detection.

    Args:
        batch: List of tuples (image, target)

    Returns:
        images: Stacked image tensor (B, C, H, W)
        targets: List of target dicts
    """
    images = []
    targets = []

    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)

    images = torch.stack(images, dim=0)

    return images, targets
