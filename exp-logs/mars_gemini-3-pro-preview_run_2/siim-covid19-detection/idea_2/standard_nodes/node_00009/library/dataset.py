import os
import cv2
import ast
import torch
import pydicom
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config


def get_transforms(split, img_size):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        split (str): 'train', 'val', or 'test'.
        img_size (int): Target image size.

    Returns:
        A.Compose: The transformation pipeline.
    """
    transforms = []

    # Preprocessing: CLAHE is crucial for X-ray feature enhancement
    if Config.USE_CLAHE:
        transforms.append(
            A.CLAHE(
                clip_limit=Config.CLAHE_CLIP_LIMIT,
                tile_grid_size=Config.CLAHE_TILE_GRID_SIZE,
                p=1.0,
            )
        )

    # Resizing
    transforms.append(A.Resize(height=img_size, width=img_size))

    if split == "train":
        # Augmentations for training
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.2
                ),
            ]
        )

    # Normalization and Tensor conversion
    transforms.extend(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )


class SIIMDataset(Dataset):
    def __init__(self, split="train", debug=False):
        """
        SIIM-FISABIO-RSNA COVID-19 Detection Dataset.

        Args:
            split (str): One of 'train', 'val', 'test'.
            debug (bool): If True, subsets the data for quick debugging.
        """
        super().__init__()
        self.split = split
        self.debug = debug
        self.img_size = Config.IMG_SIZE

        # Load Metadata
        if split == "train":
            self.df = pd.read_csv(Config.TRAIN_METADATA_PATH)
        elif split == "val":
            self.df = pd.read_csv(Config.VAL_METADATA_PATH)
        elif split == "test":
            self.df = pd.read_csv(Config.TEST_METADATA_PATH)
        else:
            raise ValueError(f"Unknown split: {split}")

        # Debugging: Subset data
        if self.debug:
            limit = 100
            if split == "train" and Config.MAX_TRAIN_SAMPLES:
                limit = Config.MAX_TRAIN_SAMPLES
            elif split == "val" and Config.MAX_VAL_SAMPLES:
                limit = Config.MAX_VAL_SAMPLES
            self.df = self.df.iloc[:limit].reset_index(drop=True)

        self.transforms = get_transforms(split, self.img_size)

        # Study Class Mapping (as per Config)
        # 0: Negative, 1: Typical, 2: Indeterminate, 3: Atypical
        self.class_cols = [
            "Negative for Pneumonia",
            "Typical Appearance",
            "Indeterminate Appearance",
            "Atypical Appearance",
        ]

    def __len__(self):
        return len(self.df)

    def read_dicom(self, path):
        """
        Reads a DICOM file and converts it to a standard 8-bit RGB numpy array.
        Handles MONOCHROME1 inversion.
        """
        full_path = os.path.join(Config.INPUT_DIR, path)
        if not os.path.exists(full_path):
            # Fallback for robustness, though metadata should be clean
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        try:
            dcm = pydicom.dcmread(full_path)
            img = dcm.pixel_array

            # Handle Photometric Interpretation
            if (
                hasattr(dcm, "PhotometricInterpretation")
                and dcm.PhotometricInterpretation == "MONOCHROME1"
            ):
                img = np.max(img) - img

            # Normalize to 0-255
            img = img.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
            img = (img * 255).astype(np.uint8)

            # Convert to RGB (3 channels) for backbone compatibility
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)

            return img

        except Exception as e:
            # Fallback to cv2 if pydicom fails
            img = cv2.imread(full_path)
            if img is None:
                return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            return img

    def get_study_label(self, row):
        """
        Extracts the integer study label from the row.
        """
        # If test set, return dummy
        if self.split == "test":
            return 0

        # Check which column is 1
        for idx, col in enumerate(self.class_cols):
            if row[col] == 1:
                return idx
        return 0  # Default to Negative if none found (should not happen in clean data)

    def parse_boxes(self, row):
        """
        Parses bounding boxes from string format to [x_min, y_min, x_max, y_max].
        """
        if self.split == "test":
            return []

        box_str = row.get("boxes", np.nan)
        if pd.isna(box_str):
            return []

        try:
            # boxes are stored as list of dicts: [{'x':..., 'y':..., 'width':..., 'height':...}]
            boxes_dicts = ast.literal_eval(box_str)
            boxes = []
            for b in boxes_dicts:
                x_min = float(b["x"])
                y_min = float(b["y"])
                w = float(b["width"])
                h = float(b["height"])
                x_max = x_min + w
                y_max = y_min + h
                boxes.append([x_min, y_min, x_max, y_max])
            return boxes
        except:
            return []

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]

        # 1. Load Image
        image = self.read_dicom(row["file_path"])

        # 2. Get Targets
        boxes = self.parse_boxes(row)
        study_label = self.get_study_label(row)

        # Assign labels to boxes
        # Strategy: Use study_label as the class ID for the boxes.
        # If study is Negative (0), there are no boxes.
        # If study is Typical (1), boxes are class 1.
        # If study is Indeterminate (2), boxes are class 2.
        # If study is Atypical (3), boxes are class 3.
        labels = [study_label] * len(boxes)

        # 3. Apply Transforms
        # Albumentations requires boxes and labels
        if len(boxes) > 0:
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
            image_tensor = transformed["image"]
            boxes_tensor = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            labels_tensor = torch.tensor(transformed["labels"], dtype=torch.int64)
        else:
            # Handle images with no boxes
            # We must still apply image transforms (resize, normalize)
            # Pass dummy box to satisfy Compose if needed, or just transform image
            # Albumentations handles empty lists gracefully usually
            transformed = self.transforms(image=image, bboxes=[], labels=[])
            image_tensor = transformed["image"]
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        # 4. Construct Target Dict
        target = {}
        target["boxes"] = boxes_tensor
        target["labels"] = labels_tensor
        target["image_id"] = torch.tensor(
            [idx]
        )  # Use index as ID for simplicity in collate

        # Area and iscrowd (required for COCO evaluators / some models)
        if len(boxes_tensor) > 0:
            target["area"] = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (
                boxes_tensor[:, 2] - boxes_tensor[:, 0]
            )
            target["iscrowd"] = torch.zeros((len(boxes_tensor),), dtype=torch.int64)
        else:
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        # Custom Study Label for Global Head
        target["study_label"] = torch.tensor(study_label, dtype=torch.int64)

        return image_tensor, target
