import os
import cv2
import ast
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config
from library.utils import read_dicom


class SIIMDataset(Dataset):
    """
    PyTorch Dataset for SIIM-FISABIO-RSNA COVID-19 Detection.
    Handles loading DICOM images, parsing bounding boxes, and mapping study labels
    to box-level classes for Class-Aware Object Detection.
    """

    def __init__(self, dataframe, mode="train", transforms=None, limit_size=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing metadata.
            mode (str): 'train', 'val', or 'test'.
            transforms (albumentations.Compose): Transforms to apply.
            limit_size (int, optional): Limit dataset size for debugging.
        """
        self.df = dataframe
        self.mode = mode
        self.transforms = transforms

        # Filter for debugging if requested
        if limit_size is not None:
            self.df = self.df.iloc[:limit_size].reset_index(drop=True)

        # Pre-parse columns if they exist to speed up __getitem__
        if self.mode in ["train", "val"]:
            self.image_ids = self.df["image_id"].values
            self.file_paths = self.df["file_path"].values
            self.boxes_str = self.df["boxes"].values

            # Extract class labels for each row
            # Priority: Typical (1) > Indeterminate (2) > Atypical (3) > Negative (0/None)
            self.class_ids = []
            for _, row in self.df.iterrows():
                if row.get("Typical Appearance", 0) == 1:
                    self.class_ids.append(1)
                elif row.get("Indeterminate Appearance", 0) == 1:
                    self.class_ids.append(2)
                elif row.get("Atypical Appearance", 0) == 1:
                    self.class_ids.append(3)
                else:
                    # Negative for Pneumonia or no label
                    self.class_ids.append(0)
        else:
            # Test mode
            self.image_ids = self.df["image_id"].values
            self.file_paths = self.df["file_path"].values
            self.study_ids = self.df["StudyInstanceUID"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load Image
        file_path = self.file_paths[idx]
        full_path = os.path.join(Config.INPUT_DIR, file_path)

        # read_dicom returns RGB numpy array (H, W, 3) in uint8 0-255
        image = read_dicom(full_path)

        # 2. Handle Targets (Train/Val)
        boxes = []
        labels = []

        if self.mode in ["train", "val"]:
            box_data = self.boxes_str[idx]
            class_id = self.class_ids[idx]

            # Parse boxes if they exist and class is not Negative (0)
            if class_id > 0 and pd.notna(box_data):
                try:
                    # box_data format: [{'x': ..., 'y': ..., 'width': ..., 'height': ...}, ...]
                    decoded_boxes = ast.literal_eval(box_data)
                    for box in decoded_boxes:
                        x = float(box["x"])
                        y = float(box["y"])
                        w = float(box["width"])
                        h = float(box["height"])

                        # Convert to Pascal VOC format [xmin, ymin, xmax, ymax]
                        xmin = x
                        ymin = y
                        xmax = x + w
                        ymax = y + h

                        # Clip boxes to image dimensions to prevent Albumentations errors
                        img_h, img_w = image.shape[:2]
                        xmin = max(0, min(xmin, img_w))
                        ymin = max(0, min(ymin, img_h))
                        xmax = max(0, min(xmax, img_w))
                        ymax = max(0, min(ymax, img_h))

                        # Ensure valid box area
                        if (xmax > xmin) and (ymax > ymin):
                            boxes.append([xmin, ymin, xmax, ymax])
                            labels.append(class_id)
                except (ValueError, SyntaxError):
                    # Fallback for malformed strings
                    pass

            # 3. Apply Transforms
            if self.transforms:
                # Albumentations expects boxes in a specific format if bbox_params are set
                # We use format='pascal_voc' in get_transforms
                if len(boxes) > 0:
                    transformed = self.transforms(
                        image=image, bboxes=boxes, labels=labels
                    )
                    image = transformed["image"]
                    boxes = transformed["bboxes"]
                    labels = transformed["labels"]
                else:
                    # Apply only image transforms if no boxes
                    # We need a separate call or handle empty lists depending on config
                    # Ideally, the transform pipeline handles empty bboxes gracefully
                    transformed = self.transforms(image=image, bboxes=[], labels=[])
                    image = transformed["image"]
                    boxes = []  # Ensure it's empty list
                    labels = []

            # 4. Convert to Tensor Targets
            # Faster R-CNN expects:
            # - boxes: FloatTensor[N, 4]
            # - labels: Int64Tensor[N]

            if len(boxes) > 0:
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)

                # Area and iscrowd (optional but good for compatibility)
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
            else:
                # Negative sample
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
                area = torch.zeros((0,), dtype=torch.float32)
                iscrowd = torch.zeros((0,), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = torch.tensor([idx])
            target["area"] = area
            target["iscrowd"] = iscrowd

            return image, target, self.image_ids[idx]

        else:
            # Test Mode
            if self.transforms:
                # Only image transforms (Resize, Normalize, ToTensor)
                transformed = self.transforms(image=image, bboxes=[], labels=[])
                image = transformed["image"]

            return image, self.image_ids[idx], self.study_ids[idx]


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for the dataset.

    Args:
        mode (str): 'train', 'val', or 'test'
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.CLAHE(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.2
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )

    elif mode == "val":
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )

    else:  # Test
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )
