import os
import cv2
import torch
import numpy as np
import pandas as pd
import ast
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config
from library import utils


def get_transforms(data="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        data (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    transforms_list = [
        A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE, p=1.0),
    ]

    if data == "train":
        transforms_list.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
                ),
            ]
        )

    transforms_list.extend(
        [
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ]
    )

    return A.Compose(
        transforms_list,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


def process_metadata(subset, load_cached_data=True):
    """
    Loads and processes metadata with caching mechanism.

    Args:
        subset (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Processed dataframe.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, f"cached_{subset}_metadata.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Ensure boxes are lists (parquet might save as array/list)
            # If they were saved as strings/objects, we might need to verify,
            # but usually parquet handles lists well or we might need to re-parse if schema drifted.
            # For robustness, we assume parquet saves the structure correctly.
            return df
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Recomputing.")

    # 2. Compute from scratch
    if subset == "train":
        path = Config.TRAIN_METADATA_PATH
    elif subset == "val":
        path = Config.VAL_METADATA_PATH
    else:
        path = Config.TEST_METADATA_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    df = pd.read_csv(path)

    # Process Boxes
    # 'boxes' column contains string representation of list of dicts or NaN
    if "boxes" in df.columns:

        def parse_boxes(x):
            if pd.isna(x):
                return []
            try:
                return ast.literal_eval(x)
            except:
                return []

        df["boxes_list"] = df["boxes"].apply(parse_boxes)
    else:
        # For test set, create empty lists
        df["boxes_list"] = [[] for _ in range(len(df))]

    # Process Study Labels
    # Create a single 'study_label_idx' column (0-3)
    # 0: Negative, 1: Typical, 2: Indeterminate, 3: Atypical
    if "Negative for Pneumonia" in df.columns:
        label_cols = [
            "Negative for Pneumonia",
            "Typical Appearance",
            "Indeterminate Appearance",
            "Atypical Appearance",
        ]
        # argmax returns index of the first occurrence of maximum value (1)
        df["study_label_idx"] = df[label_cols].values.argmax(axis=1)
    else:
        # For test set, default to -1 or 0 (will be ignored during inference)
        df["study_label_idx"] = 0

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

    return df


class CovidDataset(Dataset):
    def __init__(self, subset="train", transforms=None, load_cached_data=True):
        """
        Args:
            subset (str): 'train', 'val', or 'test'.
            transforms (A.Compose): Albumentations transforms.
            load_cached_data (bool): Use cached metadata if available.
        """
        self.subset = subset
        self.df = process_metadata(subset, load_cached_data=load_cached_data)
        self.transforms = transforms
        self.input_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        file_path = os.path.join(self.input_dir, row["file_path"])

        # 1. Load Image
        # utils.read_dicom returns RGB/Grayscale numpy array (H, W) or (H, W, C)
        img = utils.read_dicom(file_path)

        if img is None:
            # Return None to be filtered by collate_fn
            return None, None, None

        # Apply CLAHE
        img = utils.apply_clahe(img)

        # Ensure 3 channels for backbone (ResNeXt expects 3 channels)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # 2. Process Boxes and Labels
        raw_boxes = row["boxes_list"]  # List of dicts {'x', 'y', 'width', 'height'}
        study_label = row["study_label_idx"]

        boxes = []
        labels = []

        # If study is Negative (0), there should be no boxes.
        # If study is Positive (1, 2, 3), boxes get that class ID.

        if len(raw_boxes) > 0:
            # Convert COCO [x, y, w, h] to Pascal VOC [x1, y1, x2, y2]
            for box in raw_boxes:
                x_min = float(box["x"])
                y_min = float(box["y"])
                w = float(box["width"])
                h = float(box["height"])

                x_max = x_min + w
                y_max = y_min + h

                # Clip to image dimensions to avoid errors
                h_img, w_img = img.shape[:2]
                x_min = max(0, min(x_min, w_img - 1))
                y_min = max(0, min(y_min, h_img - 1))
                x_max = max(0, min(x_max, w_img - 1))
                y_max = max(0, min(y_max, h_img - 1))

                # Filter invalid boxes
                if (x_max > x_min) and (y_max > y_min):
                    boxes.append([x_min, y_min, x_max, y_max])
                    # Assign granular label based on study type
                    # If study is Typical(1), box is 1. Indeterminate(2)->2, Atypical(3)->3.
                    # Note: If data has boxes but study is Negative (shouldn't happen in clean data),
                    # we might want to skip or force label. Assuming clean data based on metadata script.
                    labels.append(study_label)

        # 3. Apply Transforms
        if self.transforms:
            # Albumentations requires labels for BboxParams
            transformed = self.transforms(image=img, bboxes=boxes, labels=labels)
            img = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["labels"]

        # 4. Convert to Tensor Targets
        # Boxes: FloatTensor [N, 4]
        # Labels: Int64Tensor [N]

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            # Empty tensors for images with no findings
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])  # Dummy ID for evaluator
        target["area"] = area
        target["iscrowd"] = iscrowd

        # Add study label for the auxiliary head
        target["study_label"] = torch.as_tensor(study_label, dtype=torch.int64)

        return img, target, image_id
