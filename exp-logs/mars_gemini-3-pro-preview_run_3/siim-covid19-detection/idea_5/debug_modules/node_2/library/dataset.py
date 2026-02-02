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


def get_transforms(mode="train"):
    """
    Returns the Albumentations transformation pipeline.
    Implements Letterbox resizing (LongestMaxSize + PadIfNeeded) to preserve aspect ratio.
    """
    transforms_list = [
        # Resize longest edge to target size
        A.LongestMaxSize(max_size=Config.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR),
        # Pad remaining dimensions to make it square (Letterbox)
        A.PadIfNeeded(
            min_height=Config.IMAGE_SIZE,
            min_width=Config.IMAGE_SIZE,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
        ),
    ]

    if mode == "train":
        transforms_list.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
            ]
        )

    transforms_list.extend(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    # Bbox params: format is pascal_voc (x_min, y_min, x_max, y_max)
    return A.Compose(
        transforms_list,
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["class_labels"], min_visibility=0.0
        ),
    )


def read_dicom(path):
    """
    Reads a DICOM file, handles monochrome inversion, and converts to 8-bit RGB.
    """
    try:
        dcm = pydicom.dcmread(path)
        data = dcm.pixel_array

        # Handle Photometric Interpretation
        if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
            data = np.amax(data) - data

        # Normalize to 0-255
        data = data.astype(np.float32)
        data = (data - data.min()) / (data.max() - data.min() + 1e-6)
        data = (data * 255).astype(np.uint8)

        # Convert to RGB (Backbone expects 3 channels)
        img = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
        return img
    except Exception as e:
        print(f"Error reading DICOM {path}: {e}")
        # Return a black image in case of error to prevent crash
        return np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)


def load_processed_dataframe(mode, load_cached_data=True):
    """
    Loads metadata, processes bounding boxes and labels, and caches the result.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"{mode}_processed.parquet")

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Load raw metadata
    if mode == "train":
        csv_path = Config.TRAIN_METADATA_PATH
    elif mode == "val":
        csv_path = Config.VAL_METADATA_PATH
    else:
        csv_path = Config.TEST_METADATA_PATH

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # 3. Process Data
    # Parse Boxes (only for train/val)
    if mode in ["train", "val"]:
        # Convert string representation of list of dicts to actual list of dicts
        # Format in CSV: "[{'x': 10, 'y': 10, 'width': 100, 'height': 100}]"
        # Target format for processing: [[x_min, y_min, x_max, y_max], ...]

        def parse_boxes(box_str):
            if pd.isna(box_str):
                return []
            try:
                # ast.literal_eval is safer than eval
                box_dicts = ast.literal_eval(box_str)
                boxes = []
                for b in box_dicts:
                    x_min = b["x"]
                    y_min = b["y"]
                    w = b["width"]
                    h = b["height"]
                    boxes.append([x_min, y_min, x_min + w, y_min + h])
                return boxes
            except:
                return []

        df["parsed_boxes"] = df["boxes"].apply(parse_boxes)

        # Create Study Labels (Integer index 0-3)
        # Columns: Negative for Pneumonia, Typical Appearance, Indeterminate Appearance, Atypical Appearance
        # We assume one-hot encoding logic where argmax gives the correct class index matching Config.STUDY_CLASSES
        label_cols = Config.STUDY_CLASSES
        df["study_label"] = df[label_cols].values.argmax(axis=1)

    # 4. Save Cache
    if load_cached_data:
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        # Parquet handles lists/arrays better than CSV
        df.to_parquet(cache_path, index=False)

    return df


class CovidDataset(Dataset):
    def __init__(self, mode="train", transform=None, load_cached_data=True):
        self.mode = mode
        self.df = load_processed_dataframe(mode, load_cached_data)

        # Debug subset
        if Config.DEBUG:
            self.df = self.df.head(Config.DEBUG_SAMPLE_SIZE).reset_index(drop=True)
            print(f"[{mode.upper()}] Debug mode: using {len(self.df)} samples.")

        self.transform = transform if transform else get_transforms(mode)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image Path
        # Metadata file_path is relative to ./input
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load Image
        image = read_dicom(img_path)

        # Prepare Targets
        boxes = []
        labels = []
        study_label = 0

        if self.mode in ["train", "val"]:
            # Boxes
            # Retrieve parsed boxes (numpy array or list)
            raw_boxes = row["parsed_boxes"]
            # Parquet might load lists as numpy arrays or lists, ensure list
            if isinstance(raw_boxes, np.ndarray):
                raw_boxes = raw_boxes.tolist()

            # If no boxes, we still need to pass empty list to albumentations
            # But Albumentations requires non-empty boxes for bbox_params if we don't handle it carefully.
            # However, we configured bbox_params, so we must pass fields.

            if len(raw_boxes) > 0:
                boxes = raw_boxes
                # Class 1 for Opacity
                labels = [1] * len(boxes)
            else:
                boxes = []
                labels = []

            # Study Label
            study_label = int(row["study_label"])

        # Apply Transforms
        # Albumentations expects boxes in format [x_min, y_min, x_max, y_max] (pascal_voc)
        # and requires a label field.

        if self.mode in ["train", "val"]:
            transformed = self.transform(image=image, bboxes=boxes, class_labels=labels)
            image = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["class_labels"]

            # Convert to Tensors
            # Boxes: FloatTensor [N, 4]
            # Labels: Int64Tensor [N]
            # Study: Int64Tensor [1]

            target = {}
            if len(boxes) > 0:
                target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            else:
                target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
            target["study_ids"] = torch.as_tensor(study_label, dtype=torch.int64)
            target["image_id"] = torch.tensor([idx])

            # Area and Iscrowd (useful for COCO eval, good practice to include)
            if len(boxes) > 0:
                area = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (
                    target["boxes"][:, 2] - target["boxes"][:, 0]
                )
                target["area"] = area
                target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
            else:
                target["area"] = torch.as_tensor([], dtype=torch.float32)
                target["iscrowd"] = torch.as_tensor([], dtype=torch.int64)

            return image, target

        else:
            # Test mode
            # We pass dummy boxes to transform if needed, but usually just image is enough
            # unless we use transforms that depend on boxes (Crop).
            # Letterbox/Resize/Normalize don't need boxes.
            # However, our transform pipeline defines bbox_params.
            # We must pass dummy lists to satisfy the Compose interface if bbox_params are present.

            transformed = self.transform(image=image, bboxes=[], class_labels=[])
            image = transformed["image"]

            # Return image and metadata for identification
            info = {
                "study_id": row["study_id"],
                "image_id": row["image_id"],
                "original_width": row.get(
                    "width", Config.IMAGE_SIZE
                ),  # Fallback if not in test meta
                "original_height": row.get("height", Config.IMAGE_SIZE),
            }

            return image, info


def collate_fn(batch):
    """
    Custom collate function for object detection.
    Batches images as a tensor, but keeps targets as a list of dictionaries.
    """
    images = []
    targets = []

    for b in batch:
        images.append(b[0])
        targets.append(b[1])

    images = torch.stack(images, dim=0)

    return images, targets
