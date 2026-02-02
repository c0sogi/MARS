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
from library.utils import read_dicom, get_cached_data

# ==========================================
# Metadata Processing & Caching
# ==========================================


def process_metadata(csv_path, split_name):
    """
    Loads metadata CSV, parses bounding boxes, maps study labels,
    and prepares it for the Dataset class.
    """
    df = pd.read_csv(csv_path)

    # 1. Parse Bounding Boxes
    # Format in CSV: [{'x': ..., 'y': ..., 'width': ..., 'height': ...}]
    def parse_boxes(row):
        if "boxes" not in row or pd.isna(row["boxes"]):
            return []

        try:
            boxes_list = ast.literal_eval(row["boxes"])
            parsed_boxes = []
            for box in boxes_list:
                x = float(box["x"])
                y = float(box["y"])
                w = float(box["width"])
                h = float(box["height"])
                # Convert to Pascal VOC [xmin, ymin, xmax, ymax]
                parsed_boxes.append([x, y, x + w, y + h])
            return parsed_boxes
        except Exception:
            return []

    if "boxes" in df.columns:
        df["parsed_boxes"] = df.apply(parse_boxes, axis=1)
    else:
        # Test set does not have boxes
        df["parsed_boxes"] = df.apply(lambda x: [], axis=1)

    # 2. Map Study Labels
    # Columns: "Negative for Pneumonia", "Typical Appearance", etc.
    def get_study_label(row):
        for label_name, label_id in Config.STUDY_CLASS_MAP.items():
            if label_name in row and row[label_name] == 1:
                return label_id
        return 0  # Default to 0 (Negative) or handle as unknown

    # Only map if columns exist (Train/Val)
    if "Negative for Pneumonia" in df.columns:
        df["study_label"] = df.apply(get_study_label, axis=1)
    else:
        df["study_label"] = 0

    return df


def get_processed_metadata(split, load_cached_data=True):
    """
    Wrapper to get cached metadata dataframe.
    """
    if split == "train":
        path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        path = Config.VAL_METADATA_PATH
    elif split == "test":
        path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    def generate():
        return process_metadata(path, split)

    return get_cached_data(
        cache_name=f"metadata_{split}",
        generate_fn=generate,
        load_cached_data=load_cached_data,
        base_dir=Config.WORKING_DIR,
    )


# ==========================================
# Augmentations
# ==========================================


def get_transforms(split):
    """
    Returns Albumentations transforms for train/val/test.
    Includes CLAHE and Resizing as per strategy.
    """
    common_transforms = [
        # CLAHE for contrast enhancement (Strategy requirement)
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        # Fixed resolution
        A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
    ]

    if split == "train":
        return A.Compose(
            common_transforms
            + [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )
    else:
        return A.Compose(
            common_transforms
            + [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )


# ==========================================
# Dataset Class
# ==========================================


class COVIDDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata.
            transforms (albumentations.Compose): Transforms to apply.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode

        # Debugging: subset if configured
        if Config.DEBUG:
            self.df = self.df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]

        # 1. Load Image
        # file_path is relative to INPUT_DIR
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        try:
            # read_dicom returns numpy array [0-255] uint8
            image = read_dicom(full_path)
            # Convert to 3-channel for backbone compatibility (ResNet expects 3 channels)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        except Exception as e:
            # Fallback for missing/corrupt images (should not happen in clean data)
            print(f"Error loading {full_path}: {e}")
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)

        # 2. Prepare Boxes and Labels
        boxes = []
        labels = []

        if self.mode in ["train", "val"]:
            raw_boxes = row["parsed_boxes"]
            study_label = row["study_label"]

            # If study is Negative (0), there are no boxes.
            # If study is Typical(1), Indeterminate(2), Atypical(3),
            # boxes inherit the study label class ID.
            if len(raw_boxes) > 0:
                boxes = np.array(list(raw_boxes), dtype=np.float32)
                # Ensure boxes are valid [xmin, ymin, xmax, ymax]
                # Clip to image dimensions just in case
                h, w, _ = image.shape
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)

                # Filter degenerate boxes
                keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                boxes = boxes[keep]

                # Assign labels based on study type
                # Create a list of labels corresponding to the number of boxes
                labels = np.full(len(boxes), study_label, dtype=np.int64)
            else:
                boxes = np.empty((0, 4), dtype=np.float32)
                labels = np.empty((0,), dtype=np.int64)
        else:
            # Test mode
            study_label = 0  # Dummy
            boxes = np.empty((0, 4), dtype=np.float32)
            labels = np.empty((0,), dtype=np.int64)

        # 3. Apply Transforms
        if self.transforms:
            # Albumentations requires labels for bbox_params
            augmented = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = augmented["image"]
            boxes = augmented["bboxes"]
            labels = augmented["labels"]

        # 4. Construct Target Dict (for Faster R-CNN)
        if self.mode in ["train", "val"]:
            # Convert to tensors
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)

            # Handle empty boxes case after augmentation
            if boxes_t.shape[0] == 0:
                boxes_t = torch.zeros((0, 4), dtype=torch.float32)
                labels_t = torch.zeros((0,), dtype=torch.int64)
                area = torch.zeros((0,), dtype=torch.float32)
            else:
                area = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])

            target = {}
            target["boxes"] = boxes_t
            target["labels"] = labels_t
            target["image_id"] = torch.tensor([idx])
            target["area"] = area
            target["iscrowd"] = torch.zeros((len(labels_t),), dtype=torch.int64)

            # Auxiliary Head Target
            target["study_label"] = torch.tensor(study_label, dtype=torch.int64)

            return image, target, image_id
        else:
            # Test mode: return image and ID
            return image, image_id


def get_datasets(load_cached_data=True):
    """
    Factory function to create Train and Validation datasets.
    """
    # Load Metadata
    train_df = get_processed_metadata("train", load_cached_data)
    val_df = get_processed_metadata("val", load_cached_data)

    # Create Datasets
    train_dataset = COVIDDataset(
        train_df, transforms=get_transforms("train"), mode="train"
    )

    val_dataset = COVIDDataset(val_df, transforms=get_transforms("val"), mode="val")

    return train_dataset, val_dataset


def get_test_dataset(load_cached_data=True):
    """
    Factory function to create Test dataset.
    """
    test_df = get_processed_metadata("test", load_cached_data)

    test_dataset = COVIDDataset(test_df, transforms=get_transforms("test"), mode="test")

    return test_dataset
