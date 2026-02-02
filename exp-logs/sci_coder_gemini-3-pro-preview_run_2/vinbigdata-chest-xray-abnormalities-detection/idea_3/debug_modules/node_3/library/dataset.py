import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import collate_fn


class LungDiseaseDataset(Dataset):
    def __init__(self, dataframe, transforms=None, is_test=False):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing metadata (image_id, class_id, bbox, file_path).
            transforms (A.Compose): Albumentations transforms.
            is_test (bool): If True, returns only image and image_id/size info.
        """
        self.df = dataframe
        self.transforms = transforms
        self.is_test = is_test

        # Group by image_id to handle multiple objects per image
        self.image_ids = self.df["image_id"].unique()
        # Create a fast lookup for image data
        self.group_map = self.df.groupby("image_id")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        group = self.group_map.get_group(image_id)

        # Get file path from the first row of the group
        file_path = group.iloc[0]["file_path"]

        # Load Image
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if image is None:
            # Fallback for missing images (should not happen with robust preprocess)
            # Create a blank image
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0  # Normalize to [0, 1] before transforms if needed,
        # but usually Albumentations Normalize handles pixel values.
        # Reverting to uint8 for Albumentations standard flow
        image = (image * 255).astype(np.uint8)

        # Prepare boxes and labels
        boxes = []
        labels = []

        if not self.is_test:
            for _, row in group.iterrows():
                class_id = row["class_id"]

                # Handle "No finding" (Class 14)
                # In object detection, "No finding" means no boxes.
                if class_id == 14:
                    continue

                # Shift labels: 0 (Back) -> 1 (Aortic) ... 14 (Fibrosis)
                # Dataset Class 0 -> Model Label 1
                label = int(class_id) + 1

                x_min, y_min = row["x_min"], row["y_min"]
                x_max, y_max = row["x_max"], row["y_max"]

                # basic validity check
                if x_max > x_min and y_max > y_min:
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(label)

            # Convert to numpy for transforms
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)

            # If no boxes (e.g. "No finding" image), ensure shapes are correct for albumentations
            if len(boxes) == 0:
                boxes = np.empty((0, 4), dtype=np.float32)
                labels = np.empty((0,), dtype=np.int64)

        # Apply Transforms
        if self.transforms:
            if self.is_test:
                # Test transform usually doesn't need bboxes, but we keep consistency
                sample = self.transforms(image=image)
                image = sample["image"]
            else:
                # Train/Val transform with bboxes
                sample = self.transforms(image=image, bboxes=boxes, labels=labels)
                image = sample["image"]
                boxes = np.array(sample["bboxes"], dtype=np.float32)
                labels = np.array(sample["labels"], dtype=np.int64)

        # Prepare Target Dict (for PyTorch Detection Models)
        if self.is_test:
            target = {
                "image_id": image_id,
                "original_size": (image.shape[1], image.shape[2]),  # C, H, W
            }
            return image, target
        else:
            # Convert to tensors
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)

            # Calculate Area
            if len(boxes_t) > 0:
                area = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])
            else:
                area = torch.as_tensor([], dtype=torch.float32)

            target = {
                "boxes": boxes_t,
                "labels": labels_t,
                "image_id": torch.tensor([idx]),
                "area": area,
                "iscrowd": torch.zeros((len(labels_t),), dtype=torch.int64),
            }
            return image, target


def get_transforms(split="train"):
    """
    Returns Albumentations transforms for train/val/test.
    """
    # Try to read dataset stats, otherwise use ImageNet
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    stats_path = os.path.join(Config.WORKING_DIR, "dataset_stats.txt")
    if os.path.exists(stats_path):
        try:
            with open(stats_path, "r") as f:
                lines = f.readlines()
                # format: mean,0.123
                for line in lines:
                    k, v = line.strip().split(",")
                    if k == "mean":
                        # Replicate for RGB
                        val = float(v) / 255.0
                        mean = [val, val, val]
                    elif k == "std":
                        val = float(v) / 255.0
                        std = [val, val, val]
        except Exception:
            pass

    if split == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
                ),
                # CoarseDropout for robustness against occlusion and to force distributed feature learning
                A.CoarseDropout(
                    max_holes=8,
                    max_height=Config.IMG_SIZE // 10,
                    max_width=Config.IMG_SIZE // 10,
                    min_holes=1,
                    fill_value=0,
                    p=0.2,
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )

    else:
        # Val / Test
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ],
            bbox_params=(
                A.BboxParams(format="pascal_voc", label_fields=["labels"])
                if split != "test"
                else None
            ),
        )


def get_dataloaders(load_cached_data=True):
    """
    Loads data from parquet files and returns DataLoaders.

    Args:
        load_cached_data (bool): Whether to look for cached parquet files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure processed data exists
    if not (
        os.path.exists(Config.PROCESSED_TRAIN_PKL)
        and os.path.exists(Config.PROCESSED_VAL_PKL)
        and os.path.exists(Config.PROCESSED_TEST_PKL)
    ):
        raise FileNotFoundError(
            "Processed parquet files not found. Run preprocessing first."
        )

    # Load DataFrames
    train_df = pd.read_parquet(Config.PROCESSED_TRAIN_PKL)
    val_df = pd.read_parquet(Config.PROCESSED_VAL_PKL)
    test_df = pd.read_parquet(Config.PROCESSED_TEST_PKL)

    # Create Datasets
    train_dataset = LungDiseaseDataset(
        train_df, transforms=get_transforms("train"), is_test=False
    )
    val_dataset = LungDiseaseDataset(
        val_df, transforms=get_transforms("val"), is_test=False
    )
    test_dataset = LungDiseaseDataset(
        test_df, transforms=get_transforms("test"), is_test=True
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
