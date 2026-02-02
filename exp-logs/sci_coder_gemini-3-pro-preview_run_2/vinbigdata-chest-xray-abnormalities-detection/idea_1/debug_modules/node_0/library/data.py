import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import rasterio
import tensorflow as tf

# Import from library
from library.config import Config
from library.utils import collate_fn


class LungDataset(Dataset):
    """
    Custom Dataset for Thoracic Lung Disease Detection.
    Handles image loading (rasterio/tensorflow fallback), preprocessing, and target formatting.
    """

    def __init__(self, df, transforms=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            transforms (albumentations.Compose): Transforms to apply.
        """
        self.df = df
        self.transforms = transforms

        # Group annotations by image_id for efficient retrieval.
        # This converts the dataframe into a dictionary: {image_id: [row_dict, ...]}
        # This is significantly faster than filtering the dataframe in __getitem__.
        self.annotations = {}

        # Ensure necessary columns exist (handling test set which might lack them)
        required_cols = ["image_id", "file_path"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain {required_cols}")

        # Grouping
        group_obj = df.groupby("image_id")
        for img_id, group in group_obj:
            self.annotations[img_id] = group.to_dict("records")

        self.image_ids = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, file_path):
        """
        Loads an image from disk.
        Strategy:
        1. Try rasterio (GDAL) which often supports DICOM.
        2. Fallback to TensorFlow's decode_image.
        3. Preprocess: Convert to RGB, Normalize [0,1].
        """
        image = None

        # Attempt 1: Rasterio
        try:
            with rasterio.open(file_path) as src:
                # rasterio reads as (Count, H, W)
                img_data = src.read()
                # Transpose to (H, W, C)
                image = np.transpose(img_data, (1, 2, 0))
        except Exception:
            pass

        # Attempt 2: TensorFlow Fallback
        if image is None:
            try:
                if os.path.exists(file_path):
                    file_content = tf.io.read_file(file_path)
                    # channels=0 allows original channels, we fix to RGB later
                    img_tensor = tf.io.decode_image(
                        file_content, channels=0, expand_animations=False
                    )
                    image = img_tensor.numpy()
            except Exception:
                pass

        # Fallback for failures (should be rare/non-existent)
        if image is None:
            # Return black image
            image = np.zeros(
                (Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.float32
            )

        # Ensure Numpy Array
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # Handle Dimensions
        # If (H, W) -> (H, W, 1)
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        # If (H, W, 1) -> (H, W, 3)
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)

        # If (H, W, 4) -> (H, W, 3) (Remove Alpha)
        if image.shape[-1] == 4:
            image = image[:, :, :3]

        # Normalize to [0, 1]
        # DICOMs can be high bit-depth (12, 16).
        image = image.astype(np.float32)
        img_min = image.min()
        img_max = image.max()
        if img_max > img_min:
            image = (image - img_min) / (img_max - img_min)
        else:
            image = np.zeros_like(image)

        return image

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        records = self.annotations[image_id]

        # Load Image
        file_path = records[0]["file_path"]
        image = self.load_image(file_path)

        # Parse Annotations
        boxes = []
        labels = []

        for row in records:
            # Check if 'class_id' exists (it should for train/val)
            if "class_id" in row:
                cid = row["class_id"]
                # Class 14 is "No finding" -> Skip adding a box
                if cid == 14:
                    continue

                # Get coordinates
                if "x_min" in row:
                    x_min, y_min = row["x_min"], row["y_min"]
                    x_max, y_max = row["x_max"], row["y_max"]

                    # Basic validation
                    if x_max > x_min and y_max > y_min:
                        boxes.append([x_min, y_min, x_max, y_max])
                        # Map Dataset Class (0-13) to Model Class (1-14)
                        # Model Class 0 is reserved for background
                        labels.append(cid + 1)

        # Convert to numpy
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        # Apply Transforms
        if self.transforms:
            # Albumentations handles empty boxes list gracefully if configured
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = transformed["image"]
            boxes = np.array(transformed["bboxes"], dtype=np.float32)
            labels = np.array(transformed["labels"], dtype=np.int64)
        else:
            # Manual ToTensor
            image = torch.from_numpy(image.transpose(2, 0, 1))

        # Construct Target Dictionary
        target = {}
        target["image_id"] = torch.tensor([idx])

        if len(boxes) > 0:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
            # Area for evaluation metrics
            target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (
                target["boxes"][:, 2] - target["boxes"][:, 0]
            )
            target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            # Negative Sample (Background)
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        return image, target, image_id


def get_transforms(split="train"):
    """
    Returns the Albumentations composition for the specified split.
    """
    if split == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )
    else:
        # Validation / Test
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )


def get_dataloaders(debug=False):
    """
    Initializes and returns the training and validation DataLoaders.

    Args:
        debug (bool): If True, uses a small subset of data for debugging.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_META_PATH)
    val_df = pd.read_csv(Config.VAL_META_PATH)

    if debug:
        # Filter for a small number of unique images
        train_imgs = train_df["image_id"].unique()[:100]
        val_imgs = val_df["image_id"].unique()[:50]
        train_df = train_df[train_df["image_id"].isin(train_imgs)]
        val_df = val_df[val_df["image_id"].isin(val_imgs)]

    # Instantiate Datasets
    train_dataset = LungDataset(train_df, transforms=get_transforms("train"))
    val_dataset = LungDataset(val_df, transforms=get_transforms("val"))

    # Instantiate DataLoaders
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

    return train_loader, val_loader


def get_test_dataloader(test_df):
    """
    Initializes and returns the test DataLoader.
    Adds dummy columns to the dataframe to satisfy Dataset requirements.
    """
    df = test_df.copy()

    # Add dummy columns if they don't exist
    if "class_id" not in df.columns:
        df["class_id"] = 14
    if "x_min" not in df.columns:
        df["x_min"] = 0.0
        df["y_min"] = 0.0
        df["x_max"] = 1.0
        df["y_max"] = 1.0

    dataset = LungDataset(df, transforms=get_transforms("val"))

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return loader
