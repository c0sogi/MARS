import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library import utils


class ThoracicDataset(Dataset):
    def __init__(
        self, metadata_df, transforms=None, mode="train", load_cached_data=True
    ):
        """
        Args:
            metadata_df: DataFrame containing metadata.
            transforms: Albumentations transforms.
            mode: 'train', 'val', or 'test'.
            load_cached_data: Whether to use cached .npy files.
        """
        self.metadata_df = metadata_df
        self.transforms = transforms
        self.mode = mode
        self.load_cached_data = load_cached_data

        # Group by image_id to handle multiple findings per image
        if mode in ["train", "val"]:
            self.image_ids = self.metadata_df["image_id"].unique()
            self.grouped = self.metadata_df.groupby("image_id")
        else:
            self.image_ids = self.metadata_df["image_id"].unique()

        # Ensure cache directory exists
        self.cache_subdir = os.path.join(Config.CACHE_DIR, mode)
        os.makedirs(self.cache_subdir, exist_ok=True)

    def __len__(self):
        return len(self.image_ids)

    def _load_image_original(self, image_id, rel_path):
        """
        Loads the original image with caching logic.
        Returns: numpy array (H, W) float32
        """
        cache_path = os.path.join(self.cache_subdir, f"{image_id}.npy")

        # 1. Try to load from cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                pass  # Fallback to source

        # 2. Load from source
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        img = utils.read_dicom_binary(full_path)

        # 3. Save to cache
        try:
            np.save(cache_path, img)
        except Exception:
            pass

        return img

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # Retrieve metadata
        if self.mode in ["train", "val"]:
            rows = self.grouped.get_group(image_id)
            rel_path = rows.iloc[0]["file_path"]
        else:
            # Test mode: find row
            row = self.metadata_df[self.metadata_df["image_id"] == image_id].iloc[0]
            rel_path = row["file_path"]

        # Load Original Image
        img_original = self._load_image_original(image_id, rel_path)
        orig_h, orig_w = img_original.shape[:2]

        # Resize to Model Input Size
        img_resized = cv2.resize(img_original, (Config.IMG_SIZE, Config.IMG_SIZE))

        # Normalize to 0-1
        if img_resized.max() > 0:
            img_resized = img_resized.astype(np.float32) / img_resized.max()
        else:
            img_resized = img_resized.astype(np.float32)

        # Convert to RGB (EfficientNet expects 3 channels)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

        target = {}

        if self.mode in ["train", "val"]:
            boxes = []
            labels = []
            is_finding = 0.0

            for _, row in rows.iterrows():
                class_id = row["class_id"]

                # Class 14 is "No finding"
                if class_id == Config.NO_FINDING_CLASS_ID:
                    continue

                # It is a finding
                is_finding = 1.0

                # Scale bounding box to new resolution
                x_min = (row["x_min"] / orig_w) * Config.IMG_SIZE
                y_min = (row["y_min"] / orig_h) * Config.IMG_SIZE
                x_max = (row["x_max"] / orig_w) * Config.IMG_SIZE
                y_max = (row["y_max"] / orig_h) * Config.IMG_SIZE

                # Clip to image boundaries
                x_min = max(0, min(x_min, Config.IMG_SIZE))
                y_min = max(0, min(y_min, Config.IMG_SIZE))
                x_max = max(0, min(x_max, Config.IMG_SIZE))
                y_max = max(0, min(y_max, Config.IMG_SIZE))

                # Ensure valid box
                if x_max > x_min and y_max > y_min:
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id)

            # Apply Transforms
            if self.transforms:
                if self.mode == "train":
                    # Train: augment image and boxes
                    try:
                        transformed = self.transforms(
                            image=img_rgb, bboxes=boxes, labels=labels
                        )
                        img_tensor = transformed["image"]
                        boxes = transformed["bboxes"]
                        labels = transformed["labels"]
                    except ValueError:
                        # Fallback if augmentation fails (e.g. bbox issues)
                        img_tensor = ToTensorV2()(image=img_rgb)["image"]
                else:
                    # Val: just normalize image
                    transformed = self.transforms(image=img_rgb)
                    img_tensor = transformed["image"]
            else:
                img_tensor = ToTensorV2()(image=img_rgb)["image"]

            target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.tensor(labels, dtype=torch.int64)
            target["cls_target"] = torch.tensor([is_finding], dtype=torch.float32)
            target["orig_size"] = torch.tensor([orig_w, orig_h], dtype=torch.int32)
            target["image_id"] = image_id

            return img_tensor, target, image_id

        else:
            # Test Mode
            if self.transforms:
                transformed = self.transforms(image=img_rgb)
                img_tensor = transformed["image"]
            else:
                img_tensor = ToTensorV2()(image=img_rgb)["image"]

            target["orig_size"] = torch.tensor([orig_w, orig_h], dtype=torch.int32)
            target["image_id"] = image_id

            return img_tensor, target, image_id


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for the given mode.
    """
    if mode == "train":
        return A.Compose(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["labels"],
                min_visibility=Config.MIN_VISIBILITY,
            ),
        )

    else:
        # Val/Test: Normalize and Convert to Tensor
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def collate_fn(batch):
    """
    Custom collate function to handle variable number of boxes.
    """
    images, targets, image_ids = zip(*batch)
    images = torch.stack(images)
    return images, targets, image_ids


def get_dataloaders(train_meta=None, val_meta=None, test_meta=None):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    dataloaders = {}

    if train_meta:
        df_train = pd.read_csv(train_meta)
        ds_train = ThoracicDataset(
            df_train, transforms=get_transforms("train"), mode="train"
        )
        dataloaders["train"] = DataLoader(
            ds_train,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    if val_meta:
        df_val = pd.read_csv(val_meta)
        ds_val = ThoracicDataset(df_val, transforms=get_transforms("val"), mode="val")
        dataloaders["val"] = DataLoader(
            ds_val,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    if test_meta:
        df_test = pd.read_csv(test_meta)
        ds_test = ThoracicDataset(
            df_test, transforms=get_transforms("test"), mode="test"
        )
        dataloaders["test"] = DataLoader(
            ds_test,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    return dataloaders
