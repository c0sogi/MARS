import os
import ast
import cv2
import torch
import pydicom
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# ==========================================
# Helper Functions
# ==========================================


def read_dicom(path):
    """
    Reads a DICOM file, handles photometric interpretation, and normalizes to 8-bit.
    """
    try:
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(np.float32)

        # Handle Photometric Interpretation
        if (
            hasattr(dcm, "PhotometricInterpretation")
            and dcm.PhotometricInterpretation == "MONOCHROME1"
        ):
            img = np.max(img) - img

        # Normalize to 0-255
        if np.max(img) > np.min(img):
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
        else:
            img = np.zeros_like(img)

        img = (img * 255).astype(np.uint8)

        # Convert to 3 channels for ConvNeXt backbone
        img = np.stack([img, img, img], axis=-1)
        return img
    except Exception as e:
        print(f"Error reading DICOM {path}: {e}")
        # Return a black image in case of error to prevent crash
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)


def get_transforms(split):
    """
    Returns Albumentations transforms for the specified split.
    Implements Letterbox Resizing via LongestMaxSize + PadIfNeeded.
    """
    if split == "train":
        return A.Compose(
            [
                # Letterbox Resize
                A.LongestMaxSize(
                    max_size=Config.IMG_SIZE, interpolation=cv2.INTER_LINEAR
                ),
                A.PadIfNeeded(
                    min_height=Config.IMG_SIZE,
                    min_width=Config.IMG_SIZE,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                # Augmentations
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=Config.IMG_SIZE // 20,
                    max_width=Config.IMG_SIZE // 20,
                    p=0.5,
                ),
                # Normalization
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )

    else:
        return A.Compose(
            [
                # Letterbox Resize
                A.LongestMaxSize(
                    max_size=Config.IMG_SIZE, interpolation=cv2.INTER_LINEAR
                ),
                A.PadIfNeeded(
                    min_height=Config.IMG_SIZE,
                    min_width=Config.IMG_SIZE,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                # Normalization
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )


def process_metadata(csv_path, split, load_cached_data=True):
    """
    Loads and processes metadata CSV. Implements strict caching logic using Parquet.
    """
    cache_filename = f"{split}_metadata.parquet"
    cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # print(f"Loaded {split} metadata from cache.")
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Add integer study labels if available (Train/Val)
    if split in ["train", "val"]:
        label_cols = Config.STUDY_LABELS
        # Argmax to get index (0-3)
        df["study_label_idx"] = df[label_cols].values.argmax(axis=1)
    else:
        # For test set, add dummy column
        df["study_label_idx"] = -1

    # Ensure file_path is treated as relative to INPUT_DIR
    # (The metadata generator already made them relative, but we verify)

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
        # print(f"Saved {split} metadata to cache at {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

    return df


# ==========================================
# Dataset Class
# ==========================================


class ChestXRayDataset(Dataset):
    def __init__(
        self, split="train", load_cached_data=True, transform=None, debug=False
    ):
        self.split = split
        self.transform = transform or get_transforms(split)

        # Determine CSV path
        if split == "train":
            csv_path = Config.TRAIN_METADATA_PATH
        elif split == "val":
            csv_path = Config.VAL_METADATA_PATH
        else:
            csv_path = Config.TEST_METADATA_PATH

        # Load Metadata
        self.df = process_metadata(csv_path, split, load_cached_data)

        # Debug Mode
        if debug or Config.DEBUG:
            self.df = self.df.iloc[: Config.DEBUG_SAMPLE_SIZE].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Image
        # Path is relative to INPUT_DIR
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        image = read_dicom(file_path)
        h_orig, w_orig = image.shape[:2]

        # 2. Prepare Boxes & Labels
        boxes = []
        labels = []  # 1 for opacity, 0 for background (though 0 is usually reserved)

        if self.split in ["train", "val"]:
            # Parse boxes string: "[{'x': ..., 'y': ..., 'width': ..., 'height': ...}]"
            box_str = row.get("boxes", np.nan)

            if pd.notna(box_str):
                try:
                    box_dicts = ast.literal_eval(box_str)
                    for b in box_dicts:
                        x_min = b["x"]
                        y_min = b["y"]
                        w = b["width"]
                        h = b["height"]
                        x_max = x_min + w
                        y_max = y_min + h

                        # Clip to image boundaries
                        x_min = max(0, min(x_min, w_orig))
                        y_min = max(0, min(y_min, h_orig))
                        x_max = max(0, min(x_max, w_orig))
                        y_max = max(0, min(y_max, h_orig))

                        # Filter invalid boxes
                        if (x_max - x_min > 1) and (y_max - y_min > 1):
                            boxes.append([x_min, y_min, x_max, y_max])
                            labels.append(1)  # Class 1: Opacity
                except:
                    pass

        # 3. Apply Transforms
        # Albumentations requires boxes to be a list of lists
        if len(boxes) == 0:
            # Albumentations requires at least one box if bbox_params are set,
            # OR we pass empty list and it handles it if we provide label_fields.
            # However, for empty boxes, we just transform the image.
            # But we must pass the kwargs to match the signature if we want consistency.
            transformed = self.transform(image=image, bboxes=[], labels=[])
        else:
            transformed = self.transform(image=image, bboxes=boxes, labels=labels)

        image_tensor = transformed["image"]
        boxes_transformed = transformed["bboxes"]

        # 4. Format Target for R-CNN
        # Convert boxes to tensor
        if len(boxes_transformed) > 0:
            boxes_tensor = torch.tensor(boxes_transformed, dtype=torch.float32)
            labels_tensor = torch.ones((len(boxes_transformed),), dtype=torch.int64)
            area_tensor = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (
                boxes_tensor[:, 3] - boxes_tensor[:, 1]
            )
            iscrowd_tensor = torch.zeros((len(boxes_transformed),), dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area_tensor = torch.zeros((0,), dtype=torch.float32)
            iscrowd_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes_tensor
        target["labels"] = labels_tensor
        target["image_id"] = torch.tensor([idx])
        target["area"] = area_tensor
        target["iscrowd"] = iscrowd_tensor

        # Add Study Label
        study_label = row["study_label_idx"]
        target["study_label"] = torch.tensor(study_label, dtype=torch.int64)

        # Add metadata for inference mapping
        target["original_size"] = torch.tensor([h_orig, w_orig], dtype=torch.int64)
        target["id_str"] = row["image_id"]  # String ID for submission
        target["study_id_str"] = row["study_id"]

        return image_tensor, target


# ==========================================
# DataLoader Builder
# ==========================================


def collate_fn(batch):
    """
    Custom collate function for R-CNN.
    Images are stacked (since they are resized to same dims).
    Targets are kept as a list of dictionaries.
    """
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, list(targets)


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=False
):

    train_ds = ChestXRayDataset(split="train", debug=debug)
    val_ds = ChestXRayDataset(split="val", debug=debug)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    test_ds = ChestXRayDataset(split="test", debug=False)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return test_loader
