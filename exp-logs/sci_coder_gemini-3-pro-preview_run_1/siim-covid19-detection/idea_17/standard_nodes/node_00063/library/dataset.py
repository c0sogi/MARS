import os
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import ast
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config
from library.utils import seed_everything


def get_transforms(data="train"):
    """
    Returns the augmentation pipeline using Albumentations.

    Args:
        data (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
                ),
                # Crucial: CoarseDropout with mask_fill_value=0 ensures consistency
                A.CoarseDropout(
                    max_holes=Config.CD_MAX_HOLES,
                    max_height=Config.CD_MAX_HEIGHT,
                    max_width=Config.CD_MAX_WIDTH,
                    min_holes=Config.CD_MIN_HOLES,
                    fill_value=Config.CD_FILL_VALUE,
                    mask_fill_value=Config.CD_MASK_FILL_VALUE,
                    p=0.5,
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def process_dicom(path, fix_monochrome=True):
    """
    Reads a DICOM file, handles photometric interpretation, normalizes to 0-255,
    and converts to RGB.
    """
    try:
        dicom = pydicom.dcmread(path)

        # Handle pixel array
        data = dicom.pixel_array

        # Handle Photometric Interpretation
        if fix_monochrome and hasattr(dicom, "PhotometricInterpretation"):
            if dicom.PhotometricInterpretation == "MONOCHROME1":
                data = np.amax(data) - data

        # Normalize to 0-255
        data = data - np.min(data)
        data = data / np.max(data)
        data = (data * 255).astype(np.uint8)

        # Convert to RGB (3 channels) for ResNet backbone
        img = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
        return img

    except Exception as e:
        print(f"Error processing {path}: {e}")
        # Return a black image in case of error to prevent crash
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)


def create_mask(boxes_str, height, width):
    """
    Creates a binary mask from bounding box string.
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    if pd.isna(boxes_str) or boxes_str == "":
        return mask

    try:
        boxes = ast.literal_eval(boxes_str)
        for box in boxes:
            x = int(box["x"])
            y = int(box["y"])
            w = int(box["width"])
            h = int(box["height"])

            # Draw rectangle (fill with 1)
            cv2.rectangle(mask, (x, y), (x + w, y + h), 1, -1)
    except:
        pass

    return mask


def load_data(metadata_path, split, load_cached_data=True):
    """
    Loads dataset arrays (images, masks, labels, ids).
    Implements caching logic using .npy files.

    Args:
        metadata_path (str): Path to the metadata CSV.
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing 'images', 'masks', 'labels', 'ids'.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache paths
    cache_prefix = os.path.join(Config.WORKING_DIR, f"{split}")
    paths = {
        "images": f"{cache_prefix}_images.npy",
        "masks": f"{cache_prefix}_masks.npy",
        "labels": f"{cache_prefix}_labels.npy",
        "ids": f"{cache_prefix}_ids.npy",
        "dims": f"{cache_prefix}_dims.npy",  # Store original dimensions for resizing boxes back
    }

    # 1. Try to load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in paths.values())
        if all_exist:
            print(f"Loading {split} data from cache...")
            data = {}
            for k, v in paths.items():
                data[k] = np.load(v)
            return data
        else:
            print(f"Cache missing for {split}. Processing from scratch...")
    else:
        print(f"Forcing re-processing for {split}...")

    # 2. Process from scratch
    df = pd.read_csv(metadata_path)

    # Debug mode: sample small subset
    if Config.DEBUG:
        df = df.sample(n=min(len(df), 50), random_state=Config.SEED).reset_index(
            drop=True
        )

    num_samples = len(df)

    # Pre-allocate arrays
    # Images: (N, Size, Size, 3) uint8
    images = np.zeros(
        (num_samples, Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8
    )
    # Masks: (N, Size, Size) uint8 - only for train/val
    masks = np.zeros((num_samples, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint8)
    # Labels: (N, 4) float32 - only for train/val
    labels = np.zeros((num_samples, Config.NUM_CLASSES), dtype=np.float32)
    # IDs: List of strings (store as numpy array of objects or strings)
    ids = np.array(df["study_id"].values)
    # Dims: (N, 2) [height, width] to rescale predictions later
    dims = np.zeros((num_samples, 2), dtype=np.int32)

    # Study level columns for labels
    label_cols = [
        "Negative for Pneumonia",
        "Typical Appearance",
        "Indeterminate Appearance",
        "Atypical Appearance",
    ]

    print(f"Processing {num_samples} images for {split}...")

    for i, row in df.iterrows():
        # Path relative to input dir
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # 1. Process Image
        # Read original to get dims
        try:
            dcm = pydicom.dcmread(full_path, stop_before_pixels=True)
            orig_h, orig_w = dcm.Rows, dcm.Columns
            dims[i] = [orig_h, orig_w]

            # Read and Resize Image
            img = process_dicom(full_path)
            img_resized = cv2.resize(
                img, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_AREA
            )
            images[i] = img_resized

            # 2. Process Mask & Labels (only for train/val)
            if split != "test":
                # Labels
                labels[i] = row[label_cols].values.astype(np.float32)

                # Mask
                # Create mask at original resolution then resize
                mask = create_mask(row.get("boxes", ""), orig_h, orig_w)
                mask_resized = cv2.resize(
                    mask,
                    (Config.IMG_SIZE, Config.IMG_SIZE),
                    interpolation=cv2.INTER_NEAREST,
                )
                masks[i] = mask_resized

        except Exception as e:
            print(f"Failed to process index {i}: {e}")
            continue

    # 3. Save to cache
    print(f"Saving {split} data to cache at {Config.WORKING_DIR}...")
    np.save(paths["images"], images)
    np.save(paths["masks"], masks)
    np.save(paths["labels"], labels)
    np.save(paths["ids"], ids)
    np.save(paths["dims"], dims)

    return {
        "images": images,
        "masks": masks,
        "labels": labels,
        "ids": ids,
        "dims": dims,
    }


class SIIMDataset(Dataset):
    """
    Torch Dataset for SIIM-FISABIO-RSNA COVID-19 Detection.
    """

    def __init__(self, data_dict, split="train", transform=None):
        """
        Args:
            data_dict (dict): Dictionary containing images, masks, labels, etc.
            split (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transform pipeline.
        """
        self.images = data_dict["images"]
        self.masks = data_dict["masks"]
        self.labels = data_dict["labels"]
        self.ids = data_dict["ids"]
        self.dims = data_dict["dims"]
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        # Prepare targets
        if self.split != "test":
            mask = self.masks[idx]
            label = self.labels[idx]
        else:
            # Dummy targets for test
            mask = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint8)
            label = np.zeros(Config.NUM_CLASSES, dtype=np.float32)

        # Apply transforms
        if self.transform:
            # Albumentations expects mask to be passed if it exists
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Ensure Mask is (1, H, W) float tensor
        # mask coming out of ToTensorV2 is (H, W) or (H, W, 1) depending on input
        if isinstance(mask, torch.Tensor):
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            elif mask.ndim == 3 and mask.shape[2] == 1:
                mask = mask.permute(2, 0, 1)
        else:
            # Fallback if not tensor (e.g. no transform)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        # Convert mask to float
        mask = mask.float()

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.float32)

        return {
            "image": image,
            "label": label,
            "mask": mask,
            "id": self.ids[idx],
            "orig_dim": self.dims[idx],  # (h, w)
        }
