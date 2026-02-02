import os
import cv2
import numpy as np
import pandas as pd
import torch
import ast
import pydicom
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def get_transforms(data_split):
    """
    Returns the Albumentations transforms for the given data split.

    Args:
        data_split (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The composition of transforms.
    """
    if data_split == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                # Label-Consistent CoarseDropout
                # Drops rectangular regions in the image and sets corresponding mask area to 0
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(Config.IMG_SIZE * 0.1),
                    max_width=int(Config.IMG_SIZE * 0.1),
                    min_holes=1,
                    min_height=int(Config.IMG_SIZE * 0.05),
                    min_width=int(Config.IMG_SIZE * 0.05),
                    fill_value=0,
                    mask_fill_value=Config.MASK_FILL_VALUE,
                    p=0.5,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test transforms (No augmentation)
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def prepare_data(split, load_cached_data=True, debug=False):
    """
    Prepares the data arrays (images, masks, labels) for the given split.
    Handles caching to .npy files in Config.OUTPUT_DIR to speed up subsequent runs.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from disk first.
        debug (bool): If True, processes only a small subset of data.

    Returns:
        tuple: (images, masks, labels, ids)
            - images: np.ndarray (N, H, W) uint8
            - masks: np.ndarray (N, H, W) uint8 (or None for test)
            - labels: np.ndarray (N, 4) float32 (or None for test)
            - ids: np.ndarray (N,) string
    """
    # Ensure output directory exists
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # Load Metadata first to verify cache integrity or process from scratch
    if split == "train":
        df = pd.read_csv(Config.TRAIN_CSV)
    elif split == "val":
        df = pd.read_csv(Config.VAL_CSV)
    elif split == "test":
        df = pd.read_csv(Config.TEST_CSV)
    else:
        raise ValueError(f"Unknown split: {split}")

    if debug:
        df = df.head(50)

    # Define cache file paths
    cache_prefix = f"{split}"
    if debug:
        cache_prefix += "_debug"

    img_cache_path = os.path.join(Config.OUTPUT_DIR, f"{cache_prefix}_images.npy")
    mask_cache_path = os.path.join(Config.OUTPUT_DIR, f"{cache_prefix}_masks.npy")
    label_cache_path = os.path.join(Config.OUTPUT_DIR, f"{cache_prefix}_labels.npy")
    id_cache_path = os.path.join(Config.OUTPUT_DIR, f"{cache_prefix}_ids.npy")

    # 1. Try to load from cache
    if load_cached_data:
        # Check if basic files exist
        if os.path.exists(img_cache_path) and os.path.exists(id_cache_path):
            # Cite debug_lesson_3: Validate Cache Integrity Against Runtime Configuration
            try:
                # For train/val, we also need masks and labels
                if split in ["train", "val"]:
                    if os.path.exists(mask_cache_path) and os.path.exists(
                        label_cache_path
                    ):
                        images = np.load(img_cache_path)
                        masks = np.load(mask_cache_path)
                        labels = np.load(label_cache_path)
                        ids = np.load(id_cache_path, allow_pickle=True)

                        if len(images) == len(df):
                            print(f"Loading cached data for {split} (debug={debug})...")
                            return images, masks, labels, ids
                        else:
                            print(
                                f"Cache mismatch for {split}: expected {len(df)}, found {len(images)}. Reloading..."
                            )
                else:
                    # Test split only needs images and IDs
                    images = np.load(img_cache_path)
                    ids = np.load(id_cache_path, allow_pickle=True)

                    if len(images) == len(df):
                        print(f"Loading cached data for {split} (debug={debug})...")
                        return images, None, None, ids
                    else:
                        print(
                            f"Cache mismatch for {split}: expected {len(df)}, found {len(images)}. Reloading..."
                        )
            except Exception as e:
                print(f"Error loading cache for {split}: {e}. Reloading...")

    # 2. Process from scratch
    print(f"Processing data for {split} (debug={debug}) from source...")

    images = []
    masks = []
    labels = []
    ids = []

    for idx, row in df.iterrows():
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        try:
            # Read DICOM
            dcm = pydicom.dcmread(file_path)
            img = dcm.pixel_array

            # Handle Photometric Interpretation (Invert if MONOCHROME1)
            if (
                hasattr(dcm, "PhotometricInterpretation")
                and dcm.PhotometricInterpretation == "MONOCHROME1"
            ):
                img = np.max(img) - img

            # Normalize to 0-255 uint8
            img = img.astype(float)
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)
            img = (img * 255).astype(np.uint8)

            # Resize Image
            img_resized = cv2.resize(
                img, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_AREA
            )
            images.append(img_resized)

            # Store ID
            ids.append(row["image_id"])

            # Process Masks and Labels (Train/Val only)
            if split in ["train", "val"]:
                mask = np.zeros((dcm.Rows, dcm.Columns), dtype=np.uint8)

                # Parse bounding boxes
                if pd.notna(row["boxes"]):
                    try:
                        boxes = ast.literal_eval(row["boxes"])
                        for box in boxes:
                            x = int(box["x"])
                            y = int(box["y"])
                            w = int(box["width"])
                            h = int(box["height"])
                            # Draw filled rectangle
                            cv2.rectangle(mask, (x, y), (x + w, y + h), 1, -1)
                    except:
                        pass  # No boxes or parse error

                # Resize Mask
                mask_resized = cv2.resize(
                    mask,
                    (Config.IMG_SIZE, Config.IMG_SIZE),
                    interpolation=cv2.INTER_NEAREST,
                )
                masks.append(mask_resized)

                # Process Labels (One-hot encoding)
                label_vec = np.array([row[c] for c in Config.CLASSES], dtype=np.float32)
                labels.append(label_vec)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    # Convert lists to numpy arrays
    images = np.array(images, dtype=np.uint8)
    ids = np.array(ids)

    if split in ["train", "val"]:
        masks = np.array(masks, dtype=np.uint8)
        labels = np.array(labels, dtype=np.float32)
    else:
        masks = None
        labels = None

    # Save to cache
    print(f"Saving processed data to {Config.OUTPUT_DIR}...")
    np.save(img_cache_path, images)
    np.save(id_cache_path, ids)
    if split in ["train", "val"]:
        np.save(mask_cache_path, masks)
        np.save(label_cache_path, labels)

    return images, masks, labels, ids


class SIIMDataset(Dataset):
    def __init__(self, images, masks=None, labels=None, ids=None, transforms=None):
        """
        PyTorch Dataset for SIIM-FISABIO-RSNA COVID-19 Detection.

        Args:
            images (np.ndarray): Array of images (N, H, W) uint8.
            masks (np.ndarray, optional): Array of masks (N, H, W) uint8.
            labels (np.ndarray, optional): Array of labels (N, 4) float32.
            ids (np.ndarray, optional): Array of image IDs.
            transforms (albumentations.Compose): Transforms to apply.
        """
        self.images = images
        self.masks = masks
        self.labels = labels
        self.ids = ids
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image
        image = self.images[idx]

        # Convert grayscale to RGB (3 channels) for ResNet compatibility
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.masks is not None:
            # Train/Val mode
            mask = self.masks[idx]
            label = self.labels[idx]

            if self.transforms:
                # Apply transforms (including CoarseDropout)
                # Albumentations handles image and mask simultaneously
                augmented = self.transforms(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]

            # Ensure mask has channel dimension (1, H, W)
            # ToTensorV2 converts mask to (H, W) tensor, we need to unsqueeze
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

            mask = mask.float()

            # Return tuple: (image, mask, label, index)
            # Index is returned to allow mapping back to IDs during validation if needed
            return image, mask, label, idx

        else:
            # Test mode
            if self.transforms:
                augmented = self.transforms(image=image)
                image = augmented["image"]

            # Return image and index (to map back to ID for submission)
            return image, idx
