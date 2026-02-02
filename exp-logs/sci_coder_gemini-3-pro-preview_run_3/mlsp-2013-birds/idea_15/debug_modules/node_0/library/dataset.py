import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import seed_everything


def get_transforms(phase: str):
    """
    Returns the Albumentations transformation pipeline for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The composition of transforms.
    """
    if phase == "train":
        return A.Compose(
            [
                # Ensure size is correct (redundant if cached correctly, but safe)
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                # Horizontal Translation (Time-shifting) using Zero-Padding
                # shift_limit_x=0.2 means +/- 20% width shift.
                # border_mode=cv2.BORDER_CONSTANT with value=0 ensures zero-padding.
                A.ShiftScaleRotate(
                    shift_limit_x=0.2,
                    shift_limit_y=0.0,
                    scale_limit=0.0,
                    rotate_limit=0,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
                # Photometric Augmentation
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                # Normalization (ImageNet stats)
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                # Convert to PyTorch Tensor
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: Resize and Normalize only
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def preprocess_and_cache_images(load_cached_data=True):
    """
    Loads all spectrograms, resizes them to Config.IMG_SIZE, and caches them
    as a single numpy array indexed by rec_id.

    Args:
        load_cached_data (bool): If True, attempts to load from disk first.

    Returns:
        numpy.ndarray: Array of shape (max_id + 1, H, W) containing uint8 images.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "spectrograms.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached images from {cache_path}...")
            images_cache = np.load(cache_path, allow_pickle=False)
            return images_cache
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print("Processing images from scratch...")

    # Load all CSVs to get the full list of rec_ids and filenames
    dfs = []
    for csv_path in [Config.TRAIN_CSV, Config.VAL_CSV, Config.TEST_CSV]:
        if os.path.exists(csv_path):
            dfs.append(pd.read_csv(csv_path))

    full_df = pd.concat(dfs, ignore_index=True)

    # Determine array size
    max_rec_id = full_df["rec_id"].max()
    # Initialize cache array: (max_id + 1, 224, 224)
    # We store as uint8 to save space. Channels are added in Dataset.
    images_cache = np.zeros(
        (max_rec_id + 1, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint8
    )

    # Process each file
    # We iterate unique rec_ids to avoid double processing
    unique_records = full_df.drop_duplicates(subset=["rec_id"])

    for _, row in unique_records.iterrows():
        rec_id = row["rec_id"]

        # Construct path to spectrogram
        # Original path in CSV: essential_data/src_wavs/filename.wav
        # Target path: supplemental_data/spectrograms/filename.bmp
        wav_rel_path = row["file_path"]
        filename = os.path.basename(wav_rel_path)
        filename_bmp = os.path.splitext(filename)[0] + ".bmp"

        spectrogram_path = os.path.join(Config.SPECTROGRAM_DIR, filename_bmp)

        if os.path.exists(spectrogram_path):
            # Load image (Grayscale)
            img = cv2.imread(spectrogram_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Resize
                img_resized = cv2.resize(
                    img,
                    (Config.IMG_SIZE, Config.IMG_SIZE),
                    interpolation=cv2.INTER_LINEAR,
                )
                images_cache[rec_id] = img_resized
            else:
                print(f"Warning: Failed to read image {spectrogram_path}")
        else:
            print(f"Warning: Image not found {spectrogram_path}")

    # 3. Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.save(cache_path, images_cache, allow_pickle=False)
    print(f"Cached {len(unique_records)} images to {cache_path}")

    return images_cache


class BirdDataset(Dataset):
    def __init__(self, df, images_cache, transforms=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata.
            images_cache (np.ndarray): Pre-loaded array of images indexed by rec_id.
            transforms (A.Compose): Albumentations transforms.
            is_test (bool): If True, labels are ignored/dummy.
        """
        self.df = df.reset_index(drop=True)
        self.images_cache = images_cache
        self.transforms = transforms
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rec_id = int(row["rec_id"])

        # Retrieve image from cache
        # Shape: (H, W)
        img = self.images_cache[rec_id]

        # 3-Channel Rule: Replicate grayscale to RGB
        # Shape becomes (H, W, 3)
        img = np.stack([img, img, img], axis=-1)

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=img)
            image_tensor = augmented["image"]
        else:
            # Fallback if no transforms provided (shouldn't happen)
            image_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        # Process Labels
        label_vec = torch.zeros(Config.NUM_CLASSES, dtype=torch.float32)

        if not self.is_test:
            label_str = str(row["labels"])
            if label_str != "?" and label_str.strip():
                try:
                    indices = [int(x) for x in label_str.split()]
                    for cls_idx in indices:
                        if 0 <= cls_idx < Config.NUM_CLASSES:
                            label_vec[cls_idx] = 1.0
                except ValueError:
                    pass  # Empty or malformed label

        return image_tensor, label_vec, rec_id


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Creates and returns DataLoaders for train, val, and test sets.

    Args:
        debug (bool): If True, subsets data for quick debugging.
        load_cached_data (bool): Whether to use cached image array.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Prepare Data
    images_cache = preprocess_and_cache_images(load_cached_data=load_cached_data)

    # 2. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Debugging: Subset data
    if debug:
        train_df = train_df.head(Config.DEBUG_SUBSET_SIZE)
        val_df = val_df.head(Config.DEBUG_SUBSET_SIZE)
        test_df = test_df.head(Config.DEBUG_SUBSET_SIZE)
        print(f"DEBUG MODE: Reduced train size to {len(train_df)}")

    # 3. Create Datasets
    train_dataset = BirdDataset(
        train_df, images_cache, transforms=get_transforms("train"), is_test=False
    )

    val_dataset = BirdDataset(
        val_df, images_cache, transforms=get_transforms("val"), is_test=False
    )

    test_dataset = BirdDataset(
        test_df, images_cache, transforms=get_transforms("test"), is_test=True
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches to stabilize BatchNorm
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
