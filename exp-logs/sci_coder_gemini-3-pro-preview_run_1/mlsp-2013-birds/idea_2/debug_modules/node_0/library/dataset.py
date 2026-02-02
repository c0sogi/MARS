import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def load_metadata(csv_path, load_cached_data=True):
    """
    Loads the metadata CSV file. Implements caching using Parquet format
    to speed up subsequent loads and ensure deterministic processing.

    Args:
        csv_path (str): Path to the source CSV file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The loaded metadata dataframe.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Construct cache filename based on the input CSV filename
    filename = os.path.basename(csv_path)
    cache_filename = filename.replace(".csv", ".parquet")
    cache_path = os.path.join(Config.WORKING_DIR, f"cached_{cache_filename}")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If loading fails, fall back to processing from scratch
            pass

    # 2. Process from scratch
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")

    return df


def get_transforms(
    mode="train", img_height=Config.IMG_HEIGHT, img_width=Config.IMG_WIDTH
):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        mode (str): 'train' for augmentation, 'val' or 'test' for deterministic resizing.
        img_height (int): Target height.
        img_width (int): Target width.

    Returns:
        A.Compose: Composed transforms.
    """
    # ImageNet normalization statistics
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transforms_list = []

    # Resize to fixed dimensions
    # We use Resize rather than RandomCrop to ensure the model sees the entire 10s context,
    # even if the time axis is compressed.
    transforms_list.append(A.Resize(height=img_height, width=img_width))

    if mode == "train":
        # SpecAugment-style masking using CoarseDropout
        # This masks out rectangular regions in the spectrogram, simulating
        # time masking and frequency masking.
        transforms_list.append(
            A.CoarseDropout(
                max_holes=8,
                max_height=int(img_height * 0.15),  # Mask up to 15% of frequency
                max_width=int(img_width * 0.15),  # Mask up to 15% of time
                min_holes=2,
                fill_value=0,
                p=0.5,
            )
        )
        # We can also add a slight amount of Gaussian Noise to robustness
        transforms_list.append(A.GaussNoise(var_limit=(10.0, 50.0), p=0.3))

    # Normalize and convert to Tensor
    transforms_list.append(A.Normalize(mean=mean, std=std))
    transforms_list.append(ToTensorV2())

    return A.Compose(transforms_list)


class BirdDataset(Dataset):
    """
    PyTorch Dataset for Bird Species Classification using Spectrograms.
    """

    def __init__(
        self,
        metadata_path,
        mode="train",
        transform=None,
        max_samples=None,
        load_cached_data=True,
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV (train.csv, val.csv, or test.csv).
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transforms.
            max_samples (int, optional): Limit dataset size for debugging.
            load_cached_data (bool): Whether to use cached metadata.
        """
        self.mode = mode
        self.transform = transform
        self.df = load_metadata(metadata_path, load_cached_data=load_cached_data)

        # Debugging: Limit samples if requested
        if max_samples is not None:
            self.df = self.df.iloc[:max_samples].reset_index(drop=True)

        # Identify label columns (species_0 to species_18)
        self.label_cols = [c for c in self.df.columns if c.startswith("species_")]
        self.num_classes = len(self.label_cols)

        # Pre-check consistency
        if self.num_classes != Config.NUM_CLASSES:
            # If columns are missing (e.g. in test set if not generated correctly), handle gracefully or warn
            pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rec_id = row["rec_id"]

        # 1. Construct Image Path
        # The metadata contains relative wav path, e.g., essential_data/src_wavs/PC10_...wav
        # We need to map this to supplemental_data/spectrograms/PC10_...bmp
        wav_rel_path = row["file_path"]
        base_name = os.path.basename(wav_rel_path)
        bmp_name = os.path.splitext(base_name)[0] + ".bmp"
        img_path = os.path.join(Config.SPECTROGRAM_DIR, bmp_name)

        # 2. Load Image
        # Load as grayscale (spectrograms are single channel intensity maps)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            # Fallback for missing files (should not happen in valid dataset)
            # Create a black image of default size
            image = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH), dtype=np.uint8)

        # 3. Convert to RGB
        # ResNet expects 3 input channels. We replicate the grayscale channel.
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # 4. Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Minimal transform if none provided: ToTensor
            image = ToTensorV2()(image=image)["image"]

        # 5. Get Labels
        # For test set, labels might be all 0s, which is fine.
        labels = row[self.label_cols].values.astype(np.float32)

        return image, torch.tensor(labels), rec_id
