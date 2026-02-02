import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def get_transforms(phase: str):
    """
    Creates the augmentation pipeline using Albumentations.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The composed transform pipeline.
    """
    transforms = []

    # 1. Resize to global context (224x224)
    transforms.append(A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE))

    if phase == "train":
        # 2. Geometric Augmentation: Time Shift (Horizontal Translation)
        # Strictly avoid cyclic wrapping by using BORDER_CONSTANT (Zero Padding)
        transforms.append(
            A.ShiftScaleRotate(
                shift_limit_x=Config.TIME_SHIFT_LIMIT,
                shift_limit_y=0,
                rotate_limit=0,
                scale_limit=0,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5,
            )
        )

        # 3. Photometric Augmentation: Brightness and Contrast
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=Config.BRIGHTNESS_JITTER,
                contrast_limit=Config.CONTRAST_JITTER,
                p=0.5,
            )
        )

        # Note: Horizontal Flip is strictly EXCLUDED as per strategy.

    # 4. Normalization (ImageNet stats)
    transforms.append(
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    )

    # 5. Convert to Tensor
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


class BirdDataset(Dataset):
    """
    PyTorch Dataset for Bird Species Classification.
    Handles loading of BMP spectrograms, 3-channel conversion, caching, and label parsing.
    """

    def __init__(
        self, metadata_path, phase, load_cached_data=True, transform=None, debug=False
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            phase (str): 'train', 'val', or 'test'. Used for cache naming and default transforms.
            load_cached_data (bool): If True, attempts to load/save data from/to .npy cache.
            transform (A.Compose, optional): Albumentations transform pipeline. If None, uses get_transforms(phase).
            debug (bool): If True, limits the dataset to a small subset for debugging.
        """
        self.metadata_path = metadata_path
        self.phase = phase
        self.transform = transform if transform is not None else get_transforms(phase)
        self.debug = debug

        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        # Define cache file paths
        self.cache_images_path = os.path.join(Config.CACHE_DIR, f"{phase}_images.npy")
        self.cache_labels_path = os.path.join(Config.CACHE_DIR, f"{phase}_labels.npy")

        # Load Data
        self.images, self.labels = self._load_data(load_cached_data)

    def _load_data(self, load_cached_data):
        """
        Internal method to handle caching logic.
        """
        # 1. Try to load from cache
        if (
            load_cached_data
            and os.path.exists(self.cache_images_path)
            and os.path.exists(self.cache_labels_path)
        ):
            try:
                images = np.load(self.cache_images_path)
                labels = np.load(self.cache_labels_path)
                if self.debug:
                    images = images[: Config.DEBUG_SAMPLE_SIZE]
                    labels = labels[: Config.DEBUG_SAMPLE_SIZE]
                return images, labels
            except Exception as e:
                print(
                    f"Failed to load cache for {self.phase}: {e}. Reloading from source."
                )

        # 2. Load from source
        df = pd.read_csv(self.metadata_path)

        if self.debug:
            df = df.head(Config.DEBUG_SAMPLE_SIZE)

        images_list = []
        labels_list = []

        for _, row in df.iterrows():
            # --- Image Loading ---
            # Metadata file_path points to WAV: essential_data/src_wavs/filename.wav
            # We need BMP: supplemental_data/spectrograms/filename.bmp
            wav_rel_path = row["file_path"]
            filename = os.path.basename(wav_rel_path)
            bmp_filename = filename.replace(".wav", ".bmp")
            bmp_path = os.path.join(Config.SPECTROGRAM_DIR, bmp_filename)

            # Read image
            img = cv2.imread(bmp_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                # Fallback for missing files (should not happen based on verification)
                # Create a black image of approximate size (spectrograms are usually ~500x250)
                # We use 224x224 as a safe placeholder since it will be resized anyway
                img = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.uint8)

            # --- 3-Channel Rule ---
            # Replicate single channel 3 times to mimic RGB
            img_rgb = cv2.merge([img, img, img])
            images_list.append(img_rgb)

            # --- Label Parsing ---
            label_str = row["labels"]
            label_vec = np.zeros(Config.NUM_CLASSES, dtype=np.float32)

            if pd.notna(label_str) and label_str != "?":
                try:
                    indices = [int(x) for x in str(label_str).split()]
                    # Clip indices just in case
                    indices = [i for i in indices if 0 <= i < Config.NUM_CLASSES]
                    label_vec[indices] = 1.0
                except ValueError:
                    pass  # Keep zero vector on error

            labels_list.append(label_vec)

        # Convert to numpy arrays
        images_array = np.array(images_list, dtype=np.uint8)
        labels_array = np.array(labels_list, dtype=np.float32)

        # 3. Save to cache (only if not in debug mode to avoid overwriting full cache with subset)
        if not self.debug:
            try:
                np.save(self.cache_images_path, images_array)
                np.save(self.cache_labels_path, labels_array)
            except Exception as e:
                print(f"Warning: Failed to save cache for {self.phase}: {e}")

        return images_array, labels_array

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, torch.tensor(label, dtype=torch.float32)
