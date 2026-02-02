import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


def load_data(metadata_path, cache_prefix, load_cached_data=True):
    """
    Loads data from metadata CSV and spectrogram images.
    Implements caching using .npy files in the working directory.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        cache_prefix (str): Prefix for the cache files (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        images (np.ndarray): Array of spectrogram images (N, H, W).
        labels (np.ndarray): Array of multi-hot labels (N, Num_Classes).
        rec_ids (np.ndarray): Array of recording IDs (N,).
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    img_cache_path = os.path.join(cache_dir, f"{cache_prefix}_images.npy")
    lbl_cache_path = os.path.join(cache_dir, f"{cache_prefix}_labels.npy")
    id_cache_path = os.path.join(cache_dir, f"{cache_prefix}_ids.npy")

    # Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(img_cache_path)
            and os.path.exists(lbl_cache_path)
            and os.path.exists(id_cache_path)
        ):
            print(f"Loading {cache_prefix} data from cache...")
            images = np.load(img_cache_path)
            labels = np.load(lbl_cache_path)
            rec_ids = np.load(id_cache_path)

            # Validate cache dimensions
            if labels.shape[1] != Config.NUM_CLASSES:
                print(
                    f"Cache mismatch for {cache_prefix}: Expected {Config.NUM_CLASSES} classes, got {labels.shape[1]}. Regenerating..."
                )
            else:
                # Handle Debug mode by slicing
                if Config.DEBUG:
                    print(
                        f"DEBUG mode: Slicing {cache_prefix} data to {Config.DEBUG_SAMPLES} samples."
                    )
                    return (
                        images[: Config.DEBUG_SAMPLES],
                        labels[: Config.DEBUG_SAMPLES],
                        rec_ids[: Config.DEBUG_SAMPLES],
                    )

                return images, labels, rec_ids

    # Process from scratch
    print(f"Processing {cache_prefix} data from scratch...")
    df = pd.read_csv(metadata_path)

    images_list = []
    labels_list = []
    ids_list = []

    # Identify label columns
    label_cols = [f"species_{i}" for i in range(Config.NUM_CLASSES)]

    for idx, row in df.iterrows():
        # Construct image path
        # file_path is relative, e.g., essential_data/src_wavs/PC...wav
        wav_rel_path = row["file_path"]
        wav_basename = os.path.basename(wav_rel_path)
        bmp_basename = os.path.splitext(wav_basename)[0] + ".bmp"
        img_path = os.path.join(Config.SPECTROGRAM_DIR, bmp_basename)

        if not os.path.exists(img_path):
            # Should not happen based on EDA, but handle gracefully
            print(f"Warning: Image not found {img_path}")
            # Create a blank image or skip. Here we create blank to keep alignment
            img = np.zeros((Config.IMG_HEIGHT, 1246), dtype=np.uint8)
        else:
            # Load image in grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((Config.IMG_HEIGHT, 1246), dtype=np.uint8)
            else:
                # Resize entire image to (IMG_WIDTH, IMG_HEIGHT)
                # Global compression (Cite solution_lesson_node_00010)
                img = cv2.resize(
                    img,
                    (Config.IMG_WIDTH, Config.IMG_HEIGHT),
                    interpolation=cv2.INTER_LINEAR,
                )

        images_list.append(img)
        ids_list.append(row["rec_id"])

        # Get labels
        if label_cols:
            lbl = row[label_cols].values.astype(np.float32)
            labels_list.append(lbl)
        else:
            # Fallback for test set if columns missing (though metadata gen ensures they exist)
            labels_list.append(np.zeros(Config.NUM_CLASSES, dtype=np.float32))

    images = np.array(images_list, dtype=np.uint8)
    labels = np.array(labels_list, dtype=np.float32)
    rec_ids = np.array(ids_list, dtype=np.int64)

    # Save to cache
    np.save(img_cache_path, images)
    np.save(lbl_cache_path, labels)
    np.save(id_cache_path, rec_ids)
    print(f"Saved {cache_prefix} data to cache.")

    if Config.DEBUG:
        print(
            f"DEBUG mode: Slicing {cache_prefix} data to {Config.DEBUG_SAMPLES} samples."
        )
        return (
            images[: Config.DEBUG_SAMPLES],
            labels[: Config.DEBUG_SAMPLES],
            rec_ids[: Config.DEBUG_SAMPLES],
        )

    return images, labels, rec_ids


class BirdDataset(Dataset):
    """
    Dataset for training. Applies random temporal cropping and augmentation.
    """

    def __init__(self, images, labels, transform=None):
        """
        Args:
            images (np.ndarray): (N, H, W) uint8 spectrograms.
            labels (np.ndarray): (N, Num_Classes) float32 labels.
            transform: Optional transform (not used heavily here as we do manual transforms).
        """
        self.images = images
        self.labels = labels
        self.transform = transform

        # ImageNet stats
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.images)

    def apply_spec_augment(self, img_tensor):
        """
        Applies time and frequency masking.
        img_tensor: (C, H, W)
        """
        C, H, W = img_tensor.shape

        # Frequency masking
        if Config.SPECAUG_FREQ_MASK > 0:
            f_width = np.random.randint(0, Config.SPECAUG_FREQ_MASK)
            f0 = np.random.randint(0, max(1, H - f_width))
            img_tensor[:, f0 : f0 + f_width, :] = 0.0

        # Time masking
        if Config.SPECAUG_TIME_MASK > 0:
            t_width = np.random.randint(0, Config.SPECAUG_TIME_MASK)
            t0 = np.random.randint(0, max(1, W - t_width))
            img_tensor[:, :, t0 : t0 + t_width] = 0.0

        return img_tensor

    def __getitem__(self, idx):
        img = self.images[idx]  # (H, W) - already resized in load_data
        label = self.labels[idx]

        # Normalize to 0-1
        img = img.astype(np.float32) / 255.0

        # Convert to Tensor (1, H, W)
        img_tensor = torch.from_numpy(img).unsqueeze(0)

        # Replicate to 3 channels for ResNet
        img_tensor = img_tensor.repeat(3, 1, 1)

        # SpecAugment
        img_tensor = self.apply_spec_augment(img_tensor)

        # Normalize with ImageNet stats
        img_tensor = (img_tensor - self.mean) / self.std

        return img_tensor, torch.from_numpy(label)


class InferenceDataset(Dataset):
    """
    Dataset for Validation/Testing. Returns full image for sliding window inference.
    """

    def __init__(self, images, rec_ids, labels=None):
        self.images = images
        self.rec_ids = rec_ids
        self.labels = labels

        # ImageNet stats
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # (H, W_full)
        rec_id = self.rec_ids[idx]

        # Normalize to 0-1
        img = img.astype(np.float32) / 255.0

        # Convert to Tensor (1, H, W)
        img_tensor = torch.from_numpy(img).unsqueeze(0)

        # Replicate to 3 channels
        img_tensor = img_tensor.repeat(3, 1, 1)

        # Normalize with ImageNet stats
        img_tensor = (img_tensor - self.mean) / self.std

        if self.labels is not None:
            label = torch.from_numpy(self.labels[idx])
            return img_tensor, label, rec_id
        else:
            # Return dummy label if not available
            return img_tensor, torch.zeros(Config.NUM_CLASSES), rec_id
