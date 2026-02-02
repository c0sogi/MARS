import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import set_seed


def load_processed_data(metadata_path, split_name, load_cached_data=True):
    """
    Loads images based on metadata. Implements caching using .npy files.

    Args:
        metadata_path (str): Path to the CSV metadata file.
        split_name (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        list of dicts: Each dict contains 'id', 'noisy' (np.array), and optional 'clean' (np.array).
    """
    # Ensure cache directory exists
    os.makedirs(Config.IDEA_DIR, exist_ok=True)
    cache_dir = os.path.join(Config.IDEA_DIR, "cache", split_name)
    os.makedirs(cache_dir, exist_ok=True)

    # Load metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    if Config.DEBUG:
        df = df.head(Config.DEBUG_SUBSET_SIZE)

    data_list = []

    # Check cache validity
    cache_valid = False
    if load_cached_data:
        all_files_exist = True
        for _, row in df.iterrows():
            img_id = str(row["id"])
            noisy_path = os.path.join(cache_dir, f"{img_id}_noisy.npy")
            if not os.path.exists(noisy_path):
                all_files_exist = False
                break
            if "label_path" in row and pd.notna(row["label_path"]):
                clean_path = os.path.join(cache_dir, f"{img_id}_clean.npy")
                if not os.path.exists(clean_path):
                    all_files_exist = False
                    break

        if all_files_exist:
            cache_valid = True

    if cache_valid:
        # Load from cache
        for _, row in df.iterrows():
            img_id = str(row["id"])
            sample = {"id": img_id}

            noisy_path = os.path.join(cache_dir, f"{img_id}_noisy.npy")
            sample["noisy"] = np.load(noisy_path)

            if "label_path" in row and pd.notna(row["label_path"]):
                clean_path = os.path.join(cache_dir, f"{img_id}_clean.npy")
                sample["clean"] = np.load(clean_path)

            data_list.append(sample)
    else:
        # Process from scratch
        for _, row in df.iterrows():
            img_id = str(row["id"])
            feature_path = os.path.join(Config.INPUT_DIR, row["feature_path"])

            # Load Noisy Image
            noisy_img = cv2.imread(feature_path, cv2.IMREAD_GRAYSCALE)
            if noisy_img is None:
                raise FileNotFoundError(f"Could not load image: {feature_path}")

            # Normalize to 0-1
            noisy_img = noisy_img.astype(np.float32) / 255.0

            sample = {"id": img_id, "noisy": noisy_img}

            # Save to cache
            np.save(os.path.join(cache_dir, f"{img_id}_noisy.npy"), noisy_img)

            # Load Clean Image (if available)
            if "label_path" in row and pd.notna(row["label_path"]):
                label_path = os.path.join(Config.INPUT_DIR, row["label_path"])
                clean_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                if clean_img is None:
                    raise FileNotFoundError(f"Could not load image: {label_path}")

                clean_img = clean_img.astype(np.float32) / 255.0
                sample["clean"] = clean_img

                # Save to cache
                np.save(os.path.join(cache_dir, f"{img_id}_clean.npy"), clean_img)

            data_list.append(sample)

    return data_list


class DenoisingDataset(Dataset):
    def __init__(self, data_list, mode="train", patch_size=Config.PATCH_SIZE):
        """
        Args:
            data_list (list): List of dicts with image data.
            mode (str): 'train' (extracts patches) or 'val'/'test' (full images).
            patch_size (int): Size of random crop for training.
        """
        self.data_list = data_list
        self.mode = mode
        self.patch_size = patch_size

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        noisy = sample["noisy"]

        if self.mode == "train":
            clean = sample["clean"]
            h, w = noisy.shape

            # Ensure image is large enough for patch via padding
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)

            if pad_h > 0 or pad_w > 0:
                noisy = np.pad(noisy, ((0, pad_h), (0, pad_w)), mode="reflect")
                clean = np.pad(clean, ((0, pad_h), (0, pad_w)), mode="reflect")
                h, w = noisy.shape

            # Random crop
            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)

            noisy_patch = noisy[
                top : top + self.patch_size, left : left + self.patch_size
            ]
            clean_patch = clean[
                top : top + self.patch_size, left : left + self.patch_size
            ]

            # Convert to tensor and add channel dim: (H, W) -> (1, H, W)
            noisy_t = torch.from_numpy(noisy_patch).unsqueeze(0)
            clean_t = torch.from_numpy(clean_patch).unsqueeze(0)

            return noisy_t, clean_t

        else:
            # Validation/Test: Return full image
            noisy_t = torch.from_numpy(noisy).unsqueeze(0)

            if "clean" in sample:
                clean_t = torch.from_numpy(sample["clean"]).unsqueeze(0)
                return noisy_t, clean_t, sample["id"]
            else:
                return noisy_t, sample["id"]
