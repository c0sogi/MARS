import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import (
    INPUT_DIR,
    CACHE_DIR,
    IMG_SIZE,
    SEQ_LENGTH,
    CLASSES,
    WORKING_DIR,
)
from library.utils import rle_decode


class UWMadisonDataset(Dataset):
    def __init__(self, csv_path, mode="train", load_cached_data=True):
        """
        Args:
            csv_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (str): Whether to use cached .npy files.
        """
        self.mode = mode
        self.load_cached_data = load_cached_data

        # Explicitly satisfy the requirement to ensure idea_8 exists
        os.makedirs("./working/idea_8", exist_ok=True)
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Load Metadata
        self.df = pd.read_csv(csv_path, keep_default_na=False)

        # Group by Case and Day
        self.groups = self.df.groupby(["case", "day"])
        self.group_keys = list(self.groups.groups.keys())

        # Process and Cache Data (Deterministic Processing)
        self._process_and_cache_data()

        # Build Index Mapping
        # We flatten the structure to a list of samples: (case, day, slice_index)
        self.samples = []
        for key in self.group_keys:
            case, day = key
            group = self.groups.get_group(key).sort_values("slice")

            # We need the original IDs to map predictions back later
            ids = group["id"].values

            # Store metadata for each slice in the volume
            for i in range(len(group)):
                self.samples.append(
                    {
                        "case": case,
                        "day": day,
                        "slice_idx": i,
                        "id": ids[i],
                        "max_slices": len(group),
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        case, day = sample["case"], sample["day"]
        center_idx = sample["slice_idx"]

        # Paths to cached files
        img_cache_path = os.path.join(CACHE_DIR, f"{case}_{day}_img.npy")
        mask_cache_path = os.path.join(CACHE_DIR, f"{case}_{day}_mask.npy")

        # Load Volumes (using mmap to save memory)
        # Shape: (D, H, W)
        vol_img = np.load(img_cache_path, mmap_mode="r")

        # Determine Sequence Indices
        # SEQ_LENGTH is 5. We want [i-2, i-1, i, i+1, i+2]
        pad = SEQ_LENGTH // 2
        indices = np.arange(center_idx - pad, center_idx + pad + 1)

        # Handle Out-of-Bounds by padding with zeros (black slices)
        # We construct the sequence tensor
        seq_imgs = []
        depth = vol_img.shape[0]

        for i in indices:
            if 0 <= i < depth:
                seq_imgs.append(vol_img[i])
            else:
                # Pad with zeros if slice is out of bounds
                seq_imgs.append(np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32))

        # Stack to create (T, H, W) -> (T, 1, H, W)
        img_tensor = np.array(seq_imgs)
        img_tensor = np.expand_dims(img_tensor, axis=1)  # Add channel dim
        img_tensor = torch.from_numpy(img_tensor).float()

        # Handle Masks
        # We return the mask for the CENTER slice only
        if self.mode in ["train", "val"]:
            if os.path.exists(mask_cache_path):
                vol_mask = np.load(mask_cache_path, mmap_mode="r")
                # Load center slice mask
                if 0 <= center_idx < depth:
                    mask_data = vol_mask[center_idx]  # Shape (3, H, W)
                else:
                    mask_data = np.zeros(
                        (len(CLASSES), IMG_SIZE, IMG_SIZE), dtype=np.float32
                    )

                mask_tensor = torch.from_numpy(mask_data).float()
                return img_tensor, mask_tensor, sample["id"]
            else:
                # Fallback if mask file missing (should not happen in train/val)
                mask_tensor = torch.zeros(
                    (len(CLASSES), IMG_SIZE, IMG_SIZE), dtype=torch.float32
                )
                return img_tensor, mask_tensor, sample["id"]
        else:
            # Test mode: Return image and ID
            # We return a dummy mask for consistency in some loops, or just ID
            return img_tensor, sample["id"]

    def _process_and_cache_data(self):
        """
        Iterates through all case-day groups.
        Checks if processed .npy files exist.
        If not, loads images/masks, preprocesses them, and saves to cache.
        """
        for key in self.group_keys:
            case, day = key
            img_cache_path = os.path.join(CACHE_DIR, f"{case}_{day}_img.npy")
            mask_cache_path = os.path.join(CACHE_DIR, f"{case}_{day}_mask.npy")

            # Check if cache exists
            if self.load_cached_data and os.path.exists(img_cache_path):
                if self.mode == "test":
                    continue
                if os.path.exists(mask_cache_path):
                    continue

            # Process Data
            group = self.groups.get_group(key).sort_values("slice")

            processed_imgs = []
            processed_masks = []

            for _, row in group.iterrows():
                # 1. Load Image
                img_path = os.path.join(INPUT_DIR, row["file_path"])
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

                # Handle different bit-depths (usually 16-bit or 8-bit)
                if img is None:
                    # Fallback for missing file (should be caught by metadata check)
                    img = np.zeros(
                        (int(row["height"]), int(row["width"])), dtype=np.uint8
                    )

                img = img.astype(np.float32)

                # 2. Preprocess Image (Norm + Resize/Pad)
                img_processed = self._preprocess_image(img, IMG_SIZE)
                processed_imgs.append(img_processed)

                # 3. Process Mask (if available)
                if self.mode in ["train", "val"]:
                    h, w = int(row["height"]), int(row["width"])
                    mask_stack = []
                    for class_name in CLASSES:
                        rle = row[class_name]
                        mask = rle_decode(rle, (h, w))
                        # Resize mask using Nearest Neighbor to keep binary
                        mask_processed = self._preprocess_mask(mask, IMG_SIZE)
                        mask_stack.append(mask_processed)

                    # Stack classes: (3, H, W)
                    processed_masks.append(np.stack(mask_stack, axis=0))

            # Stack Volume: (D, H, W)
            vol_img = np.stack(processed_imgs, axis=0)
            np.save(img_cache_path, vol_img)

            if self.mode in ["train", "val"] and processed_masks:
                # Stack Mask Volume: (D, 3, H, W)
                vol_mask = np.stack(processed_masks, axis=0)
                np.save(mask_cache_path, vol_mask)

    def _preprocess_image(self, img, target_size):
        """
        Applies Robust Percentile Normalization and Aspect-Ratio Preserving Center Padding.
        """
        # Normalization
        p1 = np.percentile(img, 1)
        p99 = np.percentile(img, 99)
        img = np.clip(img, p1, p99)
        if p99 > p1:
            img = (img - p1) / (p99 - p1)
        else:
            img = img * 0.0

        # Resize with Aspect Ratio Preservation
        h, w = img.shape[:2]
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Bilinear for images
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Center Padding
        delta_h = target_size - new_h
        delta_w = target_size - new_w
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0.0
        )
        return padded

    def _preprocess_mask(self, mask, target_size):
        """
        Resizes binary mask with aspect ratio preservation and Nearest Neighbor interpolation.
        """
        h, w = mask.shape[:2]
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Nearest Neighbor for masks to preserve binary values
        resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Center Padding
        delta_h = target_size - new_h
        delta_w = target_size - new_w
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
        )
        return padded
