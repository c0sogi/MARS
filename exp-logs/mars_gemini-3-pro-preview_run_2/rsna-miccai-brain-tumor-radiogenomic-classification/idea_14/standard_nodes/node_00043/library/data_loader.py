import os
import re
import cv2
import torch
import pydicom
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.utils import set_seed

# Ensure deterministic behavior
set_seed(42)


def natural_sort_key(s):
    """
    Sorts strings that contain numbers in a natural way (e.g., Image-1, Image-2, Image-10).
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def read_dicom_robust(path):
    """
    Reads a DICOM file robustly.
    Primary: pydicom + pixel_array.
    Fallback: Raw binary tail read based on file size.
    Returns: numpy array (H, W) or zeros if failed.
    """
    try:
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array
        return img
    except Exception:
        # Fallback: Raw binary read
        try:
            file_size = os.path.getsize(path)
            # Heuristic for resolution based on file size
            # 512x512x2 bytes = 524288
            # 256x256x2 bytes = 131072
            if file_size >= 524288:
                rows, cols = 512, 512
            elif file_size >= 131072:
                rows, cols = 256, 256
            else:
                # Unknown small size, return placeholder
                return np.zeros((224, 224), dtype=np.float32)

            num_pixels = rows * cols
            num_bytes = num_pixels * 2  # uint16

            with open(path, "rb") as f:
                f.seek(-num_bytes, os.SEEK_END)
                data = f.read(num_bytes)

            img = np.frombuffer(data, dtype=np.uint16).reshape((rows, cols))
            return img
        except Exception:
            return np.zeros((224, 224), dtype=np.float32)


def select_anchor_integral(flair_paths):
    """
    Selects the anchor slice index based on the maximum integral (sum) of intensity
    within the 15%-85% depth range of the FLAIR modality.
    """
    if not flair_paths:
        return 0

    num_slices = len(flair_paths)
    start_idx = int(num_slices * 0.15)
    end_idx = int(num_slices * 0.85)

    # Handle small volumes
    if start_idx >= end_idx:
        start_idx = 0
        end_idx = num_slices

    max_integral = -1
    best_idx = num_slices // 2  # Default to middle

    # Iterate through the valid range
    for i in range(start_idx, end_idx):
        path = flair_paths[i]
        img = read_dicom_robust(path)

        if img is None:
            continue

        current_integral = np.sum(img)

        if current_integral > max_integral:
            max_integral = current_integral
            best_idx = i

    return best_idx


def get_roi_cache(df, root_dir, load_cached_data=True):
    """
    Manages caching of the ROI anchor points.
    Returns a dictionary: {BraTS21ID: anchor_index}
    """
    # Cite solution_lesson_node_00024: Use absolute indexing instead of relative ratio.
    cache_dir = "./working/roi_cache_absolute"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "roi_cache.parquet")

    cache_dict = {}

    # 1. Load Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            cache_df = pd.read_parquet(cache_path)
            # Convert to dict
            cache_dict = pd.Series(
                cache_df.anchor_index.values, index=cache_df.BraTS21ID
            ).to_dict()
        except Exception as e:
            print(f"Failed to load cache: {e}")
            cache_dict = {}

    # 2. Identify missing IDs
    ids_to_process = []
    for _, row in df.iterrows():
        if row["BraTS21ID"] not in cache_dict:
            ids_to_process.append(row)

    # 3. Process missing
    if ids_to_process:
        new_entries = []
        for row in ids_to_process:
            subject_id = row["BraTS21ID"]
            flair_dir = os.path.join(root_dir, row["path_FLAIR"])

            anchor_idx = -1
            if os.path.exists(flair_dir):
                files = os.listdir(flair_dir)
                files.sort(key=natural_sort_key)
                full_paths = [os.path.join(flair_dir, f) for f in files]

                anchor_idx = select_anchor_integral(full_paths)

            cache_dict[subject_id] = anchor_idx
            new_entries.append({"BraTS21ID": subject_id, "anchor_index": anchor_idx})

        # 4. Save updated cache
        if new_entries:
            new_df = pd.DataFrame(new_entries)
            if os.path.exists(cache_path):
                existing_df = pd.read_parquet(cache_path)
                combined_df = pd.concat([existing_df, new_df]).drop_duplicates(
                    subset=["BraTS21ID"]
                )
                combined_df.to_parquet(cache_path)
            else:
                new_df.to_parquet(cache_path)

    return cache_dict


def get_transforms(phase="train"):
    """
    Returns the Albumentations transform pipeline.
    """
    if phase == "train":
        return A.Compose(
            [
                A.Rotate(limit=15, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


class BraTSDataset(Dataset):
    def __init__(
        self,
        df,
        root_dir="./input",
        phase="train",
        transform=None,
        load_cached_data=True,
    ):
        self.df = df
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform

        # Initialize Cache
        self.anchor_cache = get_roi_cache(
            df, root_dir, load_cached_data=load_cached_data
        )

        self.modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]
        self.img_size = 224
        self.stride = 5

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subject_id = row["BraTS21ID"]

        # Get anchor index (Cite solution_lesson_node_00024)
        anchor_idx = self.anchor_cache.get(subject_id, -1)

        channels = []

        for mod in self.modalities:
            mod_dir = os.path.join(self.root_dir, row[f"path_{mod}"])

            # Get file list
            if os.path.exists(mod_dir):
                files = os.listdir(mod_dir)
                files.sort(key=natural_sort_key)
            else:
                files = []

            num_files = len(files)

            if num_files == 0:
                # Handle missing modality with zeros
                for _ in range(3):
                    channels.append(
                        np.zeros((self.img_size, self.img_size), dtype=np.float32)
                    )
                continue

            # Determine center index using Absolute Indexing (Cite solution_lesson_node_00024)
            if anchor_idx == -1:
                center_idx = num_files // 2
            else:
                center_idx = anchor_idx

            # Clamp to valid range for this modality
            center_idx = max(0, min(center_idx, num_files - 1))

            # Select 3 slices: center-stride, center, center+stride
            indices = [center_idx - self.stride, center_idx, center_idx + self.stride]

            for i in indices:
                # Clamp index
                i = max(0, min(i, num_files - 1))

                file_path = os.path.join(mod_dir, files[i])
                img = read_dicom_robust(file_path)

                # Resize to 224x224
                img = cv2.resize(
                    img.astype(np.float32),
                    (self.img_size, self.img_size),
                    interpolation=cv2.INTER_AREA,
                )

                # Conservative Min-Max Normalization [0, 1]
                min_val = img.min()
                max_val = img.max()
                if max_val > min_val:
                    img = (img - min_val) / (max_val - min_val)
                else:
                    img = np.zeros_like(img)

                channels.append(img)

        # Stack channels: (H, W, 12) for Albumentations
        # Order: FLAIR(3), T1w(3), T1wCE(3), T2w(3)
        img_stack = np.stack(channels, axis=-1)

        if self.transform:
            augmented = self.transform(image=img_stack)
            img_tensor = augmented["image"]  # (12, H, W) via ToTensorV2
        else:
            # Manual ToTensor if no transform provided
            img_tensor = torch.from_numpy(img_stack.transpose(2, 0, 1)).float()

        # Get label if available
        if "MGMT_value" in row:
            target = torch.tensor(row["MGMT_value"], dtype=torch.float32)
            return img_tensor, target
        else:
            return img_tensor


def get_dataloader(
    df, root_dir, phase="train", batch_size=32, num_workers=2, load_cached_data=True
):
    transform = get_transforms(phase)
    dataset = BraTSDataset(df, root_dir, phase, transform, load_cached_data)

    shuffle = phase == "train"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader
