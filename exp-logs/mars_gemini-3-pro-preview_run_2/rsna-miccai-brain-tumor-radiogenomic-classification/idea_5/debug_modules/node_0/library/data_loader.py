import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from scipy.signal import find_peaks

# Import from provided library
from library.config import Config
from library.dicom_utils import read_dicom_file, normalize_min_max

# Set seeds for reproducibility
import random

random.seed(Config.SEED)
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)


def get_sorted_files(dir_path):
    """
    Returns a sorted list of DICOM files in a directory.
    Sorts based on the integer index in 'Image-X.dcm'.
    """
    if not os.path.exists(dir_path):
        return []

    files = [f for f in os.listdir(dir_path) if f.endswith(".dcm")]
    # Sort by the integer value X in Image-X.dcm
    # Handle potential naming inconsistencies gracefully
    try:
        files.sort(key=lambda x: int(x.replace("Image-", "").replace(".dcm", "")))
    except ValueError:
        files.sort()
    return files


def compute_flair_peaks(subject_row, num_instances=3, min_distance=10):
    """
    Computes the indices of the top-K intensity peaks in the FLAIR modality.
    """
    flair_path = os.path.join(Config.INPUT_DIR, subject_row["path_FLAIR"])
    files = get_sorted_files(flair_path)

    if not files:
        # Fallback if no files found
        return [0] * num_instances

    # Calculate intensity profile
    intensities = []
    valid_indices = []

    for i, f in enumerate(files):
        f_path = os.path.join(flair_path, f)
        # We use a fast read here; if it fails, 0 intensity
        try:
            # Use the robust reader
            img = read_dicom_file(f_path)
            # Use mean intensity of the slice as the signal
            val = np.mean(img)
            intensities.append(val)
            valid_indices.append(i)
        except Exception:
            intensities.append(0)
            valid_indices.append(i)

    intensities = np.array(intensities)

    # Find peaks
    # height=np.max(intensities)*0.1 ensures we don't pick background noise
    peaks, properties = find_peaks(
        intensities, distance=min_distance, height=np.max(intensities) * 0.1
    )

    if len(peaks) == 0:
        # Fallback: just take the max intensity slice
        peaks = [np.argmax(intensities)]

    # Sort peaks by peak height (prominence)
    # If properties is empty or not enough info, sort by intensity value at peak
    peak_heights = intensities[peaks]
    sorted_peak_indices = np.argsort(peak_heights)[::-1]  # Descending
    sorted_peaks = peaks[sorted_peak_indices]

    # Select top K
    selected_indices = sorted_peaks[:num_instances].tolist()

    # Pad if we don't have enough peaks
    while len(selected_indices) < num_instances:
        # Pad with the best peak repeated (or 0 if empty)
        if len(selected_indices) > 0:
            selected_indices.append(selected_indices[0])
        else:
            selected_indices.append(len(files) // 2)  # Middle slice fallback

    # Map back to file filenames or just keep 0-based index relative to sorted file list
    # We will use 0-based index relative to the sorted file list in the Dataset
    return selected_indices


def generate_roi_cache(dfs, load_cached_data=True):
    """
    Generates or loads a cache of ROI indices for all subjects.

    Args:
        dfs (list): List of DataFrames (train, val, test).
        load_cached_data (bool): Whether to attempt loading from disk.

    Returns:
        pd.DataFrame: DataFrame containing BraTS21ID and list of selected indices.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "roi_cache.parquet")

    if load_cached_data and os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass  # Fallback to re-computing

    # Combine all unique IDs to process
    all_ids = []
    for df in dfs:
        if df is not None and not df.empty:
            all_ids.extend(df.to_dict("records"))

    # Remove duplicates based on BraTS21ID
    unique_subjects = {x["BraTS21ID"]: x for x in all_ids}.values()

    results = []
    for subject in unique_subjects:
        indices = compute_flair_peaks(
            subject, num_instances=Config.NUM_INSTANCES, min_distance=10
        )
        results.append({"BraTS21ID": subject["BraTS21ID"], "roi_indices": indices})

    cache_df = pd.DataFrame(results)

    # Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_df.to_parquet(cache_path, index=False)

    return cache_df


class MGMTDataset(Dataset):
    def __init__(self, df, roi_cache_df, transform=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata for the split.
            roi_cache_df (pd.DataFrame): DataFrame with 'BraTS21ID' and 'roi_indices'.
            transform (albumentations.Compose): Augmentation pipeline.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        # Convert cache to dictionary for O(1) lookup
        self.roi_map = roi_cache_df.set_index("BraTS21ID")["roi_indices"].to_dict()
        self.transform = transform
        self.mode = mode
        self.modalities = Config.MODALITIES  # ["FLAIR", "T1w", "T1wCE", "T2w"]
        self.stride = Config.STACK_STRIDE

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subject_id = row["BraTS21ID"]

        # Get target label (if available)
        if "MGMT_value" in row:
            target = torch.tensor(row["MGMT_value"], dtype=torch.float32)
        else:
            target = torch.tensor(-1.0, dtype=torch.float32)  # Test set

        # Get ROI indices (centers)
        # Default to middle if not in cache for some reason
        center_indices = self.roi_map.get(subject_id, [50] * Config.NUM_INSTANCES)

        # Prepare container for instances: (Num_Instances, Channels, H, W)
        # Channels = 4 modalities * 3 slices = 12
        instances_list = []

        for center_idx in center_indices:
            # Define slice offsets: e.g., [-5, 0, +5]
            offsets = [-self.stride, 0, self.stride]

            # Collect all channels for this instance
            instance_channels = []

            for mod in self.modalities:
                mod_path_rel = row[f"path_{mod}"]
                mod_full_path = os.path.join(Config.INPUT_DIR, mod_path_rel)
                sorted_files = get_sorted_files(mod_full_path)
                num_files = len(sorted_files)

                for off in offsets:
                    target_idx = center_idx + off

                    # Clamp index
                    target_idx = max(0, min(target_idx, num_files - 1))

                    if num_files > 0:
                        file_name = sorted_files[target_idx]
                        file_path = os.path.join(mod_full_path, file_name)
                        img = read_dicom_file(file_path)
                    else:
                        img = np.zeros(
                            (Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint16
                        )

                    # Resize to ensure consistency (dicom_utils returns original size)
                    if (
                        img.shape[0] != Config.IMG_SIZE
                        or img.shape[1] != Config.IMG_SIZE
                    ):
                        img = cv2.resize(
                            img,
                            (Config.IMG_SIZE, Config.IMG_SIZE),
                            interpolation=cv2.INTER_LINEAR,
                        )

                    # Normalize
                    img = normalize_min_max(img)
                    instance_channels.append(img)

            # Stack channels: (12, H, W)
            instance_volume = np.stack(instance_channels, axis=0)

            # Apply Augmentations
            # Albumentations expects (H, W, C)
            if self.transform:
                # Transpose to (H, W, 12)
                instance_volume_hwc = np.transpose(instance_volume, (1, 2, 0))
                augmented = self.transform(image=instance_volume_hwc)["image"]
                # ToTensorV2 converts to tensor and HWC->CHW, but we might want manual control
                # If ToTensorV2 is not last, we get numpy.
                # Let's assume transform returns tensor or numpy.

                if isinstance(augmented, torch.Tensor):
                    instance_volume = augmented  # Already (C, H, W) via ToTensorV2
                else:
                    # Back to (C, H, W)
                    instance_volume = np.transpose(augmented, (2, 0, 1))
                    instance_volume = torch.from_numpy(instance_volume).float()
            else:
                instance_volume = torch.from_numpy(instance_volume).float()

            instances_list.append(instance_volume)

        # Stack instances: (Num_Instances, C, H, W)
        # e.g. (3, 12, 256, 256)
        batch_tensor = torch.stack(instances_list, dim=0)

        return batch_tensor, target


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get PyTorch DataLoaders for train, val, and test.
    """
    # 1. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    df_val = pd.read_csv(Config.VAL_METADATA)
    df_test = pd.read_csv(Config.TEST_METADATA)

    # 2. Generate/Load ROI Cache
    # We pass all dataframes so we can compute peaks for test set too
    roi_cache = generate_roi_cache(
        [df_train, df_val, df_test], load_cached_data=load_cached_data
    )

    # 3. Define Transforms
    # Geometric augmentations for training
    train_transform = A.Compose(
        [
            A.Rotate(limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # Resize is handled in __getitem__ explicitly before stacking,
            # but we can add it here as a safety net or for other augs.
            # However, since we stack 12 channels, we must ensure albumentations handles it.
            # Albumentations supports arbitrary channels.
            ToTensorV2(),
        ]
    )

    # No geometric augs for val/test, just tensor conversion
    val_transform = A.Compose([ToTensorV2()])

    # 4. Create Datasets
    train_dataset = MGMTDataset(
        df_train, roi_cache, transform=train_transform, mode="train"
    )
    val_dataset = MGMTDataset(df_val, roi_cache, transform=val_transform, mode="val")
    test_dataset = MGMTDataset(df_test, roi_cache, transform=val_transform, mode="test")

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Avoid incomplete batches for BatchNorm stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
