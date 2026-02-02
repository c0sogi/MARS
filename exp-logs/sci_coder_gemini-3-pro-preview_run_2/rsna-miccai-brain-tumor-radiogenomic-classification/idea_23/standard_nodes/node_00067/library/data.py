import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import (
    read_dicom_robust,
    normalize_channel,
    get_normalized_sum_anchor,
)


def get_roi_cache(df, cache_name, load_cached_data=True):
    """
    Computes or loads the ROI anchor indices for the dataset.

    Args:
        df (pd.DataFrame): Metadata dataframe containing BraTS21ID and path_FLAIR.
        cache_name (str): Filename for the cache (e.g., 'train_roi_cache.parquet').
        load_cached_data (bool): Whether to attempt loading from disk.

    Returns:
        pd.DataFrame: Dataframe with 'BraTS21ID' and 'anchor_idx'.
    """
    cache_path = os.path.join(Config.WORKING_DIR, cache_name)

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            cached_df = pd.read_parquet(cache_path)
            # Verify it covers the current dataframe
            if set(df["BraTS21ID"].unique()).issubset(
                set(cached_df["BraTS21ID"].unique())
            ):
                print(f"Loaded ROI cache from {cache_path}")
                return cached_df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Computing ROI anchors for {len(df)} subjects...")
    results = []

    # Iterate over unique subjects to avoid redundant computation
    unique_subjects = df[["BraTS21ID", "path_FLAIR"]].drop_duplicates()

    for idx, row in unique_subjects.iterrows():
        subject_id = row["BraTS21ID"]
        flair_rel_path = row["path_FLAIR"]
        flair_full_path = os.path.join(Config.INPUT_DIR, flair_rel_path)

        # Use the utility function to find the robust anchor
        # Cite Lesson 00038: Using Sum Intensity instead of Max
        anchor_idx = get_normalized_sum_anchor(flair_full_path)

        results.append({"BraTS21ID": subject_id, "anchor_idx": anchor_idx})

    roi_df = pd.DataFrame(results)

    # 3. Save to cache
    try:
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        roi_df.to_parquet(cache_path, index=False)
        print(f"Saved ROI cache to {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")

    return roi_df


class MGMTDataset(Dataset):
    def __init__(self, df, roi_df, transform=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            roi_df (pd.DataFrame): Dataframe containing 'BraTS21ID' and 'anchor_idx'.
            transform (A.Compose, optional): Albumentations transforms.
        """
        self.df = df.reset_index(drop=True)
        # Create a mapping for O(1) access
        self.roi_map = dict(zip(roi_df["BraTS21ID"], roi_df["anchor_idx"]))
        self.transform = transform

        # Define modalities in order.
        # Crucial: This order must match the Grouped Convolution expectation.
        # Channels 0-2: FLAIR, 3-5: T1w, 6-8: T1wCE, 9-11: T2w
        self.modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]

    def __len__(self):
        return len(self.df)

    def _get_sorted_files(self, dir_path):
        """Helper to list and sort DICOM files numerically."""
        if not os.path.exists(dir_path):
            return []

        files = [f for f in os.listdir(dir_path) if f.endswith(".dcm")]

        # Sort numerically to match the logic in get_downsampled_max_anchor
        try:
            files.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
        except Exception:
            files.sort()

        return files

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subject_id = row["BraTS21ID"]

        # Get ground truth label (if available, else 0.0 for test)
        label = row.get("MGMT_value", 0.0)

        # 1. Determine Slice Indices
        anchor_idx = self.roi_map.get(subject_id, 0)
        stride = Config.STRIDE

        # We want: [Anchor - Stride, Anchor, Anchor + Stride]
        target_indices = [anchor_idx - stride, anchor_idx, anchor_idx + stride]

        channels = []

        # 2. Extract Slices for each Modality
        for mod in self.modalities:
            # Construct path
            mod_path = os.path.join(Config.INPUT_DIR, row[f"path_{mod}"])
            files = self._get_sorted_files(mod_path)
            num_files = len(files)

            for slice_idx in target_indices:
                # Edge Clamping
                if num_files == 0:
                    # Missing modality: use black image
                    img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)
                else:
                    # Clamp index to valid range
                    clamped_idx = max(0, min(slice_idx, num_files - 1))
                    file_path = os.path.join(mod_path, files[clamped_idx])

                    # Read and Preprocess
                    img = read_dicom_robust(file_path)  # Returns float32
                    img = normalize_channel(img)  # Min-Max -> [0, 1]

                    # Resize to target size (Area interpolation for downsampling/resizing)
                    try:
                        img = cv2.resize(
                            img,
                            (Config.IMG_SIZE, Config.IMG_SIZE),
                            interpolation=cv2.INTER_AREA,
                        )
                    except Exception:
                        img = np.zeros(
                            (Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
                        )

                channels.append(img)

        # 3. Stack Channels
        # Result shape: (H, W, C) -> (224, 224, 12)
        image = np.stack(channels, axis=-1)

        # 4. Apply Augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Convert to tensor manually if no transform provided
            image = torch.from_numpy(image.transpose(2, 0, 1))  # HWC -> CHW

        return image.float(), torch.tensor(label, dtype=torch.float32)


def get_transforms(phase):
    """
    Returns Albumentations transforms for the specified phase.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Rotate +/- 15 degrees.
                # Cite Lesson 00048: Use reflection padding to match the "Current Best" configuration
                # which achieved higher metric despite theoretical instability.
                A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),
                ToTensorV2(),  # Converts HWC to CHW and to Tensor
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


def get_train_val_datasets(load_cached_data=True):
    """
    Factory function to create training and validation datasets.
    """
    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Combine for cache computation to ensure all IDs are covered
    full_df = pd.concat([train_df, val_df], ignore_index=True)

    # Compute/Load ROI Cache
    roi_df = get_roi_cache(
        full_df, "roi_cache.parquet", load_cached_data=load_cached_data
    )

    # Create Datasets
    train_dataset = MGMTDataset(train_df, roi_df, transform=get_transforms("train"))

    val_dataset = MGMTDataset(val_df, roi_df, transform=get_transforms("val"))

    return train_dataset, val_dataset


def get_test_dataset(load_cached_data=True):
    """
    Factory function to create test dataset.
    """
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Compute/Load ROI Cache for test set
    roi_df = get_roi_cache(
        test_df, "test_roi_cache.parquet", load_cached_data=load_cached_data
    )

    test_dataset = MGMTDataset(test_df, roi_df, transform=get_transforms("test"))

    return test_dataset
