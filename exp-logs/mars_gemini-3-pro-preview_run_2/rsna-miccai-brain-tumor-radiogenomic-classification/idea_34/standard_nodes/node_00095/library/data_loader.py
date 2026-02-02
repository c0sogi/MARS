import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.dicom_utils import read_dicom_robust, map_slice_ids, get_slice_id


class FidelityPreprocessing:
    """
    Handles the 'Fidelity-Aligned' ROI selection logic.
    Identifies the 'Single Dominant Reference' (Anchor ID) based on FLAIR intensity integral.
    Caches results to avoid re-computation.
    """

    @staticmethod
    def get_cache_path(split_name):
        return os.path.join(Config.CACHE_DIR, f"roi_cache_{split_name}.parquet")

    @staticmethod
    def compute_anchor(subject_path_flair):
        """
        Scans FLAIR slices in the 15-85% range to find the slice with max intensity sum.
        Returns the integer Slice ID.
        """
        # 1. Map all slice IDs
        slice_map = map_slice_ids(subject_path_flair)
        if not slice_map:
            return -1  # Error case

        sorted_ids = sorted(slice_map.keys())
        num_slices = len(sorted_ids)

        # 2. Define Search Range
        start_idx = int(num_slices * Config.ROI_SEARCH_START)
        end_idx = int(num_slices * Config.ROI_SEARCH_END)

        # Ensure valid range
        if start_idx >= end_idx:
            start_idx = 0
            end_idx = num_slices

        search_ids = sorted_ids[start_idx:end_idx]

        max_intensity = -1.0
        best_id = sorted_ids[num_slices // 2]  # Default to middle if search fails

        # 3. Iterate and find max integral
        # We use the robust reader but don't resize yet to save some time,
        # or resize to small fixed size for speed if IO is bottleneck.
        # Here we read full size as per "Fidelity" requirement, but calculation is simple sum.
        for sid in search_ids:
            path = slice_map[sid]
            # Read raw
            img = read_dicom_robust(
                path, target_size=(128, 128)
            )  # Small resize for speed in heuristic
            current_intensity = np.sum(img)

            if current_intensity > max_intensity:
                max_intensity = current_intensity
                best_id = sid

        return best_id

    @classmethod
    def process_and_cache(cls, df, split_name, load_cached_data=True):
        """
        Iterates through the dataframe. If cache exists and load_cached_data is True, returns it.
        Otherwise, computes anchor IDs for all subjects and saves to parquet.
        """
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        cache_path = cls.get_cache_path(split_name)

        # 1. Try Load
        if load_cached_data and os.path.exists(cache_path):
            try:
                print(f"Loading ROI cache for {split_name} from {cache_path}...")
                cached_df = pd.read_parquet(cache_path)
                # Verify alignment
                if len(cached_df) == len(df) and np.all(
                    cached_df["BraTS21ID"].values == df["BraTS21ID"].values
                ):
                    return cached_df
                else:
                    print("Cache mismatch. Recomputing...")
            except Exception as e:
                print(f"Error loading cache: {e}. Recomputing...")

        # 2. Compute
        print(f"Computing ROI Anchors for {split_name} ({len(df)} subjects)...")
        anchor_ids = []

        for idx, row in df.iterrows():
            # Construct full path to FLAIR
            flair_path = os.path.join(Config.INPUT_DIR, row["path_FLAIR"])
            anchor = cls.compute_anchor(flair_path)
            anchor_ids.append(anchor)

            if idx % 50 == 0 and idx > 0:
                print(f"  Processed {idx}/{len(df)}")

        # 3. Save
        result_df = df.copy()
        result_df["anchor_id"] = anchor_ids

        # Save only ID and anchor to save space, or full DF?
        # Returning full DF with anchor column is easiest for Dataset
        cache_data = result_df[["BraTS21ID", "anchor_id"]]
        cache_data.to_parquet(cache_path, index=False)

        return result_df


class BraTSDataset(Dataset):
    def __init__(self, df, transform=None, mode="train"):
        self.df = df
        self.transform = transform
        self.mode = mode

        # Modality order matches the Grouped Conv expectations
        self.modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        anchor_id = row["anchor_id"]

        # Define target slice IDs based on stride
        # Stride 5: [ID-5, ID, ID+5]
        offsets = [-Config.ROI_STRIDE, 0, Config.ROI_STRIDE]
        target_ids = [anchor_id + off for off in offsets]

        # Prepare container for 12 channels
        # Shape: (12, H, W)
        channels = []

        for mod in self.modalities:
            # 1. Map files for this modality
            mod_path = os.path.join(Config.INPUT_DIR, row[f"path_{mod}"])
            slice_map = map_slice_ids(mod_path)
            valid_ids = np.array(sorted(slice_map.keys()))

            # 2. Retrieve slices
            for tid in target_ids:
                read_id = tid

                # Edge Clamping / Nearest Neighbor
                if tid not in slice_map:
                    if len(valid_ids) > 0:
                        # Find nearest
                        nearest_idx = (np.abs(valid_ids - tid)).argmin()
                        read_id = valid_ids[nearest_idx]
                    else:
                        # Modality empty or missing
                        read_id = -1

                if read_id != -1 and read_id in slice_map:
                    img = read_dicom_robust(
                        slice_map[read_id], target_size=Config.IMG_SIZE
                    )
                else:
                    # Fallback for completely missing data
                    img = np.zeros(Config.IMG_SIZE, dtype=np.float32)

                # 3. Independent Per-Channel Min-Max Normalization
                min_val = img.min()
                max_val = img.max()
                if max_val - min_val > 1e-6:
                    img = (img - min_val) / (max_val - min_val)
                else:
                    img = np.zeros_like(img)  # Avoid division by zero, flat signal

                channels.append(img)

        # Stack to (H, W, 12) for Albumentations
        # Albumentations expects (H, W, C)
        volume = np.stack(channels, axis=-1)

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=volume)
            volume = augmented["image"]  # Returns Tensor (C, H, W) if ToTensorV2 used
        else:
            # Manual ToTensor if no transform provided (e.g. test w/o TTA)
            volume = torch.from_numpy(volume.transpose(2, 0, 1))

        if self.mode == "test":
            return volume, row["BraTS21ID"]
        else:
            label = torch.tensor(row["MGMT_value"], dtype=torch.float32)
            return volume, label


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Main entry point to get dataloaders.
    Handles metadata loading, preprocessing (caching), and dataset creation.
    """

    # 1. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    df_val = pd.read_csv(Config.VAL_METADATA)
    df_test = pd.read_csv(Config.TEST_METADATA)

    # 2. Run Preprocessing (Fidelity Anchor Search)
    # We merge results back into the dataframes

    # Train
    train_cache = FidelityPreprocessing.process_and_cache(
        df_train, "train", load_cached_data
    )
    df_train = df_train.merge(
        train_cache[["BraTS21ID", "anchor_id"]], on="BraTS21ID", how="left"
    )

    # Val
    val_cache = FidelityPreprocessing.process_and_cache(df_val, "val", load_cached_data)
    df_val = df_val.merge(
        val_cache[["BraTS21ID", "anchor_id"]], on="BraTS21ID", how="left"
    )

    # Test
    test_cache = FidelityPreprocessing.process_and_cache(
        df_test, "test", load_cached_data
    )
    df_test = df_test.merge(
        test_cache[["BraTS21ID", "anchor_id"]], on="BraTS21ID", how="left"
    )

    # 3. Define Transforms
    # Note: Reflection padding is critical for rotation to avoid edge artifacts
    train_transform = A.Compose(
        [
            A.Rotate(
                limit=Config.ROTATION_DEGREES, border_mode=cv2.BORDER_REFLECT, p=0.5
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose([ToTensorV2()])

    # 4. Create Datasets
    train_ds = BraTSDataset(df_train, transform=train_transform, mode="train")
    val_ds = BraTSDataset(df_val, transform=val_transform, mode="val")
    test_ds = BraTSDataset(df_test, transform=val_transform, mode="test")

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batch to stabilize BatchNorm
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
