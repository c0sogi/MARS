import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import (
    INPUT_DIR,
    CACHE_DIR,
    TRAIN_METADATA,
    VAL_METADATA,
    TEST_METADATA,
    Z_DIM,
    PATCH_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
)
from library.utils import set_seed


class InkDataset(Dataset):
    """
    PyTorch Dataset for Ink Detection.
    Loads pre-processed .npy files from the cache directory.
    """

    def __init__(self, metadata_df, cache_dir, mode="train", transform=None):
        self.metadata = metadata_df
        self.cache_dir = cache_dir
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        sample_id = row.sample_id

        # Paths to cached files
        vol_path = os.path.join(self.cache_dir, f"{sample_id}_vol.npy")

        # Load volume (uint8) and normalize to float32 [0, 1]
        try:
            volume = np.load(vol_path)
            volume = volume.astype(np.float32) / 255.0
        except FileNotFoundError:
            # Fallback or error handling if cache is corrupted
            raise FileNotFoundError(f"Cached file not found: {vol_path}")

        if self.mode in ["train", "val"]:
            mask_path = os.path.join(self.cache_dir, f"{sample_id}_mask.npy")
            label = np.load(mask_path)
            label = label.astype(np.float32)
            # Add channel dimension to label: (H, W) -> (1, H, W)
            label = np.expand_dims(label, axis=0)

            return torch.from_numpy(volume), torch.from_numpy(label)

        else:
            # Test mode: return volume and sample_id for identification
            return torch.from_numpy(volume), sample_id


def preprocess_data(df, split_name, load_cached_data=True):
    """
    Pre-processes raw TIFF slices into cropped, resized, and stacked .npy files.
    Implements the required caching logic.
    """
    cache_subdir = os.path.join(CACHE_DIR, split_name)
    os.makedirs(cache_subdir, exist_ok=True)

    # 1. Check if cache exists
    expected_vol_files = [
        os.path.join(cache_subdir, f"{row.sample_id}_vol.npy")
        for row in df.itertuples()
    ]

    # If we are in train/val mode, we also expect masks
    has_labels = (
        "inklabels_path" in df.columns and not df["inklabels_path"].isna().all()
    )

    if load_cached_data:
        all_exist = all(os.path.exists(f) for f in expected_vol_files)
        if has_labels:
            expected_mask_files = [
                os.path.join(cache_subdir, f"{row.sample_id}_mask.npy")
                for row in df.itertuples()
            ]
            all_exist = all_exist and all(
                os.path.exists(f) for f in expected_mask_files
            )

        if all_exist:
            print(f"Loading cached data for {split_name} from {cache_subdir}...")
            return cache_subdir

    print(f"Processing data for {split_name} (Cache miss or overwrite)...")

    # 2. Process data grouped by fragment to minimize I/O
    # Group by fragment_id to load each slice only once per fragment
    grouped = df.groupby("fragment_id")

    for fragment_id, group in grouped:
        # Prepare buffers for all patches in this fragment
        # Key: sample_id, Value: 3D array (Z, H, W)
        # We use uint8 to save memory during processing
        vol_buffers = {
            row.sample_id: np.zeros((Z_DIM, PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
            for row in group.itertuples()
        }

        # Get paths from the first sample (assuming all in group share the same volume path)
        first_row = group.iloc[0]
        surface_dir = os.path.join(INPUT_DIR, first_row.surface_volume_path)

        # Iterate through Z slices
        for z in range(Z_DIM):
            slice_path = os.path.join(surface_dir, f"{z:02d}.tif")
            if not os.path.exists(slice_path):
                print(f"Warning: Slice {slice_path} not found. Skipping.")
                continue

            # Load full slice
            img_slice = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)
            if img_slice is None:
                continue

            # Crop and resize for each patch
            for row in group.itertuples():
                # Crop
                crop = img_slice[row.y : row.y + row.h, row.x : row.x + row.w]

                # Resize if necessary (e.g. edge patches or if PATCH_SIZE differs from crop)
                if crop.shape[0] != PATCH_SIZE or crop.shape[1] != PATCH_SIZE:
                    crop = cv2.resize(
                        crop, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_LINEAR
                    )

                vol_buffers[row.sample_id][z] = crop

        # Save volumes
        for row in group.itertuples():
            np.save(
                os.path.join(cache_subdir, f"{row.sample_id}_vol.npy"),
                vol_buffers[row.sample_id],
            )

        # Process Labels (only if available)
        if has_labels:
            # Load ink label image for the fragment
            ink_path_rel = first_row.inklabels_path
            if pd.isna(ink_path_rel):
                continue

            ink_path = os.path.join(INPUT_DIR, ink_path_rel)
            if os.path.exists(ink_path):
                ink_img = cv2.imread(ink_path, cv2.IMREAD_GRAYSCALE)

                for row in group.itertuples():
                    # Crop
                    mask_crop = ink_img[row.y : row.y + row.h, row.x : row.x + row.w]

                    # Resize (Nearest Neighbor for masks to keep binary)
                    if (
                        mask_crop.shape[0] != PATCH_SIZE
                        or mask_crop.shape[1] != PATCH_SIZE
                    ):
                        mask_crop = cv2.resize(
                            mask_crop,
                            (PATCH_SIZE, PATCH_SIZE),
                            interpolation=cv2.INTER_NEAREST,
                        )

                    # Binarize
                    mask_crop = (mask_crop > 0).astype(np.uint8)

                    np.save(
                        os.path.join(cache_subdir, f"{row.sample_id}_mask.npy"),
                        mask_crop,
                    )

    print(f"Finished processing {split_name}.")
    return cache_subdir


def get_dataloaders(
    train_csv=TRAIN_METADATA,
    val_csv=VAL_METADATA,
    test_csv=TEST_METADATA,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    load_cached_data=True,
    limit_samples=None,
):
    """
    Creates DataLoaders for train, val, and test sets.
    Handles data preprocessing and caching.
    """
    set_seed(42)  # Ensure reproducibility

    # Load Metadata
    df_train = pd.read_csv(train_csv) if os.path.exists(train_csv) else pd.DataFrame()
    df_val = pd.read_csv(val_csv) if os.path.exists(val_csv) else pd.DataFrame()
    df_test = pd.read_csv(test_csv) if os.path.exists(test_csv) else pd.DataFrame()

    # Limit samples for debugging if requested
    if limit_samples is not None:
        if not df_train.empty:
            df_train = df_train.iloc[:limit_samples]
        if not df_val.empty:
            df_val = df_val.iloc[:limit_samples]
        if not df_test.empty:
            df_test = df_test.iloc[:limit_samples]

    dataloaders = {}

    # --- Train ---
    if not df_train.empty:
        cache_train = preprocess_data(
            df_train, "train", load_cached_data=load_cached_data
        )
        ds_train = InkDataset(df_train, cache_train, mode="train")
        dataloaders["train"] = DataLoader(
            ds_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    # --- Val ---
    if not df_val.empty:
        cache_val = preprocess_data(df_val, "val", load_cached_data=load_cached_data)
        ds_val = InkDataset(df_val, cache_val, mode="val")
        dataloaders["val"] = DataLoader(
            ds_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    # --- Test ---
    if not df_test.empty:
        cache_test = preprocess_data(df_test, "test", load_cached_data=load_cached_data)
        ds_test = InkDataset(df_test, cache_test, mode="test")
        dataloaders["test"] = DataLoader(
            ds_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return dataloaders
