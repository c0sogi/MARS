import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import robust_normalize, rle_decode

# Ensure OpenCV doesn't use multithreading which conflicts with PyTorch DataLoader
cv2.setNumThreads(0)


class VolumetricDataset(Dataset):
    """
    PyTorch Dataset for 3D Volumetric Data.
    Loads pre-processed .npy volumes and applies 3D patch sampling.
    """

    def __init__(self, metadata_df, subset="train", transform=None):
        """
        Args:
            metadata_df (pd.DataFrame): Dataframe containing paths to cached .npy files.
                                        Expected columns: ['case', 'day', 'npy_path']
            subset (str): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.metadata = metadata_df
        self.subset = subset
        self.transform = transform

        # Hyperparameters from Config
        self.patch_size = Config.PATCH_SIZE  # (D, H, W)
        self.spatial_size = Config.SPATIAL_SIZE  # (H, W)
        self.num_classes = Config.NUM_CLASSES

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        npy_path = row["npy_path"]

        # Load cached volume
        # Data is stored as a dictionary: {'image': ..., 'mask': ...}
        try:
            data = np.load(npy_path, allow_pickle=True).item()
        except Exception as e:
            raise RuntimeError(f"Failed to load {npy_path}: {e}")

        image_vol = data["image"]  # Shape: (D, H, W)

        # Handle masks
        if "mask" in data and data["mask"] is not None:
            mask_vol = data["mask"]  # Shape: (D, H, W, C)
        else:
            # Create dummy mask for test set or missing masks
            # Shape: (D, H, W, 3)
            mask_vol = np.zeros(image_vol.shape + (self.num_classes,), dtype=np.float32)

        # --- Patch Sampling / Processing ---
        D, H, W = image_vol.shape
        target_D, target_H, target_W = self.patch_size

        if self.subset == "train":
            # 1. Padding if depth is too small
            if D < target_D:
                pad_d = target_D - D
                # Pad with zeros (constant).
                # Image: (D, H, W) -> Pad D
                image_vol = np.pad(
                    image_vol, ((0, pad_d), (0, 0), (0, 0)), mode="constant"
                )
                # Mask: (D, H, W, C) -> Pad D
                mask_vol = np.pad(
                    mask_vol, ((0, pad_d), (0, 0), (0, 0), (0, 0)), mode="constant"
                )
                D = target_D  # Update D

            # 2. Random Crop in Depth
            if D > target_D:
                start_d = np.random.randint(0, D - target_D + 1)
            else:
                start_d = 0

            end_d = start_d + target_D

            # Extract crops
            img_crop = image_vol[start_d:end_d, :, :]
            mask_crop = mask_vol[start_d:end_d, :, :]

        else:
            # Validation/Test: Return full volume
            # We do not crop. The DataLoader batch_size must be 1.
            img_crop = image_vol
            mask_crop = mask_vol

        # --- Tensor Conversion ---
        # Image: (D, H, W) -> (C, D, H, W) where C=1
        img_tensor = torch.from_numpy(img_crop).float().unsqueeze(0)

        # Mask: (D, H, W, C) -> (C, D, H, W)
        # Permute channel to be first
        mask_tensor = torch.from_numpy(mask_crop).float().permute(3, 0, 1, 2)

        # Ensure mask is binary (0 or 1)
        mask_tensor = (mask_tensor > 0.5).float()

        return {
            "id": f"{row['case']}_{row['day']}",
            "image": img_tensor,
            "mask": mask_tensor,
            "original_shape": torch.tensor([D, H, W]),  # Useful for reconstruction
        }


def process_and_cache_volume(group_df, case_id, day_id, output_dir):
    """
    Processes a single volume (Case, Day) from raw slices and saves it as .npy.
    Handles loading, resizing, stacking, and normalization.
    """
    # Sort slices by ID to ensure correct 3D order
    group_df = group_df.copy()
    # slice column is string '0001', convert to int for sorting
    group_df["slice_idx"] = group_df["slice"].astype(int)
    group_df = group_df.sort_values("slice_idx")

    images = []
    masks = []

    has_masks = "large_bowel" in group_df.columns

    for _, row in group_df.iterrows():
        # Load Image
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        if not os.path.exists(img_path):
            continue

        # Read image (16-bit or 8-bit)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        # Handle multi-channel (rare) -> convert to gray
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize Image (Bilinear)
        img = cv2.resize(img, Config.SPATIAL_SIZE, interpolation=cv2.INTER_LINEAR)
        images.append(img)

        # Load Masks if available
        if has_masks:
            slice_masks = []
            for cls in Config.CLASSES:
                rle = row[cls]
                # rle_decode returns (H, W)
                mask = rle_decode(rle, shape=(row["height"], row["width"]))
                slice_masks.append(mask)

            # Stack classes: (H, W, 3)
            mask_stacked = np.stack(slice_masks, axis=-1)

            # Resize mask (Nearest Neighbor to preserve binary classes)
            mask_resized = cv2.resize(
                mask_stacked, Config.SPATIAL_SIZE, interpolation=cv2.INTER_NEAREST
            )

            # If resize removed the last dim (if 1 class), add it back.
            if mask_resized.ndim == 2:
                mask_resized = mask_resized[..., np.newaxis]

            masks.append(mask_resized)

    if not images:
        return None

    # Stack into Volume
    # Image: (D, H, W)
    vol_img = np.stack(images, axis=0)

    # Normalize Volume (Robust Percentile)
    vol_img = robust_normalize(vol_img)

    vol_mask = None
    if masks:
        # Mask: (D, H, W, 3)
        vol_mask = np.stack(masks, axis=0)

    # Save to NPY
    save_path = os.path.join(output_dir, f"{case_id}_{day_id}.npy")
    data_dict = {"image": vol_img, "mask": vol_mask}
    np.save(save_path, data_dict)

    return save_path


def preprocess_dataset(metadata_path, subset_name, load_cached=True):
    """
    Groups metadata by (Case, Day), processes volumes, and caches them.
    Returns a DataFrame with paths to cached files.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_index_path = os.path.join(
        Config.CACHE_DIR, f"{subset_name}_processed.parquet"
    )

    # 1. Try Loading Cached Index
    if load_cached and os.path.exists(cache_index_path):
        print(f"Loading cached {subset_name} data from {cache_index_path}")
        return pd.read_parquet(cache_index_path)

    print(f"Processing {subset_name} data from scratch...")

    # 2. Load Metadata
    if not os.path.exists(metadata_path):
        # Fallback for test if file doesn't exist yet (though it should)
        print(f"Warning: Metadata file {metadata_path} not found.")
        return pd.DataFrame()

    df = pd.read_csv(metadata_path, keep_default_na=False)

    # Handle DEBUG mode: reduce dataset size
    if Config.DEBUG:
        cases = df["case"].unique()[:2]  # Take first 2 cases
        df = df[df["case"].isin(cases)]
        print(f"DEBUG MODE: Processing subset of {len(df)} slices.")

    # 3. Group by Case + Day
    groups = df.groupby(["case", "day"])
    processed_records = []

    for (case, day), group in groups:
        npy_name = f"{case}_{day}.npy"
        npy_path = os.path.join(Config.CACHE_DIR, npy_name)

        # If caching is enabled and file exists, skip processing
        if load_cached and os.path.exists(npy_path):
            pass
        else:
            # Process and cache
            saved_path = process_and_cache_volume(group, case, day, Config.CACHE_DIR)
            if saved_path is None:
                continue
            npy_path = saved_path

        processed_records.append({"case": case, "day": day, "npy_path": npy_path})

    # 4. Save Index
    result_df = pd.DataFrame(processed_records)
    if not result_df.empty:
        result_df.to_parquet(cache_index_path)

    print(f"Processed {len(result_df)} volumes for {subset_name}.")
    return result_df


def create_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    # 1. Preprocess/Load Data
    train_df = preprocess_dataset(
        Config.TRAIN_METADATA, "train", load_cached=load_cached_data
    )
    val_df = preprocess_dataset(
        Config.VAL_METADATA, "val", load_cached=load_cached_data
    )
    test_df = preprocess_dataset(
        Config.TEST_METADATA, "test", load_cached=load_cached_data
    )

    # 2. Create Datasets
    train_ds = VolumetricDataset(train_df, subset="train")
    val_ds = VolumetricDataset(val_df, subset="val")
    test_ds = VolumetricDataset(test_df, subset="test")

    # 3. Create DataLoaders
    # Train: Batch size > 1, Shuffle, Drop Last
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Val/Test: Batch size = 1 (variable depth), No Shuffle
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
