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


class SliceDataset(Dataset):
    """
    2.5D Slice Dataset.
    Loads volumes into memory and yields (3, H, W) slices where channels are z-1, z, z+1.
    Cite solution_lesson_node_00026: Prioritize 2.5D Slice Stacking.
    """

    def __init__(self, metadata_df, subset="train"):
        self.metadata = metadata_df
        self.subset = subset
        self.volumes = {}
        self.samples = []  # List of (case_day_key, z_index)

        print(f"Initializing SliceDataset for {subset}...")

        # Load all volumes into RAM
        for _, row in self.metadata.iterrows():
            key = f"{row['case']}_{row['day']}"
            npy_path = row["npy_path"]

            try:
                data = np.load(npy_path, allow_pickle=True).item()
                image = data["image"]  # (D, H, W)

                if "mask" in data and data["mask"] is not None:
                    mask = data["mask"]  # (D, H, W, C)
                else:
                    mask = None

                self.volumes[key] = {"image": image, "mask": mask}

                # Create samples for each slice
                D = image.shape[0]
                for z in range(D):
                    self.samples.append((key, z))

            except Exception as e:
                print(f"Error loading {npy_path}: {e}")

        print(f"Loaded {len(self.volumes)} volumes, {len(self.samples)} slices.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key, z = self.samples[idx]
        vol_data = self.volumes[key]
        image_vol = vol_data["image"]
        mask_vol = vol_data["mask"]

        D, H, W = image_vol.shape

        # 2.5D Stacking: z-1, z, z+1
        # Handle boundaries by replication
        indices = [max(0, z - 1), z, min(D - 1, z + 1)]

        # Stack slices -> (3, H, W)
        img_stack = image_vol[indices, :, :]

        # Mask -> (C, H, W)
        if mask_vol is not None:
            mask_slice = mask_vol[z, :, :, :]  # (H, W, C)
            mask_tensor = (
                torch.from_numpy(mask_slice).float().permute(2, 0, 1)
            )  # (C, H, W)
            mask_tensor = (mask_tensor > 0.5).float()
        else:
            mask_tensor = torch.zeros((Config.NUM_CLASSES, H, W), dtype=torch.float32)

        img_tensor = torch.from_numpy(img_stack).float()

        return {"image": img_tensor, "mask": mask_tensor, "id": f"{key}_slice_{z:04d}"}


class VolumetricDataset(Dataset):
    """
    Dataset for Validation/Testing. Returns full volumes for reconstruction.
    """

    def __init__(self, metadata_df, subset="val"):
        self.metadata = metadata_df
        self.subset = subset
        self.num_classes = Config.NUM_CLASSES

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        npy_path = row["npy_path"]

        try:
            data = np.load(npy_path, allow_pickle=True).item()
        except Exception as e:
            raise RuntimeError(f"Failed to load {npy_path}: {e}")

        image_vol = data["image"]  # (D, H, W)

        if "mask" in data and data["mask"] is not None:
            mask_vol = data["mask"]  # (D, H, W, C)
        else:
            mask_vol = np.zeros(image_vol.shape + (self.num_classes,), dtype=np.float32)

        # Return full volume as tensor
        # Image: (D, H, W) -> (1, D, H, W) - Trainer will handle 2.5D slicing
        img_tensor = torch.from_numpy(image_vol).float().unsqueeze(0)

        # Mask: (D, H, W, C) -> (C, D, H, W)
        mask_tensor = torch.from_numpy(mask_vol).float().permute(3, 0, 1, 2)
        mask_tensor = (mask_tensor > 0.5).float()

        return {
            "id": f"{row['case']}_{row['day']}",
            "image": img_tensor,
            "mask": mask_tensor,
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
    # Use SliceDataset for training (2.5D slices)
    train_ds = SliceDataset(train_df, subset="train")
    # Use VolumetricDataset for val/test (Full volumes for reconstruction)
    val_ds = VolumetricDataset(val_df, subset="val")
    test_ds = VolumetricDataset(test_df, subset="test")

    # 3. Create DataLoaders
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
