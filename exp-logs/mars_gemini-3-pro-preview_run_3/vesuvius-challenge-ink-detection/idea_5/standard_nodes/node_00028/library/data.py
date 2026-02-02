import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything


def load_fragment_data(metadata_row, split, load_cached_data=True):
    """
    Loads volume, mask, and label (if available) for a specific fragment.
    Implements caching using .npy files to speed up subsequent loads.
    """
    fragment_id = str(metadata_row["fragment_id"])

    # Define cache paths
    cache_vol_path = os.path.join(Config.CACHE_DIR, f"{fragment_id}_volume.npy")
    cache_mask_path = os.path.join(Config.CACHE_DIR, f"{fragment_id}_mask.npy")
    cache_label_path = os.path.join(Config.CACHE_DIR, f"{fragment_id}_label.npy")

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 1. Try loading from cache
    if (
        load_cached_data
        and os.path.exists(cache_vol_path)
        and os.path.exists(cache_mask_path)
    ):
        try:
            volume = np.load(cache_vol_path)
            mask = np.load(cache_mask_path)

            label = None
            if split in ["train", "val"]:
                if os.path.exists(cache_label_path):
                    label = np.load(cache_label_path)

            # If successful, return cached data
            if split == "test" or label is not None:
                return volume, mask, label
        except Exception as e:
            print(
                f"Failed to load cache for {fragment_id}: {e}. Reloading from source."
            )

    # 2. Load from source if cache missing or failed
    # Load Volume
    vol_dir = os.path.join(Config.INPUT_DIR, metadata_row["surface_volume_path"])
    slices = []
    # We expect 65 slices (00.tif to 64.tif)
    for i in range(Config.Z_DIM):
        slice_path = os.path.join(vol_dir, f"{i:02d}.tif")
        if not os.path.exists(slice_path):
            # Fallback or error handling if specific slice missing
            # Assuming data integrity based on description
            raise FileNotFoundError(f"Slice {slice_path} not found.")

        img = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)
        slices.append(img)

    volume = np.stack(slices, axis=0)  # (65, H, W)

    # Normalize Volume (Instance Normalization)
    # Using float32 for processing
    volume = volume.astype(np.float32)
    mean = volume.mean()
    std = volume.std()
    volume = (volume - mean) / (std + 1e-6)

    # Load Mask
    mask_path = os.path.join(Config.INPUT_DIR, metadata_row["mask_path"])
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 0).astype(np.uint8)  # Binary 0/1

    # Load Label (if train/val)
    label = None
    if split in ["train", "val"] and pd.notna(metadata_row.get("inklabels_path")):
        label_path = os.path.join(Config.INPUT_DIR, metadata_row["inklabels_path"])
        if os.path.exists(label_path):
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            label = (label > 0).astype(np.uint8)  # Binary 0/1

    # 3. Save to cache
    np.save(cache_vol_path, volume)
    np.save(cache_mask_path, mask)
    if label is not None:
        np.save(cache_label_path, label)

    return volume, mask, label


class InkTrainDataset(Dataset):
    """
    Dataset for training.
    Loads all fragments into memory and performs random cropping and augmentation.
    """

    def __init__(self, df_metadata, epoch_size=2000, load_cached_data=True):
        self.epoch_size = epoch_size
        self.fragments = []

        print(f"Loading Training Data (Count: {len(df_metadata)})...")
        for _, row in df_metadata.iterrows():
            vol, mask, label = load_fragment_data(row, "train", load_cached_data)
            self.fragments.append(
                {
                    "volume": torch.from_numpy(vol),
                    "mask": torch.from_numpy(mask),
                    "label": torch.from_numpy(label),
                }
            )

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        # Select random fragment
        frag = self.fragments[idx % len(self.fragments)]
        vol = frag["volume"]
        mask = frag["mask"]
        label = frag["label"]

        _, h, w = vol.shape
        patch_size = Config.PATCH_SIZE

        # Random Crop
        # Try to find a valid crop (inside mask) for a few attempts
        for _ in range(10):
            y = np.random.randint(0, h - patch_size)
            x = np.random.randint(0, w - patch_size)

            # Check if center of patch is inside the mask
            # This ensures we don't train purely on background padding
            if mask[y + patch_size // 2, x + patch_size // 2] > 0:
                break

        # Crop
        vol_crop = vol[:, y : y + patch_size, x : x + patch_size]
        label_crop = label[y : y + patch_size, x : x + patch_size]

        # Augmentations
        # Random Flip
        if np.random.rand() > 0.5:
            vol_crop = torch.flip(vol_crop, [2])  # Flip W
            label_crop = torch.flip(label_crop, [1])

        if np.random.rand() > 0.5:
            vol_crop = torch.flip(vol_crop, [1])  # Flip H
            label_crop = torch.flip(label_crop, [0])

        # Random Rotate 90
        k = np.random.randint(0, 4)
        if k > 0:
            vol_crop = torch.rot90(vol_crop, k, [1, 2])
            label_crop = torch.rot90(label_crop, k, [0, 1])

        return vol_crop, label_crop.float().unsqueeze(0)


class InkInferenceDataset(Dataset):
    """
    Dataset for Validation and Inference.
    Generates a deterministic grid of tiles covering the fragment.
    """

    def __init__(self, df_metadata, split, load_cached_data=True):
        self.split = split
        self.tiles = []
        self.data_map = {}  # Store loaded volumes by fragment_id

        print(f"Loading {split.capitalize()} Data (Count: {len(df_metadata)})...")

        for _, row in df_metadata.iterrows():
            frag_id = str(row["fragment_id"])
            vol, mask, label = load_fragment_data(row, split, load_cached_data)

            # Convert to tensor
            vol_tensor = torch.from_numpy(vol)

            # Store in map
            self.data_map[frag_id] = {
                "volume": vol_tensor,
                "mask": mask,  # Keep as numpy for shape checking if needed
                "label": label,
                "orig_h": mask.shape[0],
                "orig_w": mask.shape[1],
            }

            # Generate Grid
            h, w = mask.shape
            stride = Config.INFERENCE_STRIDE
            size = Config.PATCH_SIZE

            y_steps = list(range(0, h - size + 1, stride))
            # Ensure coverage of the last edge
            if (h - size) % stride != 0:
                y_steps.append(h - size)

            x_steps = list(range(0, w - size + 1, stride))
            if (w - size) % stride != 0:
                x_steps.append(w - size)

            for y in y_steps:
                for x in x_steps:
                    self.tiles.append({"fragment_id": frag_id, "y": y, "x": x})

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = self.tiles[idx]
        frag_id = tile["fragment_id"]
        y = tile["y"]
        x = tile["x"]
        size = Config.PATCH_SIZE

        data = self.data_map[frag_id]
        vol = data["volume"]

        # Crop
        vol_crop = vol[:, y : y + size, x : x + size]

        # Prepare metadata for reconstruction
        meta = {"fragment_id": frag_id, "y": y, "x": x, "h": size, "w": size}

        # If validation, return label as well
        if self.split == "val" and data["label"] is not None:
            label = data["label"]
            label_crop = label[y : y + size, x : x + size]
            return vol_crop, torch.from_numpy(label_crop).float().unsqueeze(0), meta

        return vol_crop, meta


def get_loaders(load_cached_data=True):
    """
    Creates DataLoaders for training, validation, and testing.
    """
    seed_everything(Config.SEED)

    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Initialize Datasets
    # Training dataset: Random sampling
    # Cite solution_lesson_node_00020: Complexity Requires Commensurate Data Throughput
    # Increasing samples per epoch to ~12,000 (32 * 375 = 12,000)
    train_ds = InkTrainDataset(
        df_train,
        epoch_size=Config.BATCH_SIZE * 375,
        load_cached_data=load_cached_data,
    )

    # Validation dataset: Deterministic grid
    val_ds = InkInferenceDataset(df_val, split="val", load_cached_data=load_cached_data)

    # Test dataset: Deterministic grid
    test_ds = InkInferenceDataset(
        df_test, split="test", load_cached_data=load_cached_data
    )

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
