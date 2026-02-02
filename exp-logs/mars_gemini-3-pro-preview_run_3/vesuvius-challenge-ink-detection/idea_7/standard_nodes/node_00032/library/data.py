import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from library.config import Config


def load_image(path, grayscale=True):
    """
    Loads an image from a path.
    """
    path = str(path)
    if not os.path.exists(path):
        return None
    flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    return cv2.imread(path, flags)


def get_global_stats(metadata_df, cache_dir, load_cached_data=True):
    """
    Computes or loads global mean and std from the training fragments.
    """
    stats_path = cache_dir / "normalization_stats.npy"

    if load_cached_data and stats_path.exists():
        print(f"Loading global stats from {stats_path}")
        stats = np.load(stats_path, allow_pickle=True).item()
        return stats["mean"], stats["std"]

    print("Computing global stats from training data...")
    # We need to load volumes to compute stats.
    # To avoid loading everything into memory just for stats if not needed,
    # we process one by one.

    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    pixel_count = 0.0

    for _, row in metadata_df.iterrows():
        # Construct path to surface volume
        vol_dir = Config.INPUT_DIR / row["surface_volume_path"]

        # Load all slices
        slices = []
        for z in range(Config.Z_DIM):
            slice_path = vol_dir / f"{z:02d}.tif"
            if slice_path.exists():
                img = load_image(slice_path)
                if img is not None:
                    slices.append(img)
                else:
                    # Fallback for missing slice (should not happen based on EDA)
                    slices.append(
                        np.zeros((row["height"], row["width"]), dtype=np.uint8)
                    )
            else:
                slices.append(np.zeros((row["height"], row["width"]), dtype=np.uint8))

        volume = np.stack(slices, axis=0).astype(np.float32)

        pixel_sum += np.sum(volume)
        pixel_sq_sum += np.sum(volume**2)
        pixel_count += volume.size

    mean = pixel_sum / pixel_count
    std = np.sqrt((pixel_sq_sum / pixel_count) - (mean**2))

    # Save
    np.save(stats_path, {"mean": mean, "std": std})
    print(f"Global Stats Computed - Mean: {mean:.4f}, Std: {std:.4f}")

    return mean, std


def load_fragment(row, cache_dir, load_cached_data=True):
    """
    Loads a fragment volume, mask, and label (if available).
    Uses caching to speed up subsequent loads.
    """
    fragment_id = str(row["fragment_id"])

    vol_cache_path = cache_dir / f"{fragment_id}_volume.npy"
    mask_cache_path = cache_dir / f"{fragment_id}_mask.npy"
    label_cache_path = cache_dir / f"{fragment_id}_label.npy"

    # 1. Load Volume
    if load_cached_data and vol_cache_path.exists():
        volume = np.load(vol_cache_path)
    else:
        vol_dir = Config.INPUT_DIR / row["surface_volume_path"]
        slices = []
        for z in range(Config.Z_DIM):
            slice_path = vol_dir / f"{z:02d}.tif"
            if slice_path.exists():
                img = load_image(slice_path)
                if img is None:
                    img = np.zeros((row["height"], row["width"]), dtype=np.uint8)
            else:
                img = np.zeros((row["height"], row["width"]), dtype=np.uint8)
            slices.append(img)
        volume = np.stack(slices, axis=0).astype(np.float32)  # (Z, H, W)
        np.save(vol_cache_path, volume)

    # 2. Load Mask
    if load_cached_data and mask_cache_path.exists():
        mask = np.load(mask_cache_path)
    else:
        if pd.notna(row["mask_path"]):
            mask = load_image(Config.INPUT_DIR / row["mask_path"])
            mask = (mask > 0).astype(np.float32)
        else:
            mask = np.ones((volume.shape[1], volume.shape[2]), dtype=np.float32)
        np.save(mask_cache_path, mask)

    # 3. Load Label (only for train/val)
    label = None
    if pd.notna(row["inklabels_path"]):
        if load_cached_data and label_cache_path.exists():
            label = np.load(label_cache_path)
        else:
            label = load_image(Config.INPUT_DIR / row["inklabels_path"])
            label = (label > 0).astype(np.float32)
            np.save(label_cache_path, label)

    return volume, mask, label


class InkDataset(Dataset):
    def __init__(
        self,
        metadata_df,
        split,
        mean,
        std,
        transform=None,
        num_samples=4000,
        cache_dir=Config.WORKING_DIR,
        load_cached_data=True,
    ):
        self.split = split
        self.transform = transform
        self.mean = mean
        self.std = std
        self.num_samples = num_samples
        self.patch_size = Config.PATCH_SIZE

        self.fragments = []
        self.valid_indices = []  # List of (frag_idx, y, x)

        print(f"Initializing InkDataset ({split})...")

        for idx, row in metadata_df.iterrows():
            vol, mask, label = load_fragment(row, cache_dir, load_cached_data)

            # Normalize volume immediately to save compute during training
            # Note: We keep it in float32.
            # If RAM is an issue, we could normalize on the fly, but we have 220GB.
            vol = (vol - self.mean) / (self.std + 1e-6)

            self.fragments.append({"volume": vol, "mask": mask, "label": label})

            # Generate sampling indices
            h, w = mask.shape

            if split == "train":
                # Find valid pixels (where mask is 1)
                # We erode the mask slightly to avoid border artifacts if necessary,
                # but for now we just ensure patch fits.
                # We store top-left corners.

                # Valid area for top-left corner: mask[y:y+patch, x:x+patch] must be valid.
                # To speed up, we just check if the center is valid or simply use the provided mask.
                # A simple heuristic: valid pixels in the mask are candidates for patch centers.
                # We adjust to top-left.

                ys, xs = np.where(mask > 0)

                # Filter indices to ensure patch is within bounds
                # y must be <= h - patch_size
                # x must be <= w - patch_size
                valid_mask = (ys <= h - self.patch_size) & (xs <= w - self.patch_size)
                ys = ys[valid_mask]
                xs = xs[valid_mask]

                # To reduce memory for indices, we can subsample or just store them.
                # Storing millions of indices is fine in 220GB RAM.
                # We zip them into a list of tuples (frag_idx, y, x)
                frag_indices = np.column_stack([np.full_like(ys, idx), ys, xs])
                self.valid_indices.append(frag_indices)

            elif split == "val":
                # Deterministic Grid
                stride = Config.STRIDE
                y_range = range(0, h - self.patch_size + 1, stride)
                x_range = range(0, w - self.patch_size + 1, stride)

                # If the image is not perfectly divisible, we might miss the edge.
                # We can add the last possible position.
                ys = list(y_range)
                if ys[-1] != h - self.patch_size:
                    ys.append(h - self.patch_size)

                xs = list(x_range)
                if xs[-1] != w - self.patch_size:
                    xs.append(w - self.patch_size)

                grid_indices = []
                for y in ys:
                    for x in xs:
                        grid_indices.append([idx, y, x])

                self.valid_indices.append(np.array(grid_indices))

        # Concatenate all indices
        if self.valid_indices:
            self.valid_indices = np.concatenate(self.valid_indices, axis=0)
        else:
            self.valid_indices = np.array([])

        print(f"Dataset ({split}) loaded. Valid patches: {len(self.valid_indices)}")

    def __len__(self):
        if self.split == "train":
            return self.num_samples
        else:
            return len(self.valid_indices)

    def __getitem__(self, idx):
        if self.split == "train":
            # Random sampling
            idx = np.random.randint(0, len(self.valid_indices))

        frag_idx, y, x = self.valid_indices[idx]
        data = self.fragments[frag_idx]

        # Crop
        volume_patch = data["volume"][
            :, y : y + self.patch_size, x : x + self.patch_size
        ]

        if data["label"] is not None:
            label_patch = data["label"][
                y : y + self.patch_size, x : x + self.patch_size
            ]
        else:
            # Dummy label for test/inference if needed
            label_patch = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)

        # Apply Transforms (Geometric)
        if self.split == "train" and self.transform:
            volume_patch, label_patch = self.transform(volume_patch, label_patch)

        # Convert to Tensor
        # Volume: (Z, H, W) -> Tensor
        # Label: (H, W) -> (1, H, W) Tensor

        volume_tensor = torch.from_numpy(np.ascontiguousarray(volume_patch)).float()
        label_tensor = (
            torch.from_numpy(np.ascontiguousarray(label_patch)).unsqueeze(0).float()
        )

        return volume_tensor, label_tensor


def geometric_transforms(volume, label):
    """
    Applies random flips and rotations.
    volume: (Z, H, W)
    label: (H, W)
    """
    # Random Horizontal Flip
    if np.random.rand() < 0.5:
        volume = np.flip(volume, axis=2)
        label = np.flip(label, axis=1)

    # Random Vertical Flip
    if np.random.rand() < 0.5:
        volume = np.flip(volume, axis=1)
        label = np.flip(label, axis=0)

    # Random Rotation (0, 90, 180, 270)
    k = np.random.randint(0, 4)
    if k > 0:
        volume = np.rot90(volume, k=k, axes=(1, 2))
        label = np.rot90(label, k=k, axes=(0, 1))

    return volume, label


def get_dataloaders(
    train_metadata_path=Config.TRAIN_METADATA,
    val_metadata_path=Config.VAL_METADATA,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    num_train_samples=4000,
):
    """
    Creates DataLoaders for training and validation.
    """
    # Ensure cache directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Load Metadata
    df_train = pd.read_csv(train_metadata_path)
    df_val = pd.read_csv(val_metadata_path)

    # Compute Global Stats (from training data)
    mean, std = get_global_stats(df_train, Config.WORKING_DIR, load_cached_data)

    # Create Datasets
    train_dataset = InkDataset(
        metadata_df=df_train,
        split="train",
        mean=mean,
        std=std,
        transform=geometric_transforms,
        num_samples=num_train_samples,
        cache_dir=Config.WORKING_DIR,
        load_cached_data=load_cached_data,
    )

    val_dataset = InkDataset(
        metadata_df=df_val,
        split="val",
        mean=mean,
        std=std,
        transform=None,
        cache_dir=Config.WORKING_DIR,
        load_cached_data=load_cached_data,
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle the random indices
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
