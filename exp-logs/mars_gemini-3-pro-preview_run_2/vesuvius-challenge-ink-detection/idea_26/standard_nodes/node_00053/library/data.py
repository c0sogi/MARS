import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config

# -----------------------------------------------------------------------------
# Volume Caching & Management
# -----------------------------------------------------------------------------


def load_volume_slice(volume_dir, z_index):
    """
    Loads a single slice from the volume directory.
    """
    filename = f"{z_index:02d}.tif"
    path = os.path.join(volume_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Slice {filename} not found in {volume_dir}")

    # Load as grayscale uint16
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img


def get_cached_volume(fragment_id, volume_path, z_min, z_max, load_cached_data=True):
    """
    Retrieves a sub-volume of the fragment, using caching.
    Range: [z_min, z_max)
    """
    cache_filename = f"frag_{fragment_id}_slab_{z_min}_{z_max}.npy"
    cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            volume = np.load(cache_path)
            return volume
        except Exception as e:
            print(f"Warning: Failed to load cache {cache_path}: {e}. Recomputing.")

    # 2. Compute from scratch
    full_volume_path = os.path.join(Config.INPUT_DIR, volume_path)

    slices = []
    for z in range(z_min, z_max):
        img = load_volume_slice(full_volume_path, z)
        slices.append(img)

    volume = np.stack(slices, axis=0)  # (D, H, W)

    # 3. Save to cache
    try:
        np.save(cache_path, volume)
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

    return volume


def prepare_volumes(fragment_ids, metadata_df, load_cached_data=True):
    """
    Pre-loads/caches the necessary volume slices for the given fragments.
    Caches slices 16 to 36 to cover both training range (16-24 + 12 depth)
    and inference range (16, 20, 24 + 12 depth).
    """
    volumes = {}
    z_min = 16
    z_max = 36  # Covers up to start index 24 with depth 12 (indices 24..35)

    for fid in fragment_ids:
        # Get path from metadata
        frag_rows = metadata_df[metadata_df["fragment_id"] == fid]
        if len(frag_rows) == 0:
            continue
        vol_path = frag_rows.iloc[0]["volume_path"]

        vol = get_cached_volume(
            fid, vol_path, z_min, z_max, load_cached_data=load_cached_data
        )
        volumes[fid] = vol

    return volumes


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class InkDataset(Dataset):
    def __init__(
        self, metadata_df, volumes_cache, mode="train", z_start=None, transform=None
    ):
        """
        Args:
            metadata_df (pd.DataFrame): Dataframe containing patch metadata.
            volumes_cache (dict): Dictionary mapping fragment_id to 3D numpy array.
            mode (str): 'train', 'val', 'test'.
            z_start (int, optional): Fixed Z-start index for inference/validation.
            transform (A.Compose, optional): Albumentations transforms.
        """
        self.metadata = metadata_df.reset_index(drop=True)
        self.volumes = volumes_cache
        self.mode = mode
        self.z_start = z_start
        self.transform = transform

        # The cached volume starts at global index 16
        self.global_z_offset = 16

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        frag_id = row["fragment_id"]
        x, y = row["x"], row["y"]
        w, h = row["width"], row["height"]

        # 1. Determine Z-start
        if self.mode == "train":
            # Dynamic Safe-View Sampling: Randomly sample start index
            # TRAIN_Z_RANGE is (16, 24). We want to include 24.
            z_global = np.random.randint(
                Config.TRAIN_Z_RANGE[0], Config.TRAIN_Z_RANGE[1] + 1
            )
        else:
            # Deterministic Z-start
            if self.z_start is not None:
                z_global = self.z_start
            else:
                # Default for validation: Middle of range
                z_global = (Config.TRAIN_Z_RANGE[0] + Config.TRAIN_Z_RANGE[1]) // 2

        # 2. Extract Slab
        z_local = z_global - self.global_z_offset
        vol = self.volumes[frag_id]  # (D, H_frag, W_frag)

        # Safety check
        if z_local < 0 or (z_local + Config.SLAB_DEPTH) > vol.shape[0]:
            # Fallback to safe limits if out of bounds (should not happen with correct cache)
            z_local = np.clip(z_local, 0, vol.shape[0] - Config.SLAB_DEPTH)

        # Crop 3D slab
        slab_crop = vol[z_local : z_local + Config.SLAB_DEPTH, y : y + h, x : x + w]

        # Pad if crop is smaller than tile size (edges)
        pad_h = h - slab_crop.shape[1]
        pad_w = w - slab_crop.shape[2]
        if pad_h > 0 or pad_w > 0:
            slab_crop = np.pad(
                slab_crop,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=0,
            )

        # 3. Projection (MIP -> 3 Channels)
        # Split 12 slices into 3 chunks of 4
        ch1 = np.max(slab_crop[0:4], axis=0)
        ch2 = np.max(slab_crop[4:8], axis=0)
        ch3 = np.max(slab_crop[8:12], axis=0)

        image = np.stack([ch1, ch2, ch3], axis=-1)  # (H, W, 3)

        # 4. Normalization (uint16 -> float32 [0, 1])
        image = image.astype(np.float32) / 65535.0

        # 5. Load Label
        mask = None
        if "label_path" in row and pd.notna(row["label_path"]):
            label_path = os.path.join(Config.INPUT_DIR, row["label_path"])
            if os.path.exists(label_path):
                label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                label_crop = label_img[y : y + h, x : x + w]
                if pad_h > 0 or pad_w > 0:
                    label_crop = np.pad(
                        label_crop,
                        ((0, pad_h), (0, pad_w)),
                        mode="constant",
                        constant_values=0,
                    )
                mask = (label_crop > 0).astype(np.float32)
            else:
                mask = np.zeros((h, w), dtype=np.float32)

        # 6. Transforms
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1))
            if mask is not None:
                mask = torch.from_numpy(mask)

        # Ensure mask is (1, H, W)
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            return image, mask
        else:
            # For test set without labels, return dummy mask
            dummy_mask = torch.zeros((1, h, w), dtype=torch.float32)
            return image, dummy_mask


# -----------------------------------------------------------------------------
# Loader Factories
# -----------------------------------------------------------------------------


def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


def get_loaders(load_cached_data=True):
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Identify all unique fragments
    all_frags = np.unique(
        np.concatenate(
            [train_df["fragment_id"].unique(), val_df["fragment_id"].unique()]
        )
    )

    # Prepare cache
    combined_df = pd.concat([train_df, val_df])
    volumes_cache = prepare_volumes(
        all_frags, combined_df, load_cached_data=load_cached_data
    )

    # Datasets
    train_ds = InkDataset(
        train_df, volumes_cache, mode="train", transform=get_transforms("train")
    )

    val_ds = InkDataset(
        val_df, volumes_cache, mode="val", transform=get_transforms("val")
    )

    # DataLoaders
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

    return train_loader, val_loader


def get_test_loader(z_start, load_cached_data=True):
    """
    Creates a DataLoader for the test set with on-the-fly tiling.
    """
    test_meta_df = pd.read_csv(Config.TEST_METADATA_PATH)

    patches = []
    # Cache volumes for test fragments (slices 16-36)
    volumes_cache = prepare_volumes(
        test_meta_df["fragment_id"].unique(),
        test_meta_df,
        load_cached_data=load_cached_data,
    )

    for _, row in test_meta_df.iterrows():
        fid = row["fragment_id"]
        vol_path = row["volume_path"]
        mask_path = row["mask_path"]

        # Load mask to get dimensions
        full_mask_path = os.path.join(Config.INPUT_DIR, mask_path)
        mask_img = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)
        h, w = mask_img.shape

        # Generate grid patches
        for y in range(0, h, Config.STRIDE):
            for x in range(0, w, Config.STRIDE):
                patches.append(
                    {
                        "fragment_id": fid,
                        "x": x,
                        "y": y,
                        "width": Config.TILE_SIZE,
                        "height": Config.TILE_SIZE,
                        "volume_path": vol_path,
                    }
                )

    patch_df = pd.DataFrame(patches)

    dataset = InkDataset(
        patch_df,
        volumes_cache,
        mode="test",
        z_start=z_start,
        transform=get_transforms("test"),
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return loader
