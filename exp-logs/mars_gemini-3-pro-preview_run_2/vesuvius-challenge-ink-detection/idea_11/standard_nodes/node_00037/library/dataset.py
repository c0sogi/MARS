import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import glob
from library.config import Config
from library.utils import load_volume_slice, load_mask, normalize_image, seed_everything

# Constants for Volume Caching
# We load a safe range of Z-slices to cover training (start=20) and inference scanning (18-22)
# Min required: 18 (Scan A start)
# Max required: 22 (Scan C start) + 24 (Channel 3 end) = 46
# We add a buffer to be safe: 15 to 55
CACHE_Z_MIN = 15
CACHE_Z_MAX = 55


def prepare_volumes(fragment_ids, load_cached_data=True):
    """
    Loads relevant Z-slices for the specified fragments.
    Implements strict caching logic using .npy files.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    volumes = {}

    for frag_id in fragment_ids:
        frag_id = str(frag_id)
        cache_path = os.path.join(
            Config.WORKING_DIR, f"frag_{frag_id}_slab_{CACHE_Z_MIN}_{CACHE_Z_MAX}.npy"
        )

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                # Use mmap_mode='r' to avoid loading everything into RAM at once if memory is tight,
                # though 220GB RAM is sufficient to load fully. Loading fully is faster for training.
                volumes[frag_id] = np.load(cache_path)
                # print(f"Loaded cached volume for fragment {frag_id} from {cache_path}")
                continue
            except Exception as e:
                print(f"Failed to load cache for {frag_id}: {e}. Recomputing...")

        # 2. Compute from scratch
        # Determine path
        # Check if it's train or test to find the directory
        # We assume standard directory structure input/[train|test]/[frag_id]
        if os.path.exists(os.path.join(Config.INPUT_DIR, "train", frag_id)):
            base_dir = os.path.join(Config.INPUT_DIR, "train", frag_id)
        elif os.path.exists(os.path.join(Config.INPUT_DIR, "test", frag_id)):
            base_dir = os.path.join(Config.INPUT_DIR, "test", frag_id)
        else:
            raise FileNotFoundError(
                f"Fragment {frag_id} not found in train or test directories."
            )

        volume_dir = os.path.join(base_dir, "surface_volume")

        slices = []
        for z in range(CACHE_Z_MIN, CACHE_Z_MAX):
            img = load_volume_slice(volume_dir, z)
            if img is None:
                # If slice is missing (e.g. out of bounds), create black slice
                # Get shape from a valid slice or mask
                mask_path = os.path.join(base_dir, "mask.png")
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                img = np.zeros_like(mask, dtype=np.uint16)
            slices.append(img)

        # Stack into (Depth, Height, Width)
        volume_stack = np.stack(slices, axis=0)

        # Save to cache
        np.save(cache_path, volume_stack)
        volumes[frag_id] = volume_stack
        # print(f"Cached volume for fragment {frag_id} to {cache_path}")

    return volumes


class InkDataset(Dataset):
    def __init__(self, df, volumes, z_start=Config.Z_START, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing patch info.
            volumes (dict): Dictionary mapping fragment_id to 3D numpy arrays (D, H, W).
            z_start (int): The starting Z-index for the slab construction.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.volumes = volumes
        self.z_start = z_start
        self.mode = mode

        # Calculate indices relative to the cached volume
        # The cached volume starts at CACHE_Z_MIN
        # We need to map global Z coordinates to local cache coordinates
        self.z_offset_in_cache = self.z_start - CACHE_Z_MIN

        if self.z_offset_in_cache < 0:
            raise ValueError(
                f"Requested z_start {z_start} is below cached range min {CACHE_Z_MIN}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frag_id = str(row["fragment_id"])
        x, y = row["x"], row["y"]
        w, h = row["width"], row["height"]

        # 1. Get Volume Patch
        # Shape: (D_cached, H_frag, W_frag) -> slice to patch
        # Note: self.volumes[frag_id] might be huge, slicing it is efficient
        vol_full = self.volumes[frag_id]

        # Handle boundary conditions for patch
        # If the patch goes out of bounds (which shouldn't happen with correct metadata), pad or clip
        # Metadata generation ensures validity, but let's be safe
        vol_h, vol_w = vol_full.shape[1], vol_full.shape[2]

        y_end = min(y + h, vol_h)
        x_end = min(x + w, vol_w)

        # Extract the full depth stack for this spatial patch
        patch_stack = vol_full[:, y:y_end, x:x_end]

        # Pad if smaller than tile size (e.g. edge patches)
        pad_h = h - (y_end - y)
        pad_w = w - (x_end - x)
        if pad_h > 0 or pad_w > 0:
            patch_stack = np.pad(
                patch_stack, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant"
            )

        # 2. Construct Channels (MIPs)
        # We need 3 channels based on overlapping slabs
        # Channel 1: [z, z+12]
        # Channel 2: [z+6, z+18]
        # Channel 3: [z+12, z+24]
        # All indices are relative to z_start

        # Map z_start to local index in patch_stack
        z0 = self.z_offset_in_cache

        # Define slice ranges
        # Python slice end is exclusive
        r1 = (z0, z0 + Config.Z_DIM)
        r2 = (z0 + Config.Z_STEP, z0 + Config.Z_STEP + Config.Z_DIM)
        r3 = (z0 + 2 * Config.Z_STEP, z0 + 2 * Config.Z_STEP + Config.Z_DIM)

        # Check bounds
        max_z = patch_stack.shape[0]
        if r3[1] > max_z:
            # This should be caught by CACHE_Z_MAX setting, but as fallback:
            # print(f"Warning: Z-range out of bounds for frag {frag_id}. Padding.")
            pad_z = r3[1] - max_z
            patch_stack = np.pad(
                patch_stack, ((0, pad_z), (0, 0), (0, 0)), mode="constant"
            )

        # Compute MIPs
        # patch_stack is (D, H, W)
        ch1 = np.max(patch_stack[r1[0] : r1[1]], axis=0)
        ch2 = np.max(patch_stack[r2[0] : r2[1]], axis=0)
        ch3 = np.max(patch_stack[r3[0] : r3[1]], axis=0)

        # Stack to (H, W, 3) for easy transform
        image = np.stack([ch1, ch2, ch3], axis=2)

        # 3. Normalize
        image = normalize_image(image)  # Returns float32 [0, 1]

        # 4. Load Label (if available)
        mask = None
        label = None

        if self.mode in ["train", "val"]:
            label_path = os.path.join(Config.INPUT_DIR, row["label_path"])
            label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            if label_img is not None:
                label_patch = label_img[y:y_end, x:x_end]
                if pad_h > 0 or pad_w > 0:
                    label_patch = np.pad(
                        label_patch, ((0, pad_h), (0, pad_w)), mode="constant"
                    )
                label = (label_patch > 0).astype(np.float32)
            else:
                # Fallback
                label = np.zeros((h, w), dtype=np.float32)

        # Load valid mask (for loss masking if needed, though usually implicit in data generation)
        # We return it just in case
        if "mask_path" in row:
            mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
            mask_img = load_mask(mask_path)
            if mask_img is not None:
                mask_patch = mask_img[y:y_end, x:x_end]
                if pad_h > 0 or pad_w > 0:
                    mask_patch = np.pad(
                        mask_patch, ((0, pad_h), (0, pad_w)), mode="constant"
                    )
                mask = mask_patch.astype(np.float32)
            else:
                mask = np.zeros((h, w), dtype=np.float32)

        # 5. Augmentations (Train Only)
        if self.mode == "train":
            # Random Flip
            if np.random.rand() < 0.5:
                image = np.flip(image, axis=0)  # Vertical
                if label is not None:
                    label = np.flip(label, axis=0)
                if mask is not None:
                    mask = np.flip(mask, axis=0)

            if np.random.rand() < 0.5:
                image = np.flip(image, axis=1)  # Horizontal
                if label is not None:
                    label = np.flip(label, axis=1)
                if mask is not None:
                    mask = np.flip(mask, axis=1)

            # Random Rotate (k * 90 degrees)
            k = np.random.randint(0, 4)
            if k > 0:
                image = np.rot90(image, k, axes=(0, 1))
                if label is not None:
                    label = np.rot90(label, k, axes=(0, 1))
                if mask is not None:
                    mask = np.rot90(mask, k, axes=(0, 1))

        # 6. To Tensor
        # Image: (H, W, 3) -> (3, H, W)
        image = torch.from_numpy(image.copy()).permute(2, 0, 1).float()

        if label is not None:
            label = torch.from_numpy(label.copy()).unsqueeze(0).float()  # (1, H, W)
        else:
            label = torch.zeros((1, h, w), dtype=torch.float32)

        if mask is not None:
            mask = torch.from_numpy(mask.copy()).unsqueeze(0).float()
        else:
            mask = torch.zeros((1, h, w), dtype=torch.float32)

        return image, label, mask, idx


def get_datasets(load_cached_data=True):
    """
    Factory function to create training and validation datasets.
    """
    train_csv_path = os.path.join(Config.METADATA_DIR, "train.csv")
    val_csv_path = os.path.join(Config.METADATA_DIR, "validation.csv")

    df_train = pd.read_csv(train_csv_path)
    df_val = pd.read_csv(val_csv_path)

    # Identify all unique fragments needed
    all_frags = set(df_train["fragment_id"].unique()) | set(
        df_val["fragment_id"].unique()
    )

    # Prepare volumes
    volumes = prepare_volumes(list(all_frags), load_cached_data=load_cached_data)

    # Create Datasets
    train_ds = InkDataset(df_train, volumes, z_start=Config.Z_START, mode="train")
    val_ds = InkDataset(df_val, volumes, z_start=Config.Z_START, mode="val")

    return train_ds, val_ds


def get_test_dataset(fragment_id, load_cached_data=True):
    """
    Creates a dataset for a specific test fragment by tiling it.
    Used for inference.
    """
    # 1. Load Fragment Metadata
    # We assume standard path
    base_dir = os.path.join(Config.INPUT_DIR, "test", fragment_id)
    mask_path = os.path.join(base_dir, "mask.png")

    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found for test fragment {fragment_id}")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    h, w = mask.shape

    # 2. Generate Tiling Metadata
    patches = []
    # Use overlap for inference to avoid edge artifacts?
    # Config.STRIDE is 512 (non-overlapping).
    # For inference, overlapping is often better, but let's stick to Config or simple tiling first.
    # We will use the same TILE_SIZE.

    # To ensure full coverage, we might need a smaller stride or handle edges.
    # Here we use simple tiling with handling for the last chunk in __getitem__ padding.

    for y in range(0, h, Config.TILE_SIZE):
        for x in range(0, w, Config.TILE_SIZE):
            patches.append(
                {
                    "fragment_id": fragment_id,
                    "x": x,
                    "y": y,
                    "width": Config.TILE_SIZE,
                    "height": Config.TILE_SIZE,
                    "mask_path": os.path.relpath(mask_path, Config.INPUT_DIR),
                }
            )

    df_test = pd.DataFrame(patches)

    # 3. Prepare Volume
    volumes = prepare_volumes([fragment_id], load_cached_data=load_cached_data)

    # 4. Create Dataset
    # Note: z_start will be set dynamically by the inference loop,
    # but we initialize with default.
    test_ds = InkDataset(df_test, volumes, z_start=Config.Z_START, mode="test")

    return test_ds, (h, w)
