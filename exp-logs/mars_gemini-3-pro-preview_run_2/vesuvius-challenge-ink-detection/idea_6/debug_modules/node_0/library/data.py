import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


def get_transforms(mode="train"):
    """
    Returns the Albumentations transformations for the given mode.
    Strictly geometric augmentations for training; no intensity changes.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # No normalization here as we handle 16-bit to float conversion manually
                ToTensorV2(transpose_mask=True),
            ],
            additional_targets={"image": "image", "mask": "mask", "valid_mask": "mask"},
        )
    else:
        return A.Compose(
            [
                ToTensorV2(transpose_mask=True),
            ],
            additional_targets={"image": "image", "mask": "mask", "valid_mask": "mask"},
        )


def load_fragment_mips(fragment_id, volume_dir, load_cached_data=True):
    """
    Loads or computes the 6-channel MIP tensor for a given fragment.
    Implements caching to ./working/idea_6/.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"{fragment_id}_mips.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            mips = np.load(cache_path)
            # Verify shape matches expectation (Channels, H, W)
            if mips.shape[0] == Config.IN_CHANNELS:
                return mips
        except Exception as e:
            print(f"Failed to load cache for {fragment_id}: {e}. Recomputing...")

    # 2. Compute from scratch
    # Determine dimensions from the first slice
    first_slice_path = os.path.join(
        Config.INPUT_DIR, volume_dir, f"{Config.Z_START:02d}.tif"
    )
    if not os.path.exists(first_slice_path):
        raise FileNotFoundError(f"Volume slice not found: {first_slice_path}")

    img_0 = cv2.imread(first_slice_path, cv2.IMREAD_UNCHANGED)
    h, w = img_0.shape

    # Initialize storage for slabs
    # We will compute 6 slabs. Each slab is a MIP of 4 slices.
    slab_mips = []

    for slab_idx in range(Config.SLAB_COUNT):
        start_z = Config.Z_START + (slab_idx * Config.SLAB_DEPTH)
        end_z = start_z + Config.SLAB_DEPTH

        slab_slices = []
        for z in range(start_z, end_z):
            path = os.path.join(Config.INPUT_DIR, volume_dir, f"{z:02d}.tif")
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                slab_slices.append(img)
            else:
                # Fallback for missing slices (should not happen in valid range)
                slab_slices.append(np.zeros((h, w), dtype=np.uint16))

        # Stack slices for this slab: (Depth, H, W)
        slab_stack = np.stack(slab_slices, axis=0)
        # Compute MIP: (H, W)
        mip = np.max(slab_stack, axis=0)
        slab_mips.append(mip)

    # Stack all slabs: (6, H, W)
    full_mips = np.stack(slab_mips, axis=0)

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.save(cache_path, full_mips)

    return full_mips


class InkDataset(Dataset):
    def __init__(self, dataframe, mode="train", transforms=None, load_cached_data=True):
        self.df = dataframe.reset_index(drop=True)
        self.mode = mode
        self.transforms = transforms

        # Pre-load all fragment data into memory
        self.fragment_data = {}
        unique_fragments = self.df["fragment_id"].unique()

        for fid in unique_fragments:
            # Find volume path for this fragment
            # We look at the first occurrence in the dataframe
            row = self.df[self.df["fragment_id"] == fid].iloc[0]
            volume_path = row["volume_path"]

            # Load MIPs
            mips = load_fragment_mips(
                fid, volume_path, load_cached_data=load_cached_data
            )

            # Load Masks/Labels if available (for boundary checks and training)
            mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            label_img = None
            if mode != "test" and "label_path" in row and pd.notna(row["label_path"]):
                label_p = os.path.join(Config.INPUT_DIR, row["label_path"])
                if os.path.exists(label_p):
                    label_img = cv2.imread(label_p, cv2.IMREAD_GRAYSCALE)

            self.fragment_data[fid] = {
                "mips": mips,  # (C, H, W)
                "mask": mask_img,  # (H, W)
                "label": label_img,  # (H, W) or None
            }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fid = row["fragment_id"]
        x, y = row["x"], row["y"]
        w, h = row["width"], row["height"]

        data = self.fragment_data[fid]
        full_mips = data["mips"]  # (C, FullH, FullW)
        full_mask = data["mask"]
        full_label = data["label"]

        # Crop
        # Handle boundary conditions if crop goes outside
        # The metadata generation script usually ensures validity, but we clip to be safe
        fh, fw = full_mips.shape[1], full_mips.shape[2]
        y_end = min(y + h, fh)
        x_end = min(x + w, fw)

        # Crop volume: (C, H, W) -> Transpose to (H, W, C) for Albumentations
        crop_mips = full_mips[:, y:y_end, x:x_end].transpose(1, 2, 0)

        # Pad if necessary (if at right/bottom edge and smaller than TILE_SIZE)
        pad_h = h - (y_end - y)
        pad_w = w - (x_end - x)

        if pad_h > 0 or pad_w > 0:
            crop_mips = np.pad(
                crop_mips,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        # Prepare dict for transforms
        transform_data = {"image": crop_mips}

        # Handle Labels/Masks
        if self.mode != "test":
            crop_label = full_label[y:y_end, x:x_end]
            crop_valid_mask = full_mask[y:y_end, x:x_end]

            if pad_h > 0 or pad_w > 0:
                crop_label = np.pad(
                    crop_label,
                    ((0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0,
                )
                crop_valid_mask = np.pad(
                    crop_valid_mask,
                    ((0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0,
                )

            # Binarize label (0 or 1)
            crop_label = (crop_label > 0).astype(np.float32)
            crop_valid_mask = (crop_valid_mask > 0).astype(np.float32)

            transform_data["mask"] = crop_label
            transform_data["valid_mask"] = crop_valid_mask

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(**transform_data)
            image = augmented["image"]  # (C, H, W)

            if self.mode != "test":
                mask = augmented["mask"]  # (H, W) or (1, H, W) depending on ToTensorV2
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)

                # We don't strictly need valid_mask for the model input, but useful for loss masking if needed
                # Here we just return image and label
                return image.float() / Config.PIXEL_MAX, mask.float()

        # Inference mode return
        # Normalize
        image = (
            torch.from_numpy(crop_mips.transpose(2, 0, 1)).float() / Config.PIXEL_MAX
        )
        return image, torch.tensor([0])  # Dummy label


def generate_test_metadata(test_csv_path):
    """
    Generates patch metadata for test fragments by tiling them.
    """
    df_test_frags = pd.read_csv(test_csv_path)
    patches = []

    for _, row in df_test_frags.iterrows():
        fid = row["fragment_id"]
        mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])

        if not os.path.exists(mask_path):
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        h, w = mask.shape

        # Generate tiles
        for y in range(0, h, Config.STRIDE):
            for x in range(0, w, Config.STRIDE):
                patches.append(
                    {
                        "fragment_id": fid,
                        "x": x,
                        "y": y,
                        "width": Config.TILE_SIZE,
                        "height": Config.TILE_SIZE,
                        "mask_path": row["mask_path"],
                        "volume_path": row["volume_path"],
                    }
                )

    return pd.DataFrame(patches)


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, validation, and test sets.
    """
    # Load DataFrames
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VALID_METADATA_PATH)

    # Debug mode: subset data
    if Config.DEBUG:
        train_df = train_df.head(100)
        val_df = val_df.head(50)

    # Create Datasets
    train_ds = InkDataset(
        train_df,
        mode="train",
        transforms=get_transforms("train"),
        load_cached_data=load_cached_data,
    )

    val_ds = InkDataset(
        val_df,
        mode="val",
        transforms=get_transforms("val"),
        load_cached_data=load_cached_data,
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
        drop_last=False,
    )

    # Test Loader
    test_patch_df = generate_test_metadata(Config.TEST_METADATA_PATH)
    test_ds = InkDataset(
        test_patch_df,
        mode="test",
        transforms=get_transforms("test"),
        load_cached_data=load_cached_data,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, test_patch_df
