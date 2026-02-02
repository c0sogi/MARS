import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import get_fragment_slab, set_seed

# Set fixed seed for reproducibility
set_seed(Config.SEED)


class InkDataset(Dataset):
    """
    Dataset for the Matched-Depth Specialist Ensemble.
    Generates 3-channel MIP slabs based on the specific Z-range of the specialist model.
    """

    def __init__(self, metadata_df, specialist_type, mode="train"):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing patch metadata.
            specialist_type (str): 'High', 'Mid', or 'Low'. Determines Z-range.
            mode (str): 'train', 'val', or 'test'. Controls augmentation and label loading.
        """
        self.metadata = metadata_df.reset_index(drop=True)
        self.mode = mode
        self.specialist_type = specialist_type

        # Validate specialist type and get Z-range
        if specialist_type not in Config.Z_RANGES:
            raise ValueError(
                f"Unknown specialist_type: {specialist_type}. Must be one of {list(Config.Z_RANGES.keys())}"
            )

        self.z_range = Config.Z_RANGES[specialist_type]

        # Pre-load data into memory to avoid repeated IO/MIP computation
        self.fragments = {}
        unique_frags = self.metadata["fragment_id"].unique()

        print(
            f"Initializing InkDataset ({mode}) for Specialist: {specialist_type} (Z-Range: {self.z_range})"
        )

        for fid in unique_frags:
            # Get paths from the first entry of this fragment
            # Metadata paths are relative to input dir
            frag_meta = self.metadata[self.metadata["fragment_id"] == fid].iloc[0]

            # 1. Load Input Slab (3-channel MIP)
            # get_fragment_slab handles caching to ./working/idea_23/
            slab = get_fragment_slab(
                fragment_id=str(fid),
                volume_path=frag_meta["volume_path"],
                z_range=self.z_range,
                load_cached_data=True,
            )

            # 2. Load Label (Ink) - Only for train/val
            label_img = None
            if mode in ["train", "val"]:
                label_path = os.path.join(Config.INPUT_DIR, frag_meta["label_path"])
                if os.path.exists(label_path):
                    label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                    # Binarize and convert to float for BCE/Dice
                    label_img = (label_img > 0).astype(np.float32)
                else:
                    raise FileNotFoundError(f"Label file not found: {label_path}")

            self.fragments[fid] = {
                "slab": slab,  # Shape: (H_frag, W_frag, 3), float32 [0,1]
                "label": label_img,  # Shape: (H_frag, W_frag), float32 {0,1} or None
            }

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        fid = row["fragment_id"]
        x, y = row["x"], row["y"]
        w, h = row["width"], row["height"]

        frag_data = self.fragments[fid]
        slab = frag_data["slab"]
        label_full = frag_data["label"]

        # --- Cropping ---
        # Ensure we stay within bounds (metadata should be correct, but safety first)
        slab_h, slab_w = slab.shape[:2]
        y_end = min(y + h, slab_h)
        x_end = min(x + w, slab_w)

        # Crop Image
        image = slab[y:y_end, x:x_end, :].copy()

        # Crop Label
        label = None
        if label_full is not None:
            label = label_full[y:y_end, x:x_end].copy()

        # --- Padding ---
        # If the crop is smaller than target size (edges), pad with zeros
        cur_h, cur_w = image.shape[:2]
        if cur_h < h or cur_w < w:
            pad_h = h - cur_h
            pad_w = w - cur_w
            # Pad (H, W, C)
            image = np.pad(
                image,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            if label is not None:
                # Pad (H, W)
                label = np.pad(
                    label, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0
                )

        # --- Augmentation (Geometric Only) ---
        if self.mode == "train":
            # Random Horizontal Flip
            if np.random.rand() < 0.5:
                image = np.fliplr(image)
                label = np.fliplr(label)

            # Random Vertical Flip
            if np.random.rand() < 0.5:
                image = np.flipud(image)
                label = np.flipud(label)

            # Random Rotate 90 (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            if k > 0:
                # rot90 rotates axes 0 and 1 by default, which is H and W
                image = np.rot90(image, k)
                label = np.rot90(label, k)

        # --- To Tensor ---
        # Image: (H, W, 3) -> (3, H, W)
        image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()

        if label is not None:
            # Label: (H, W) -> (1, H, W)
            label = torch.from_numpy(np.ascontiguousarray(label)).float().unsqueeze(0)
            return image, label

        return image


def get_dataloaders(specialist_type):
    """
    Creates training and validation DataLoaders for a specific specialist model.

    Args:
        specialist_type (str): 'High', 'Mid', or 'Low'.

    Returns:
        tuple: (train_loader, val_loader)
    """
    print(f"\nLoading Metadata for Specialist: {specialist_type}...")

    # Load Metadata
    if not os.path.exists(Config.METADATA_TRAIN_PATH) or not os.path.exists(
        Config.METADATA_VAL_PATH
    ):
        raise FileNotFoundError(
            "Metadata files not found. Please run metadata generation script first."
        )

    train_df = pd.read_csv(Config.METADATA_TRAIN_PATH)
    val_df = pd.read_csv(Config.METADATA_VAL_PATH)

    # Debugging / Quick Run
    if Config.DEBUG:
        limit = Config.MAX_TRAIN_SAMPLES if Config.MAX_TRAIN_SAMPLES else 100
        print(f"DEBUG MODE: Limiting training data to {limit} samples.")
        train_df = train_df.iloc[:limit]
        val_df = val_df.iloc[:limit]

    # Create Datasets
    train_dataset = InkDataset(train_df, specialist_type, mode="train")
    val_dataset = InkDataset(val_df, specialist_type, mode="val")

    print(f"Train Dataset: {len(train_dataset)} samples")
    print(f"Val Dataset:   {len(val_dataset)} samples")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches to maintain batch statistics
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
