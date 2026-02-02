import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import get_cached_volume, create_3ch_input, get_transforms


class InkDataset(Dataset):
    """
    Dataset class for Vesuvius Ink Detection.
    Implements the 'Overlapping Thick Slab' strategy by loading specific 3D volume slabs
    and projecting them into 3-channel inputs.
    """

    def __init__(self, df, split, z_start=Config.Z_START_TRAIN, transforms=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing patch coordinates and paths.
            split (str): 'train', 'valid', or 'test'.
            z_start (int): The starting Z-index for the context window.
            transforms (albumentations.Compose): Transforms to apply.
        """
        self.df = df
        self.split = split
        self.z_start = z_start
        self.transforms = transforms

        # In-memory cache for volumes to avoid repeated loading/deserialization
        # Key: fragment_id, Value: np.ndarray (65, H, W)
        self.volume_cache = {}

    def __len__(self):
        return len(self.df)

    def _get_volume(self, fragment_id, rel_volume_path):
        """
        Retrieves the full 3D volume for a fragment, using RAM cache and Disk cache.
        """
        if fragment_id not in self.volume_cache:
            # Construct full path to the volume directory
            full_vol_dir = os.path.join(Config.INPUT_DIR, rel_volume_path)
            # Load from disk (utils handles .npy caching logic)
            self.volume_cache[fragment_id] = get_cached_volume(
                fragment_id, full_vol_dir
            )
        return self.volume_cache[fragment_id]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fragment_id = str(row["fragment_id"])

        # 1. Load Volume
        volume = self._get_volume(fragment_id, row["volume_path"])

        # 2. Define Patch Coordinates
        x, y = row["x"], row["y"]
        w, h = row["width"], row["height"]

        # 3. Extract Z-Slab
        # We take a slab of defined depth starting from z_start
        z_end = self.z_start + Config.Z_CONTEXT_DEPTH
        # Slice: [Depth, Height, Width]
        slab = volume[self.z_start : z_end, y : y + h, x : x + w]

        # 4. Handle Padding (for edge tiles)
        pad_h = Config.TILE_SIZE - h
        pad_w = Config.TILE_SIZE - w

        if pad_h > 0 or pad_w > 0:
            # Pad spatial dimensions (H, W) with 0
            slab = np.pad(
                slab,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=0,
            )

        # 5. Project to 3-Channel Input
        # Input: (D, H, W) -> Output: (H, W, 3)
        image = create_3ch_input(slab)

        # 6. Load Mask (Validity Mask)
        mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
        mask_full = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask_full is None:
            # Fallback
            mask_patch = np.zeros((h, w), dtype=np.uint8)
        else:
            mask_patch = mask_full[y : y + h, x : x + w]

        if pad_h > 0 or pad_w > 0:
            mask_patch = np.pad(
                mask_patch, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0
            )

        # Convert to float mask (0.0 or 1.0)
        mask_patch = (mask_patch > 0).astype(np.float32)

        # 7. Load Label (Ink) - Only for train/valid
        label_patch = None
        if self.split in ["train", "valid"]:
            label_path = os.path.join(Config.INPUT_DIR, row["label_path"])
            label_full = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

            if label_full is not None:
                label_patch = label_full[y : y + h, x : x + w]
                if pad_h > 0 or pad_w > 0:
                    label_patch = np.pad(
                        label_patch,
                        ((0, pad_h), (0, pad_w)),
                        mode="constant",
                        constant_values=0,
                    )
                label_patch = (label_patch > 0).astype(np.float32)
            else:
                label_patch = np.zeros(
                    (Config.TILE_SIZE, Config.TILE_SIZE), dtype=np.float32
                )

        # 8. Apply Transforms
        if self.transforms:
            if label_patch is not None:
                # Albumentations expects image (H,W,C) and mask (H,W)
                augmented = self.transforms(image=image, mask=label_patch)
                image = augmented["image"]
                label_patch = augmented["mask"]
            else:
                augmented = self.transforms(image=image)
                image = augmented["image"]

        # 9. Format Output
        # Ensure label is (1, H, W)
        if label_patch is not None:
            if isinstance(label_patch, np.ndarray):
                label_patch = torch.from_numpy(label_patch)
            if label_patch.ndim == 2:
                label_patch = label_patch.unsqueeze(0)

        # Ensure mask is tensor (1, H, W)
        if isinstance(mask_patch, np.ndarray):
            mask_patch = torch.from_numpy(mask_patch)
        if mask_patch.ndim == 2:
            mask_patch = mask_patch.unsqueeze(0)

        sample = {
            "image": image,
            "fragment_id": fragment_id,
            "x": x,
            "y": y,
            "valid_mask": mask_patch,
        }

        if label_patch is not None:
            sample["label"] = label_patch

        return sample


def get_test_patches(test_df):
    """
    Generates a dataframe of patches for the test fragments.
    Uses non-overlapping tiling (Stride = Tile Size).
    """
    patches = []
    stride = Config.TILE_SIZE

    for _, row in test_df.iterrows():
        frag_id = row["fragment_id"]
        mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])

        # Load mask to get dimensions
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        h_img, w_img = mask.shape

        for y in range(0, h_img, stride):
            for x in range(0, w_img, stride):
                # Calculate patch dimensions (clip at edges)
                w_patch = min(Config.TILE_SIZE, w_img - x)
                h_patch = min(Config.TILE_SIZE, h_img - y)

                patches.append(
                    {
                        "fragment_id": frag_id,
                        "mask_path": row["mask_path"],
                        "volume_path": row["volume_path"],
                        "x": x,
                        "y": y,
                        "width": w_patch,
                        "height": h_patch,
                    }
                )

    return pd.DataFrame(patches)


def get_dataset(split, z_start=Config.Z_START_TRAIN):
    """
    Factory function to create an InkDataset.

    Args:
        split (str): 'train', 'valid', or 'test'.
        z_start (int): Starting Z-slice index.

    Returns:
        InkDataset: The configured dataset.
    """
    if split == "train":
        df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    elif split == "valid":
        df = pd.read_csv(Config.VALID_METADATA_PATH)
    elif split == "test":
        test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
        df = get_test_patches(test_meta)
    else:
        raise ValueError(f"Unknown split: {split}")

    transforms = get_transforms(data=split)

    return InkDataset(df, split, z_start=z_start, transforms=transforms)
