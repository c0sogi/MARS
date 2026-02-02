import os
import numpy as np
import pandas as pd
import torch
import albumentations as A
from torch.utils.data import Dataset
from library.config import Config


class ContrailDataset(Dataset):
    """
    PyTorch Dataset for Contrail Detection.
    Loads Band 11, 14, 15 to create an 'Ash' color composite.
    """

    def __init__(self, split="train", max_samples=None, transform=None):
        """
        Args:
            split (str): One of 'train', 'validation', 'test'.
            max_samples (int, optional): Limit the number of samples for debugging.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.split = split
        self.transform = transform

        # Determine which metadata file to load
        if split == "train":
            self.metadata_path = Config.TRAIN_METADATA_PATH
        elif split == "validation":
            self.metadata_path = Config.VAL_METADATA_PATH
        elif split == "test":
            self.metadata_path = Config.TEST_METADATA_PATH
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train', 'validation', or 'test'."
            )

        # Load metadata
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        self.df = pd.read_csv(self.metadata_path)

        # Debugging: Limit dataset size
        if max_samples is not None:
            self.df = self.df.iloc[:max_samples].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def normalize_range(self, data, bounds):
        """
        Normalizes data to [0, 1] based on provided (min, max) bounds.
        Clips values outside the range.
        """
        return (data - bounds[0]) / (bounds[1] - bounds[0])

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_id = str(row["record_id"])

        # ----------------------------------------------------------------------
        # 1. Load Bands (11, 14, 15)
        # ----------------------------------------------------------------------
        # Paths are relative in the CSV, need to join with INPUT_DIR
        # We need the 5th image in the sequence (index 4)

        try:
            # Load Band 11
            p11 = os.path.join(Config.INPUT_DIR, row["band_11"])
            b11 = np.load(p11)
            # Check shape: H x W x T. We want index 4.
            img_11 = b11[..., 4].astype(np.float32)

            # Load Band 14
            p14 = os.path.join(Config.INPUT_DIR, row["band_14"])
            b14 = np.load(p14)
            img_14 = b14[..., 4].astype(np.float32)

            # Load Band 15
            p15 = os.path.join(Config.INPUT_DIR, row["band_15"])
            b15 = np.load(p15)
            img_15 = b15[..., 4].astype(np.float32)

        except Exception as e:
            # Fallback for corrupt files or missing paths (though metadata should be clean)
            print(f"Error loading bands for record {record_id}: {e}")
            # Return zeros of correct shape
            H, W = Config.IMAGE_SIZE, Config.IMAGE_SIZE
            img = torch.zeros((Config.NUM_BANDS, H, W), dtype=torch.float32)
            mask = torch.zeros((1, H, W), dtype=torch.float32)
            return img, mask

        # ----------------------------------------------------------------------
        # 2. Create Ash Composite
        # ----------------------------------------------------------------------
        # Red: Band 15 - Band 14 (Optical Depth)
        r = img_15 - img_14

        # Green: Band 14 - Band 11 (Particle Phase)
        g = img_14 - img_11

        # Blue: Band 14 (Temperature)
        b = img_14

        # ----------------------------------------------------------------------
        # 3. Normalize
        # ----------------------------------------------------------------------
        r_norm = self.normalize_range(r, Config.ASH_RED_BOUNDS)
        g_norm = self.normalize_range(g, Config.ASH_GREEN_BOUNDS)
        b_norm = self.normalize_range(b, Config.ASH_BLUE_BOUNDS)

        # Clip to [0, 1]
        r_norm = np.clip(r_norm, 0, 1)
        g_norm = np.clip(g_norm, 0, 1)
        b_norm = np.clip(b_norm, 0, 1)

        # Stack: (H, W, 3)
        img_composite = np.stack([r_norm, g_norm, b_norm], axis=-1)

        # ----------------------------------------------------------------------
        # 4. Load Mask (if available)
        # ----------------------------------------------------------------------
        mask = None
        if self.split != "test":
            mask_path_rel = row.get("human_pixel_masks")
            if pd.notna(mask_path_rel):
                full_mask_path = os.path.join(Config.INPUT_DIR, mask_path_rel)
                if os.path.exists(full_mask_path):
                    # Shape: H x W x 1
                    mask_arr = np.load(full_mask_path).astype(np.float32)
                    mask = mask_arr
                else:
                    # Should not happen if metadata is verified
                    mask = np.zeros(
                        (img_composite.shape[0], img_composite.shape[1], 1),
                        dtype=np.float32,
                    )
            else:
                mask = np.zeros(
                    (img_composite.shape[0], img_composite.shape[1], 1),
                    dtype=np.float32,
                )
        else:
            # Dummy mask for test set
            mask = np.zeros(
                (img_composite.shape[0], img_composite.shape[1], 1), dtype=np.float32
            )

        # ----------------------------------------------------------------------
        # 5. Apply Transforms & Convert to Tensor
        # ----------------------------------------------------------------------
        # If using albumentations or similar, they usually expect HWC numpy arrays
        if self.transform:
            augmented = self.transform(image=img_composite, mask=mask)
            img_composite = augmented["image"]
            mask = augmented["mask"]

        # Convert to Torch Tensors
        # Image: HWC -> CHW
        img_tensor = torch.from_numpy(img_composite).permute(2, 0, 1).float()

        # Mask: HWC -> CHW (usually 1, H, W)
        # If mask is already tensor (from transform), skip
        if not isinstance(mask, torch.Tensor):
            mask_tensor = torch.from_numpy(mask).permute(2, 0, 1).float()
        else:
            mask_tensor = mask

        return img_tensor, mask_tensor
