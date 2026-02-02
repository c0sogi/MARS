import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config, AshConfig


def get_transforms(stage: str):
    """
    Returns the Albumentations composition of transforms for the specific stage.

    Args:
        stage (str): 'train', 'validation', or 'test'.

    Returns:
        A.Compose: Composed transforms.
    """
    if stage == "train":
        # Discrete geometric transformations only
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: Just convert to tensor
        return A.Compose([ToTensorV2()])


class ContrailDataset(Dataset):
    """
    PyTorch Dataset for Contrail Identification.
    Loads satellite bands, creates Ash False-Color Composite, and applies normalization.
    """

    def __init__(self, metadata_csv_path, stage="train", transform=None):
        """
        Args:
            metadata_csv_path (str): Path to the metadata CSV file.
            stage (str): 'train', 'validation', or 'test'.
            transform (A.Compose): Albumentations transforms.
        """
        self.stage = stage
        self.transform = transform

        # Load metadata
        if os.path.exists(metadata_csv_path):
            self.df = pd.read_csv(metadata_csv_path)
        else:
            raise FileNotFoundError(f"Metadata file not found: {metadata_csv_path}")

    def __len__(self):
        return len(self.df)

    def normalize_range(self, data, min_v, max_v):
        """
        Clips data to [min_v, max_v] and normalizes to [0, 1].
        """
        data = np.clip(data, min_v, max_v)
        data = (data - min_v) / (max_v - min_v)
        return data

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_id = str(row["record_id"])

        # 1. Load Bands
        # We need Band 11, 14, 15 for Ash Composite
        # Paths are relative in CSV, need to join with INPUT_DIR
        path_b11 = os.path.join(Config.INPUT_DIR, row["band_11"])
        path_b14 = os.path.join(Config.INPUT_DIR, row["band_14"])
        path_b15 = os.path.join(Config.INPUT_DIR, row["band_15"])

        try:
            # Load full sequence: H x W x T
            # We only need the labeled frame (index 4, defined by N_TIMES_BEFORE)
            t_idx = Config.N_TIMES_BEFORE

            # Load specific time step directly if possible, but NPY usually loads full array.
            # Given file size (~2MB), loading full array is acceptable.
            b11 = np.load(path_b11)[:, :, t_idx]
            b14 = np.load(path_b14)[:, :, t_idx]
            b15 = np.load(path_b15)[:, :, t_idx]

        except Exception as e:
            print(f"Error loading bands for record {record_id}: {e}")
            # Fallback zero array
            b11 = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)
            b14 = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)
            b15 = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        # 2. Create Ash False-Color Composite
        # Red: Optical Depth Proxy (Band 15 - Band 14)
        r = b15 - b14

        # Green: Particle Phase Proxy (Band 14 - Band 11)
        g = b14 - b11

        # Blue: Temperature (Band 14)
        b = b14

        # 3. Normalize
        # Apply strict bounds to maximize contrast
        r = self.normalize_range(r, AshConfig.RED_MIN, AshConfig.RED_MAX)
        g = self.normalize_range(g, AshConfig.GREEN_MIN, AshConfig.GREEN_MAX)
        b = self.normalize_range(b, AshConfig.BLUE_MIN, AshConfig.BLUE_MAX)

        # Stack to H x W x 3 for Albumentations
        image = np.stack([r, g, b], axis=-1).astype(np.float32)

        # 4. Load Mask (if available)
        mask = None
        if self.stage in ["train", "validation"]:
            mask_path_rel = row.get("human_pixel_masks")

            if pd.notna(mask_path_rel):
                mask_path = os.path.join(Config.INPUT_DIR, mask_path_rel)
                try:
                    # Mask shape: H x W x 1
                    mask = np.load(mask_path).astype(np.float32)
                except Exception as e:
                    print(f"Error loading mask for record {record_id}: {e}")
                    mask = np.zeros(
                        (Config.IMG_SIZE, Config.IMG_SIZE, 1), dtype=np.float32
                    )
            else:
                # Validation set might not have individual masks, but should have pixel masks if labeled.
                # If missing, return zero mask.
                mask = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 1), dtype=np.float32)

        # 5. Apply Transforms
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]

        # 6. Post-process Mask
        if mask is not None:
            # Convert to tensor if not already (in case ToTensorV2 wasn't applied/configured for mask)
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask)

            # Ensure shape is (1, H, W)
            # Albumentations ToTensorV2 usually converts (H, W, 1) -> (1, H, W)
            # or (H, W) -> (H, W). We need (1, H, W) for BCE/Dice Loss.
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            elif mask.ndim == 3:
                # If it is (H, W, 1), permute it.
                # If it is (1, H, W) or (C, H, W), leave it.
                if mask.shape[2] == 1:
                    mask = mask.permute(2, 0, 1)
        else:
            # Return dummy mask for test set
            mask = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32)

        return image, mask
