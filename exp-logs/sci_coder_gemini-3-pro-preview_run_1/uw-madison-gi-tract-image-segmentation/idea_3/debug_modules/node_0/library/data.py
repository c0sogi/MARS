import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import rle_decode


class UWMadisonDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (and masks for train/val).
            transforms (albumentations.Compose): Augmentation pipeline.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.mode = mode
        self.classes = Config.CLASSES

        # Ensure sorting for 2.5D logic (case, day, slice)
        # Convert slice to int for correct numerical sorting if it's string
        if self.df["slice"].dtype == "O":
            self.df["slice_int"] = self.df["slice"].astype(int)
            self.df = self.df.sort_values(["case", "day", "slice_int"]).reset_index(
                drop=True
            )
        else:
            self.df = self.df.sort_values(["case", "day", "slice"]).reset_index(
                drop=True
            )

        # Pre-calculate indices for 2.5D context (previous, current, next)
        self.indices = np.arange(len(self.df))
        self.case_day = (self.df["case"] + "_" + self.df["day"]).values

    def __len__(self):
        return len(self.df)

    def load_slice_img(self, path):
        """
        Loads an image, applies percentile normalization, and scales to [0, 1].
        """
        full_path = os.path.join(Config.INPUT_DIR, path)
        img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            # Fallback for missing files (though validation checked this)
            # Create a black image of default size
            img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)
        else:
            img = img.astype(np.float32)

        # Percentile Normalization (Robust Min-Max)
        # Handle cases where image might be completely black
        if np.max(img) > 0:
            p_lo = np.percentile(img, 1)
            p_hi = np.percentile(img, 99)

            if p_hi > p_lo:
                img = np.clip(img, p_lo, p_hi)
                img = (img - p_lo) / (p_hi - p_lo)
            else:
                # If dynamic range is zero or very small, just normalize by max or keep 0
                if p_hi > 0:
                    img = img / p_hi
                else:
                    img[:] = 0.0
        else:
            img[:] = 0.0

        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 2.5D Logic: Retrieve indices for z-1, z, z+1
        # We check if neighbors belong to the same case_day group.
        # If not (boundary), we replicate the current slice.
        current_group = self.case_day[idx]

        # Previous slice
        if idx > 0 and self.case_day[idx - 1] == current_group:
            idx_prev = idx - 1
        else:
            idx_prev = idx

        # Next slice
        if idx < len(self.df) - 1 and self.case_day[idx + 1] == current_group:
            idx_next = idx + 1
        else:
            idx_next = idx

        # Load images
        path_prev = self.df.iloc[idx_prev]["file_path"]
        path_curr = row["file_path"]
        path_next = self.df.iloc[idx_next]["file_path"]

        img_prev = self.load_slice_img(path_prev)
        img_curr = self.load_slice_img(path_curr)
        img_next = self.load_slice_img(path_next)

        # Stack to create (H, W, 3)
        # Resize is handled by albumentations, but initial images might be different sizes
        # The dataset images are mostly 266x266 or 360x310.
        # We stack first, then resize. But to stack, they must be same size.
        # Within a case/day, resolution is constant.
        img_stack = np.stack([img_prev, img_curr, img_next], axis=-1)  # (H, W, 3)

        # Prepare Mask (only for train/val)
        mask = None
        if self.mode in ["train", "val"]:
            h, w = row["height"], row["width"]
            mask = np.zeros((h, w, len(self.classes)), dtype=np.float32)

            for i, class_name in enumerate(self.classes):
                rle = row[class_name]
                if pd.notna(rle) and rle != "":
                    mask[..., i] = rle_decode(rle, (h, w))

        # Apply Augmentations
        if self.transforms:
            if self.mode in ["train", "val"]:
                augmented = self.transforms(image=img_stack, mask=mask)
                img_stack = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transforms(image=img_stack)
                img_stack = augmented["image"]

        # Permute Mask for PyTorch (H, W, C) -> (C, H, W)
        if mask is not None:
            mask = mask.permute(
                2, 0, 1
            )  # ToTensorV2 converts to tensor but keeps HWC for mask sometimes?
            # Actually ToTensorV2 converts image to (C, H, W) and mask to (H, W) or (H, W, C).
            # If mask is multi-channel, ToTensorV2 usually keeps it (H, W, C) unless transpose_mask is True.
            # However, standard practice with ToTensorV2 is manual check.
            # Let's rely on ToTensorV2 behavior:
            # Image becomes (3, H, W). Mask (if passed) becomes Tensor.
            # If mask was (H, W, 3), ToTensorV2 returns (3, H, W) usually.
            pass

        result = {
            "image": img_stack,
            "id": row["id"],
            "slice_info": f"{row['case']}_{row['day']}_{row['slice']}",
        }

        if mask is not None:
            result["mask"] = mask

        return result


def get_transforms(mode="train"):
    """
    Returns the albumentations transform pipeline for the specified mode.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(
                    Config.IMG_SIZE, Config.IMG_SIZE, interpolation=cv2.INTER_LINEAR
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                A.OneOf(
                    [
                        A.ElasticTransform(
                            alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5
                        ),
                        A.GridDistortion(p=0.5),
                        A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=0.5),
                    ],
                    p=0.3,
                ),
                # Normalize is not strictly needed if we did manual 0-1 scaling,
                # but standardizing mean/std can help convergence.
                # Since we did percentile scaling to 0-1, we can skip standard Normalize
                # or use mean=0.5, std=0.5 to center it.
                # Let's stick to simple ToTensorV2 which converts to float tensor.
                ToTensorV2(transpose_mask=True),
            ]
        )
    else:
        # Validation / Test
        return A.Compose(
            [
                A.Resize(
                    Config.IMG_SIZE, Config.IMG_SIZE, interpolation=cv2.INTER_LINEAR
                ),
                ToTensorV2(transpose_mask=True),
            ]
        )


def prepare_loaders(debug=Config.DEBUG):
    """
    Loads metadata, creates Datasets and DataLoaders.

    Args:
        debug (bool): If True, subsamples the data for quick debugging.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load Metadata
    # Using keep_default_na=False to handle empty RLE strings correctly
    df_train = pd.read_csv(Config.TRAIN_CSV, keep_default_na=False)
    df_val = pd.read_csv(Config.VAL_CSV, keep_default_na=False)
    df_test = pd.read_csv(Config.TEST_CSV, keep_default_na=False)

    if debug:
        print(f"Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    # Datasets
    train_dataset = UWMadisonDataset(
        df_train, transforms=get_transforms("train"), mode="train"
    )

    val_dataset = UWMadisonDataset(df_val, transforms=get_transforms("val"), mode="val")

    test_dataset = UWMadisonDataset(
        df_test, transforms=get_transforms("test"), mode="test"
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
