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


class UWGI25DDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train"):
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.classes = Config.CLASSES
        self.img_size = Config.IMG_SIZE

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load 2.5D images: prev, curr, next
        paths = [row["prev_path"], row["abs_path"], row["next_path"]]
        images = []
        for p in paths:
            # Load image (16-bit usually)
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img is None:
                # Fallback for missing files (though logic ensures paths exist or are replicated)
                img = np.zeros(self.img_size, dtype=np.uint16)

            # Resize
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)

            # Min-Max Normalization per slice
            img = img.astype(np.float32)
            max_val = img.max()
            if max_val > 0:
                img = img / max_val
            else:
                # If image is completely black
                img = img / 65535.0

            images.append(img)

        # Stack to (H, W, 3)
        img_stack = np.stack(images, axis=-1)

        # Prepare Mask
        mask = np.zeros(
            (self.img_size[0], self.img_size[1], len(self.classes)), dtype=np.float32
        )

        if self.mode in ["train", "val"]:
            for i, cls in enumerate(self.classes):
                col_name = f"segmentation_{cls}"
                rle = row.get(col_name, "")

                if pd.notna(rle) and rle != "":
                    # Decode RLE to original size
                    orig_h, orig_w = row["img_height"], row["img_width"]
                    mask_cls = rle_decode(rle, (orig_h, orig_w))

                    # Resize to model input size
                    mask_cls = cv2.resize(
                        mask_cls, self.img_size, interpolation=cv2.INTER_NEAREST
                    )
                    mask[:, :, i] = mask_cls

        # Apply Augmentations
        if self.transforms:
            augmented = self.transforms(image=img_stack, mask=mask)
            img_stack = augmented["image"]
            mask = augmented["mask"]
        else:
            # If no transforms, just convert to tensor format (C, H, W)
            # Albumentations ToTensorV2 handles this, but manual fallback:
            img_stack = np.transpose(img_stack, (2, 0, 1))
            img_stack = torch.from_numpy(img_stack)

            mask = np.transpose(mask, (2, 0, 1))
            mask = torch.from_numpy(mask)

        return img_stack, mask, row["id"]


def process_metadata(metadata_path, cache_name, load_cached_data=True):
    """
    Loads metadata, pivots to wide format, adds 2.5D neighbor paths, and caches.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, f"{cache_name}.parquet")

    if load_cached_data and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    df = pd.read_csv(metadata_path)

    # Columns that identify a unique slice
    index_cols = [
        "id",
        "case",
        "day",
        "slice",
        "file_path",
        "img_width",
        "img_height",
        "pixel_spacing_w",
        "pixel_spacing_h",
    ]

    # Pivot if segmentation data exists
    if "segmentation" in df.columns:
        # Pivot to wide format: one row per slice, columns for each class segmentation
        df_pivot = df.pivot_table(
            index=index_cols, columns="class", values="segmentation", aggfunc="first"
        ).reset_index()

        # Rename columns to be consistent
        for cls in Config.CLASSES:
            if cls not in df_pivot.columns:
                df_pivot[cls] = ""
            df_pivot.rename(columns={cls: f"segmentation_{cls}"}, inplace=True)

        # Fill NaNs with empty string for RLEs
        for cls in Config.CLASSES:
            df_pivot[f"segmentation_{cls}"] = df_pivot[f"segmentation_{cls}"].fillna("")
    else:
        # For test set, just drop duplicates to get unique slices
        df_pivot = df[index_cols].drop_duplicates().reset_index(drop=True)

    # Add absolute paths
    df_pivot["abs_path"] = df_pivot["file_path"].apply(
        lambda x: os.path.join(Config.INPUT_DIR, x)
    )

    # Create lookup for 2.5D context
    # Key: (case, day, slice) -> abs_path
    path_lookup = {}
    for idx, row in df_pivot.iterrows():
        path_lookup[(row["case"], row["day"], row["slice"])] = row["abs_path"]

    def get_neighbor_path(case, day, slice_num, offset):
        target = slice_num + offset
        key = (case, day, target)
        # Return neighbor path if exists, else replicate current slice (boundary condition)
        return path_lookup.get(key, path_lookup[(case, day, slice_num)])

    df_pivot["prev_path"] = df_pivot.apply(
        lambda r: get_neighbor_path(r["case"], r["day"], r["slice"], -1), axis=1
    )
    df_pivot["next_path"] = df_pivot.apply(
        lambda r: get_neighbor_path(r["case"], r["day"], r["slice"], 1), axis=1
    )

    # Save to cache
    df_pivot.to_parquet(cache_path, index=False)
    return df_pivot


def get_transforms(data="train"):
    if data == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                ToTensorV2(),
            ]
        )
    elif data == "valid" or data == "test":
        return A.Compose(
            [
                ToTensorV2(),
            ]
        )
    return A.Compose([ToTensorV2()])


def get_dataloaders(train_df, val_df, batch_size=32, num_workers=4):
    # Balanced Sampling for Training
    # 1. Identify rows with masks
    mask_cols = [f"segmentation_{c}" for c in Config.CLASSES]

    # Check if any mask column has a non-empty string
    # We assume empty string means no mask
    has_mask = train_df[mask_cols].apply(lambda x: any(v != "" for v in x), axis=1)

    pos_df = train_df[has_mask].copy()
    neg_df = train_df[~has_mask].copy()

    # 2. Subsample negatives (50% as per idea description)
    if len(neg_df) > 0:
        neg_df = neg_df.sample(frac=0.5, random_state=Config.SEED)

    # 3. Combine and shuffle
    train_balanced = (
        pd.concat([pos_df, neg_df])
        .sample(frac=1.0, random_state=Config.SEED)
        .reset_index(drop=True)
    )

    # Create Datasets
    train_ds = UWGI25DDataset(
        train_balanced, transforms=get_transforms("train"), mode="train"
    )

    val_ds = UWGI25DDataset(val_df, transforms=get_transforms("valid"), mode="val")

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    return train_loader, val_loader
