import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import load_image, rle_decode


def process_metadata(df, phase, load_cached_data=True):
    """
    Processes the raw metadata dataframe into a format suitable for the dataset.
    Groups entries by (case, day, slice) and merges class-specific rows.
    Implements caching using Parquet.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, f"processed_dataset_{phase}.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            processed_df = pd.read_parquet(cache_path)
            return processed_df
        except Exception:
            # If load fails, proceed to process from scratch
            pass

    # 2. Process from scratch
    # Group by unique slice identifier
    groups = df.groupby(["case", "day", "slice"])

    data_list = []
    classes = Config.CLASSES

    for (case, day, slc), group in groups:
        # Extract common slice info from the first row of the group
        row = group.iloc[0]
        entry = {
            "id": row["id"],
            "case": case,
            "day": day,
            "slice": slc,
            "file_path": row["file_path"],
            "pixel_spacing_h": row["pixel_spacing_h"],
            "pixel_spacing_w": row["pixel_spacing_w"],
            "img_height": row["img_height"],
            "img_width": row["img_width"],
        }

        # Initialize RLE columns
        for c in classes:
            entry[f"rle_{c}"] = ""

        # If not testing, populate RLEs from the group
        if phase != "test":
            for _, r in group.iterrows():
                if r["class"] in classes:
                    # Handle potential NaN values in segmentation
                    seg = r.get("segmentation", "")
                    entry[f"rle_{r['class']}"] = seg if pd.notna(seg) else ""

        data_list.append(entry)

    processed_df = pd.DataFrame(data_list)

    # 3. Save to cache
    processed_df.to_parquet(cache_path, index=False)

    return processed_df


class UWMadisonDataset(Dataset):
    def __init__(self, df, phase="train", transform=None, load_cached_data=True):
        """
        Args:
            df (pd.DataFrame): Raw metadata dataframe.
            phase (str): 'train', 'val', or 'test'.
            transform (albumentations.Compose): Transforms to apply.
            load_cached_data (bool): Whether to use cached processed metadata.
        """
        self.phase = phase
        self.transform = transform

        # Process and load metadata
        self.df = process_metadata(df, phase, load_cached_data)

        # Build a lookup dictionary for fast 2.5D context retrieval
        # Maps (case, day, slice) -> DataFrame index
        self.lookup = {}
        for idx, row in self.df.iterrows():
            key = (row["case"], row["day"], row["slice"])
            self.lookup[key] = idx

    def __len__(self):
        if Config.SAMPLE_SIZE and Config.DEBUG:
            return min(len(self.df), Config.SAMPLE_SIZE)
        return len(self.df)

    def load_2_5d_stack(self, case, day, slice_idx):
        """
        Loads a 2.5D stack of images (slice i-1, i, i+1).
        Handles boundary conditions by replicating the nearest available slice.
        """
        imgs = []
        # Offsets for 2.5D: Previous, Current, Next
        for d in [-1, 0, 1]:
            key = (case, day, slice_idx + d)

            if key in self.lookup:
                idx = self.lookup[key]
                row = self.df.iloc[idx]
                path = row["file_path"]
            else:
                # If neighbor is missing (start/end of scan), use the center slice
                # We assume the center slice (d=0) always exists if this method is called
                center_key = (case, day, slice_idx)
                if center_key in self.lookup:
                    idx = self.lookup[center_key]
                    row = self.df.iloc[idx]
                    path = row["file_path"]
                else:
                    raise ValueError(f"Center slice {center_key} not found in lookup.")

            full_path = os.path.join(Config.INPUT_DIR, path)
            img = load_image(full_path)
            imgs.append(img)

        # Stack along the channel dimension -> (H, W, 3)
        return np.stack(imgs, axis=-1)

    def physical_resample(self, image, mask, current_spacing, orig_shape):
        """
        Resamples the image and mask to the target physical spacing.

        Args:
            image (np.ndarray): Input image (H, W, C).
            mask (np.ndarray or None): Input mask (H, W, C) or None.
            current_spacing (tuple): (pixel_spacing_h, pixel_spacing_w).
            orig_shape (tuple): (original_height, original_width).

        Returns:
            tuple: (resampled_image, resampled_mask)
        """
        scale_h = current_spacing[0] / Config.TARGET_SPACING
        scale_w = current_spacing[1] / Config.TARGET_SPACING

        # Avoid resizing if scales are effectively 1.0
        if abs(scale_h - 1.0) < 1e-3 and abs(scale_w - 1.0) < 1e-3:
            return image, mask

        new_h = int(round(orig_shape[0] * scale_h))
        new_w = int(round(orig_shape[1] * scale_w))

        # Resize image using Linear interpolation
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Resize mask using Nearest Neighbor to preserve binary classes
        if mask is not None:
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        return image, mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        case, day, slc = row["case"], row["day"], row["slice"]
        orig_h, orig_w = row["img_height"], row["img_width"]

        # 1. Load 2.5D Image Stack
        image = self.load_2_5d_stack(case, day, slc)

        # 2. Load Mask (if available)
        mask = None
        if self.phase != "test":
            mask = np.zeros((orig_h, orig_w, Config.NUM_CLASSES), dtype=np.float32)
            for i, cls_name in enumerate(Config.CLASSES):
                rle = row[f"rle_{cls_name}"]
                if pd.notna(rle) and rle != "":
                    mask[:, :, i] = rle_decode(rle, (orig_h, orig_w))

        # 3. Physical Space Normalization
        current_spacing = (row["pixel_spacing_h"], row["pixel_spacing_w"])
        image, mask = self.physical_resample(
            image, mask, current_spacing, (orig_h, orig_w)
        )

        # 4. Augmentations and Tensor Conversion
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                # For test set, only transform image
                augmented = self.transform(image=image)
                image = augmented["image"]
        else:
            # Fallback to simple ToTensor if no transforms provided
            # Transpose HWC -> CHW
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            if mask is not None:
                mask = torch.from_numpy(mask.transpose(2, 0, 1)).float()

        result = {
            "image": image,
            "id": row["id"],
            "orig_shape": np.array([orig_h, orig_w]),
            "case": case,
            "day": day,
            "slice": slc,
        }

        if mask is not None:
            result["mask"] = mask

        return result
