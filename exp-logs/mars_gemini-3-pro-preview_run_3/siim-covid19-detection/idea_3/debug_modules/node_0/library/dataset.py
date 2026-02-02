import os
import cv2
import torch
import pydicom
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from library import config, utils


def get_transforms(split):
    """
    Returns the Albumentations transformations for the specific split.
    """
    if split == "train":
        return A.Compose(
            [
                # Images are already resized to config.IMG_SIZE in cache, but we ensure it here
                A.Resize(config.IMG_SIZE, config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(config.IMG_SIZE, config.IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


class SIIMDataset(Dataset):
    def __init__(self, df, split, load_cached_data=True):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached metadata/images.
        """
        self.split = split
        # 1. Prepare Metadata (ensure height/width exist for mask generation)
        self.df = self._prepare_metadata(df, split, load_cached_data)

        # 2. Load Images (using utils caching mechanism)
        # This returns a dict {image_id: np.array(512, 512)}
        self.images_dict = utils.process_and_cache_images(
            self.df, split, load_cached_data
        )

        # 3. Setup Transforms
        self.transforms = get_transforms(split)

        # 4. Precompute Labels for Train/Val
        if self.split != "test":
            self.labels = self.df[config.STUDY_LABELS].values.astype(np.float32)
        else:
            self.labels = None

    def _prepare_metadata(self, df, split, load_cached_data):
        """
        Ensures the dataframe has 'height' and 'width' columns needed for
        mapping bounding boxes to masks. Caches the result to parquet.
        """
        cache_path = os.path.join(config.WORKING_DIR, f"{split}_meta_dims.parquet")

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                cached_df = pd.read_parquet(cache_path)
                # Verify it matches the input df length (basic integrity check)
                if len(cached_df) == len(df):
                    return cached_df
            except Exception:
                pass  # Fallback to recomputing

        # 2. Compute Dimensions
        # We need original dimensions to correctly rasterize the bounding boxes
        # before resizing the mask to 512x512.
        print(f"Extracting original dimensions for {split} set...")

        def get_dims(row):
            path = os.path.join(config.INPUT_DIR, row["file_path"])
            try:
                # Read only header for speed
                dcm = pydicom.dcmread(path, stop_before_pixels=True)
                return row["image_id"], int(dcm.Rows), int(dcm.Columns)
            except Exception:
                # Fallback to config size if read fails
                return row["image_id"], config.IMG_SIZE, config.IMG_SIZE

        # Convert to list of dicts for safe iteration
        rows = df.to_dict("records")

        with ThreadPoolExecutor(max_workers=config.NUM_WORKERS) as executor:
            results = list(executor.map(get_dims, rows))

        # Create a mapping
        dim_map = {r[0]: (r[1], r[2]) for r in results}

        # Assign to dataframe
        df = df.copy()
        df["height"] = df["image_id"].map(
            lambda x: dim_map.get(x, (config.IMG_SIZE, config.IMG_SIZE))[0]
        )
        df["width"] = df["image_id"].map(
            lambda x: dim_map.get(x, (config.IMG_SIZE, config.IMG_SIZE))[1]
        )

        # 3. Save to cache
        os.makedirs(config.WORKING_DIR, exist_ok=True)
        df.to_parquet(cache_path)

        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]

        # 1. Get Image
        # Images are already resized to config.IMG_SIZE and are uint8 [0, 255]
        image = self.images_dict.get(image_id)
        if image is None:
            # Fallback (should not happen if cache is correct)
            image = np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.uint8)

        # Convert Grayscale to RGB for MobileNetV2
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # 2. Get Mask & Label
        if self.split != "test":
            # Generate mask on original dimensions
            h, w = int(row["height"]), int(row["width"])
            boxes_str = row["boxes"]

            # mask is (H, W) with 0s and 1s
            mask = utils.box_to_mask(boxes_str, h, w)

            # Resize mask to target size
            mask = cv2.resize(
                mask,
                (config.IMG_SIZE, config.IMG_SIZE),
                interpolation=cv2.INTER_NEAREST,
            )

            label = self.labels[idx]
        else:
            # Dummy data for test set
            mask = np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.uint8)
            label = np.zeros(config.NUM_STUDY_CLASSES, dtype=np.float32)

        # 3. Augmentations
        # Albumentations expects image: (H, W, C), mask: (H, W)
        augmented = self.transforms(image=image, mask=mask)

        image_tensor = augmented["image"]
        mask_tensor = augmented["mask"]

        # Add channel dimension to mask: (H, W) -> (1, H, W)
        # mask_tensor is likely a float tensor after ToTensorV2 if it was float,
        # but box_to_mask returns uint8. ToTensorV2 converts to tensor but doesn't unsqueeze.
        # We ensure it is float and has channel dim.
        mask_tensor = mask_tensor.float().unsqueeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": torch.tensor(label, dtype=torch.float32),
            "study_id": row["study_id"],
            "image_id": image_id,
        }
