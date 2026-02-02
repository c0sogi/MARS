import os
import cv2
import ast
import torch
import pydicom
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import bbox2mask


def get_transforms(split):
    """
    Returns the Albumentations transformations for the specific split.
    Enforces label-consistent CoarseDropout for training.
    """
    if split == "train":
        return A.Compose(
            [
                # Label-consistent CoarseDropout: mask_fill_value=0 ensures removed regions
                # in the image are also removed from the mask.
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(Config.img_size * 0.1),
                    max_width=int(Config.img_size * 0.1),
                    min_holes=1,
                    fill_value=0,
                    mask_fill_value=0,
                    p=0.5,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def process_and_cache_data(df, split, debug=False, load_cached_data=True):
    """
    Reads DICOMs, resizes images, and caches them as numpy arrays.
    Also caches original dimensions for bounding box scaling.
    """
    cache_dir = Config.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    suffix = "_debug" if debug else ""
    images_cache_path = os.path.join(cache_dir, f"{split}_images{suffix}.npy")
    dims_cache_path = os.path.join(
        cache_dir, f"{split}_dims{suffix}.parquet"
    )  # Using parquet for small metadata is cleaner, but npy is fine too. Let's use npy for dims to match prompt preference.
    dims_cache_path = os.path.join(cache_dir, f"{split}_dims{suffix}.npy")

    # 1. Try to load from cache
    if (
        load_cached_data
        and os.path.exists(images_cache_path)
        and os.path.exists(dims_cache_path)
    ):
        print(f"Loading cached {split} data from {cache_dir}...")
        images = np.load(images_cache_path)
        dims = np.load(dims_cache_path)
        return images, dims

    # 2. Process from scratch
    print(f"Processing {split} data (Debug={debug})...")

    image_list = []
    dims_list = []

    for idx, row in df.iterrows():
        file_path = os.path.join(Config.input_dir, row["file_path"])

        try:
            dcm = pydicom.dcmread(file_path)
            pixel_array = dcm.pixel_array.astype(np.float32)

            # Handle Photometric Interpretation
            if (
                hasattr(dcm, "PhotometricInterpretation")
                and dcm.PhotometricInterpretation == "MONOCHROME1"
            ):
                pixel_array = np.amax(pixel_array) - pixel_array

            # Normalize to 0-255
            pixel_array = pixel_array - np.min(pixel_array)
            max_val = np.max(pixel_array)
            if max_val > 0:
                pixel_array = (pixel_array / max_val) * 255.0
            pixel_array = pixel_array.astype(np.uint8)

            # Store original dimensions (Height, Width)
            h, w = pixel_array.shape[:2]
            dims_list.append([h, w])

            # Resize
            if h != Config.img_size or w != Config.img_size:
                pixel_array = cv2.resize(
                    pixel_array,
                    (Config.img_size, Config.img_size),
                    interpolation=cv2.INTER_AREA,
                )

            image_list.append(pixel_array)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Fallback: create black image
            image_list.append(
                np.zeros((Config.img_size, Config.img_size), dtype=np.uint8)
            )
            dims_list.append([Config.img_size, Config.img_size])

    images = np.array(image_list)
    dims = np.array(dims_list)

    # 3. Save to cache
    print(f"Saving {split} data to cache...")
    np.save(images_cache_path, images)
    np.save(dims_cache_path, dims)

    return images, dims


class ChestXRayDataset(Dataset):
    def __init__(self, df, images, dims, transforms=None, mode="train"):
        self.df = df.reset_index(drop=True)
        self.images = images
        self.dims = dims  # (N, 2) -> [Height, Width]
        self.transforms = transforms
        self.mode = mode

        self.study_labels = [
            "Negative for Pneumonia",
            "Typical Appearance",
            "Indeterminate Appearance",
            "Atypical Appearance",
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Image Preparation
        # Expand 1-channel grayscale to 3-channel RGB for ResNet backbone
        img = self.images[idx]
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # 2. Mask Generation (if not test)
        mask = np.zeros((Config.img_size, Config.img_size), dtype=np.float32)

        if self.mode != "test":
            # Get original dimensions for scaling
            orig_h, orig_w = self.dims[idx]
            scale_y = Config.img_size / orig_h
            scale_x = Config.img_size / orig_w

            # Parse boxes
            boxes_str = self.df.loc[idx, "boxes"]
            bboxes = []

            if pd.notna(boxes_str):
                try:
                    # boxes format in csv: [{'x': 10, 'y': 10, 'width': 100, 'height': 100}, ...]
                    box_list = ast.literal_eval(boxes_str)
                    for box in box_list:
                        x = box["x"]
                        y = box["y"]
                        w = box["width"]
                        h = box["height"]

                        # Convert to x1, y1, x2, y2 and scale
                        x1 = x * scale_x
                        y1 = y * scale_y
                        x2 = (x + w) * scale_x
                        y2 = (y + h) * scale_y

                        bboxes.append([x1, y1, x2, y2])
                except:
                    pass

            # Create mask using utility
            if bboxes:
                mask = bbox2mask(bboxes, Config.img_size, Config.img_size).astype(
                    np.float32
                )

        # 3. Augmentations
        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # Ensure mask is channel-first (1, H, W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        # 4. Labels
        if self.mode != "test":
            # Extract one-hot study labels
            labels = self.df.loc[idx, self.study_labels].values.astype(np.float32)
            return img, torch.tensor(labels), mask
        else:
            # For test, return image and ID for submission mapping
            image_id = self.df.loc[idx, "image_id"]
            study_id = self.df.loc[idx, "study_id"]
            return img, image_id, study_id


def get_loaders(debug=False, load_cached_data=True):
    """
    Main function to prepare DataLoaders.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.train_metadata_path)
    val_df = pd.read_csv(Config.val_metadata_path)
    test_df = pd.read_csv(Config.test_metadata_path)

    # 2. Handle Debug Mode
    if debug:
        train_df = train_df.head(Config.debug_sample_size)
        val_df = val_df.head(Config.debug_sample_size)
        test_df = test_df.head(Config.debug_sample_size)
        print(f"DEBUG MODE: Reduced datasets to {len(train_df)} samples.")

    # 3. Process/Cache Images
    train_images, train_dims = process_and_cache_data(
        train_df, "train", debug, load_cached_data
    )
    val_images, val_dims = process_and_cache_data(
        val_df, "val", debug, load_cached_data
    )
    test_images, test_dims = process_and_cache_data(
        test_df, "test", debug, load_cached_data
    )

    # 4. Create Datasets
    train_dataset = ChestXRayDataset(
        train_df,
        train_images,
        train_dims,
        transforms=get_transforms("train"),
        mode="train",
    )

    val_dataset = ChestXRayDataset(
        val_df, val_images, val_dims, transforms=get_transforms("val"), mode="val"
    )

    test_dataset = ChestXRayDataset(
        test_df, test_images, test_dims, transforms=get_transforms("test"), mode="test"
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
