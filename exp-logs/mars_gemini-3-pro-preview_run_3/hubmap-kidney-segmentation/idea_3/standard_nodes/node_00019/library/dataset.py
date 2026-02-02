import os
import numpy as np
import pandas as pd
import torch
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import from provided libraries
from library.utils import rle_decode, get_logger
from library.data_processing import read_tiff, macenko_normalize, process_dataset

# Initialize logger
logger = get_logger("DatasetModule")


class HubmapDataset(Dataset):
    def __init__(
        self,
        tile_df,
        metadata_df=None,
        transform=None,
        do_normalization=False,
        mode="train",
    ):
        """
        Args:
            tile_df (pd.DataFrame): DataFrame containing tile coordinates and paths.
            metadata_df (pd.DataFrame): DataFrame containing global image metadata (RLE, dimensions).
                                        Required for 'train' and 'val' modes.
            transform (albumentations.Compose): Augmentation pipeline.
            do_normalization (bool): Whether to apply Macenko stain normalization.
            mode (str): 'train', 'val', or 'test'.
        """
        self.tile_df = tile_df
        self.metadata_df = metadata_df
        self.transform = transform
        self.do_normalization = do_normalization
        self.mode = mode

        # Cache for full image masks to avoid repeated RLE decoding
        # Key: image_id, Value: np.ndarray (H, W)
        self.mask_cache = {}

        # Index metadata for faster lookup if provided
        if self.metadata_df is not None:
            self.meta_lookup = self.metadata_df.set_index("id")
        else:
            self.meta_lookup = None

    def __len__(self):
        return len(self.tile_df)

    def _load_mask(self, image_id):
        """
        Loads and caches the full binary mask for a given image ID.
        """
        if image_id in self.mask_cache:
            return self.mask_cache[image_id]

        if self.meta_lookup is None or image_id not in self.meta_lookup.index:
            # Fallback or error if metadata missing in train mode
            logger.warning(f"Metadata not found for {image_id}, returning empty mask.")
            return None

        row = self.meta_lookup.loc[image_id]
        h, w = int(row["height_pixels"]), int(row["width_pixels"])
        rle = row["encoding"]

        # Decode RLE
        if pd.isna(rle):
            mask = np.zeros((h, w), dtype=np.uint8)
        else:
            mask = rle_decode(rle, (h, w))

        self.mask_cache[image_id] = mask
        return mask

    def __getitem__(self, idx):
        row = self.tile_df.iloc[idx]
        image_id = row["id"]
        x, y, w, h = row["x"], row["y"], row["w"], row["h"]

        # 1. Load Image Tile
        # Paths in tile_df are relative to input root, but read_tiff expects full path
        # We assume the paths in tile_df are correct relative to where script is run
        # The process_dataset function in library creates paths like "input/train/..." or similar.
        # We need to ensure we pass the correct path.
        # Based on library code: tile_info["image_path"] = row["image_path"] (which is relative to input/)
        # And process_dataset prepends "./input".
        # Let's check the path. If it starts with "./input", use as is.
        img_path = row["image_path"]
        if not os.path.exists(img_path):
            # Try prepending ./input if not present (safety check)
            alt_path = os.path.join("./input", img_path)
            if os.path.exists(alt_path):
                img_path = alt_path

        window = Window(x, y, w, h)
        image = read_tiff(img_path, window=window)

        if image is None:
            # Return empty tensor if read fails
            image = np.zeros((h, w, 3), dtype=np.uint8)

        # Ensure 3 channels (RGB)
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.shape[2] > 3:
            image = image[:, :, :3]

        # 2. Apply Macenko Normalization
        if self.do_normalization:
            # Macenko expects RGB. read_tiff returns RGB.
            try:
                image = macenko_normalize(image)
            except Exception as e:
                # Fallback if normalization fails (e.g. singular matrix)
                pass

        # 3. Load Mask (if train/val)
        mask = np.zeros((h, w), dtype=np.float32)
        if self.mode in ["train", "val", "validation"]:
            full_mask = self._load_mask(image_id)
            if full_mask is not None:
                # Extract crop
                # Handle boundary conditions if window is padded
                # The window read from tiff handles boundaries automatically?
                # Rasterio window reading might pad or clip.
                # We need to slice the numpy array carefully.

                # Calculate slice indices
                y_start, y_end = y, min(y + h, full_mask.shape[0])
                x_start, x_end = x, min(x + w, full_mask.shape[1])

                # Slice from full mask
                mask_crop = full_mask[y_start:y_end, x_start:x_end]

                # Place into result mask (handles edge cases where tile > image remainder)
                h_crop, w_crop = mask_crop.shape
                mask[0:h_crop, 0:w_crop] = mask_crop

        # 4. Augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            # Convert to tensor manually if no transform
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()

        # Ensure mask has channel dimension if needed, or keep as (H, W) for CrossEntropy
        # Usually for BCE/Dice we want (1, H, W) or (H, W).
        # Albumentations ToTensorV2 returns mask as (H, W) or (H, W, C) depending on input.
        # If mask was (H, W), it returns (H, W).

        return image, mask


def get_transforms(mode="train", img_size=1024):
    """
    Returns Albumentations transforms for train/val/test.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5
                ),
                A.OneOf(
                    [
                        A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                    ],
                    p=0.25,
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=img_size // 20,
                    max_width=img_size // 20,
                    min_holes=5,
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


def prepare_datasets(
    tile_size=1024,
    overlap=0.5,
    do_normalization=True,
    load_cached_data=True,
    debug=False,
):
    """
    Prepares training and validation datasets.

    Args:
        tile_size (int): Size of tiles.
        overlap (float): Overlap ratio.
        do_normalization (bool): Apply Macenko normalization.
        load_cached_data (bool): Use cached parquet files.
        debug (bool): If True, limits dataset size.

    Returns:
        tuple: (train_dataset, val_dataset)
    """
    # Define directories
    input_dir = "./input"
    metadata_dir = "./metadata"
    cache_dir = "./working/idea_3"

    os.makedirs(cache_dir, exist_ok=True)

    # 1. Load Metadata
    train_meta_path = os.path.join(metadata_dir, "train.csv")
    val_meta_path = os.path.join(metadata_dir, "val.csv")

    if not os.path.exists(train_meta_path) or not os.path.exists(val_meta_path):
        raise FileNotFoundError("Metadata files not found in ./metadata/")

    df_train_meta = pd.read_csv(train_meta_path)
    df_val_meta = pd.read_csv(val_meta_path)

    # 2. Process Datasets (Generate/Load Tile Indices)
    # Note: process_dataset handles caching logic for the tile dataframe
    logger.info("Preparing Training Tiles...")
    df_train_tiles = process_dataset(
        metadata_path=train_meta_path,
        output_dir=cache_dir,
        tile_size=tile_size,
        overlap=overlap,
        load_cached_data=load_cached_data,
    )

    logger.info("Preparing Validation Tiles...")
    df_val_tiles = process_dataset(
        metadata_path=val_meta_path,
        output_dir=cache_dir,
        tile_size=tile_size,
        overlap=overlap,  # Usually less overlap for validation, but consistent is fine
        load_cached_data=load_cached_data,
    )

    # Debug mode
    if debug:
        logger.info("Debug mode: Limiting dataset size.")
        df_train_tiles = df_train_tiles.head(50)
        df_val_tiles = df_val_tiles.head(20)

    # 3. Create Datasets
    train_transform = get_transforms(mode="train", img_size=tile_size)
    val_transform = get_transforms(mode="val", img_size=tile_size)

    train_dataset = HubmapDataset(
        tile_df=df_train_tiles,
        metadata_df=df_train_meta,
        transform=train_transform,
        do_normalization=do_normalization,
        mode="train",
    )

    val_dataset = HubmapDataset(
        tile_df=df_val_tiles,
        metadata_df=df_val_meta,
        transform=val_transform,
        do_normalization=do_normalization,
        mode="val",
    )

    return train_dataset, val_dataset


def prepare_test_dataset(
    tile_size=1024, overlap=0.5, do_normalization=True, load_cached_data=True
):
    """
    Prepares the test dataset.
    """
    metadata_dir = "./metadata"
    cache_dir = "./working/idea_3"
    test_meta_path = os.path.join(metadata_dir, "test.csv")

    os.makedirs(cache_dir, exist_ok=True)

    logger.info("Preparing Test Tiles...")
    df_test_tiles = process_dataset(
        metadata_path=test_meta_path,
        output_dir=cache_dir,
        tile_size=tile_size,
        overlap=overlap,
        load_cached_data=load_cached_data,
        tissue_threshold=0.0,  # Be more permissive for test? Or stick to cortex.
    )

    test_transform = get_transforms(mode="test", img_size=tile_size)

    test_dataset = HubmapDataset(
        tile_df=df_test_tiles,
        metadata_df=None,  # No ground truth for test
        transform=test_transform,
        do_normalization=do_normalization,
        mode="test",
    )

    return test_dataset, df_test_tiles
