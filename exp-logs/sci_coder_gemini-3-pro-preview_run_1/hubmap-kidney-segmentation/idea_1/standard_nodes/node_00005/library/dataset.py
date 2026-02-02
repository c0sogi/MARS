import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.windows import Window
from library.config import Config
from library.utils import rle_decode


class HuBMAPDataset(Dataset):
    def __init__(
        self,
        metadata_df,
        mode="train",
        transform=None,
        load_cached_data=True,
        overlap=None,
    ):
        """
        Args:
            metadata_df (pd.DataFrame): Metadata containing image paths and RLEs.
            mode (str): 'train', 'val', or 'test'.
            transform (albumentations.Compose): Augmentation pipeline.
            load_cached_data (bool): Whether to load cached tile coordinates/masks.
            overlap (float): Overlap fraction for tiling.
        """
        self.metadata_df = metadata_df
        self.mode = mode
        self.transform = transform
        self.tile_size = Config.TILE_SIZE
        # Use overlap for inference/val to smooth predictions, no overlap for training tiles
        if overlap is not None:
            self.overlap = overlap
        else:
            self.overlap = Config.INFERENCE_OVERLAP if mode != "train" else 0.0

        # Setup cache directories
        self.cache_dir = os.path.join(Config.WORKING_DIR, "cache")
        self.mask_dir = os.path.join(self.cache_dir, "masks")
        os.makedirs(self.mask_dir, exist_ok=True)

        # Define cache file path for coordinates
        self.coords_cache_path = os.path.join(self.cache_dir, f"coords_{mode}.parquet")

        # Generate or load the list of tile coordinates
        self.samples = self._prepare_data(load_cached_data)

    def _prepare_data(self, load_cached_data):
        """
        Generates or loads a dataframe of tile coordinates (id, x, y).
        Implements caching and sampling logic.
        """
        # 1. Try loading cached coordinates
        if load_cached_data and os.path.exists(self.coords_cache_path):
            # Basic integrity check: if training, ensure mask files exist
            if self.mode in ["train", "val"] and not len(os.listdir(self.mask_dir)) > 0:
                print("Cache found but mask files missing. Regenerating...")
            else:
                print(f"Loading cached dataset for mode: {self.mode}")
                return pd.read_parquet(self.coords_cache_path)

        print(f"Generating dataset for mode: {self.mode}...")

        all_samples = []

        # Iterate through each image in metadata
        # If DEBUG is on, we might want to limit this, but usually handled by passing smaller df
        for idx, row in self.metadata_df.iterrows():
            img_id = row["id"]
            img_path = row["image_path"]

            # Read image dimensions
            try:
                with rasterio.open(img_path) as src:
                    H, W = src.height, src.width
            except Exception as e:
                print(f"Error opening image {img_path}: {e}")
                continue

            # Handle Masks for Train/Val
            # We cache full binary masks as .npy files to allow fast random access later
            mask = None
            if self.mode in ["train", "val"]:
                mask_path = os.path.join(self.mask_dir, f"{img_id}.npy")

                if load_cached_data and os.path.exists(mask_path):
                    # Load mask into memory for sampling logic (determining positives)
                    # mmap_mode='r' keeps it on disk but allows slicing
                    mask = np.load(mask_path, mmap_mode="r")
                else:
                    # Decode RLE and save to disk
                    rle = row["encoding"] if "encoding" in row else None
                    decoded_mask = rle_decode(rle, (H, W))
                    np.save(mask_path, decoded_mask)
                    mask = np.load(mask_path, mmap_mode="r")  # Reload as mmap

            # Generate Grid Coordinates
            stride = int(self.tile_size * (1 - self.overlap))

            # Calculate X and Y starting points
            x_points = list(range(0, W - self.tile_size + 1, stride))
            if (W - self.tile_size) % stride != 0:
                x_points.append(max(0, W - self.tile_size))
            if W < self.tile_size:
                x_points = [0]

            y_points = list(range(0, H - self.tile_size + 1, stride))
            if (H - self.tile_size) % stride != 0:
                y_points.append(max(0, H - self.tile_size))
            if H < self.tile_size:
                y_points = [0]

            image_samples = []

            for y in y_points:
                for x in x_points:
                    sample = {"id": img_id, "x": x, "y": y}

                    if self.mode == "train":
                        # Check if tile contains glomeruli
                        # Slicing the memmap is efficient
                        mask_patch = mask[
                            y : y + self.tile_size, x : x + self.tile_size
                        ]
                        is_positive = np.any(mask_patch)
                        sample["is_positive"] = bool(is_positive)

                    image_samples.append(sample)

            # Apply Sampling Strategy for Training
            if self.mode == "train":
                df_samples = pd.DataFrame(image_samples)
                positives = df_samples[df_samples["is_positive"] == True]
                negatives = df_samples[df_samples["is_positive"] == False]

                # Positive Oversampling:
                # 1. Keep all positive tiles
                # 2. Sample equal number of negative tiles
                n_pos = len(positives)
                if n_pos > 0:
                    if len(negatives) > n_pos:
                        negatives = negatives.sample(n=n_pos, random_state=Config.SEED)
                    image_samples = pd.concat([positives, negatives]).to_dict("records")
                else:
                    # Fallback: if no positives, take a small random set of negatives
                    image_samples = negatives.sample(
                        n=min(len(negatives), 20), random_state=Config.SEED
                    ).to_dict("records")

            all_samples.extend(image_samples)

        # Save coordinates to cache
        df_all = pd.DataFrame(all_samples)
        df_all.to_parquet(self.coords_cache_path, index=False)

        return df_all

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Retrieve sample metadata
        row = self.samples.iloc[idx]
        img_id = row["id"]
        x, y = int(row["x"]), int(row["y"])

        # Get image path efficiently
        # We assume metadata_df is small enough that this lookup is negligible
        img_path = self.metadata_df.loc[
            self.metadata_df["id"] == img_id, "image_path"
        ].values[0]

        # 1. Load Image Tile
        # Use rasterio to read only the specific window
        with rasterio.open(img_path) as src:
            # boundless=True handles cases where the tile extends beyond image boundaries (padding with 0)
            window = Window(x, y, self.tile_size, self.tile_size)
            if src.count == 3:
                img = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
            else:
                img = src.read([1], window=window, boundless=True, fill_value=0)
                img = np.repeat(img, 3, axis=0)
            # Convert (C, H, W) -> (H, W, C) for albumentations/cv2
            img = np.moveaxis(img, 0, -1)

        # 2. Load Mask Tile (Train/Val only)
        mask = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
        if self.mode in ["train", "val"]:
            mask_path = os.path.join(self.mask_dir, f"{img_id}.npy")
            if os.path.exists(mask_path):
                # Load using mmap to avoid reading full file into RAM
                # mmap allows us to slice the file on disk
                full_mask = np.load(mask_path, mmap_mode="r")

                # Handle boundary conditions for mask slicing manually since numpy doesn't support 'boundless'
                h_mask, w_mask = full_mask.shape

                x_end = min(x + self.tile_size, w_mask)
                y_end = min(y + self.tile_size, h_mask)

                # Calculate dimensions of the valid data
                valid_h = y_end - y
                valid_w = x_end - x

                if valid_h > 0 and valid_w > 0:
                    mask[:valid_h, :valid_w] = full_mask[y:y_end, x:x_end]

                mask = mask.astype(np.float32)

        # 3. Augmentations
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        else:
            # Default transformations if none provided
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            img = torch.from_numpy(img)
            mask = torch.from_numpy(mask)

        # Ensure mask is (1, H, W) for BCE loss
        if isinstance(mask, torch.Tensor):
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        elif isinstance(mask, np.ndarray):
            if mask.ndim == 2:
                mask = mask[np.newaxis, :, :]

        # Return dict with metadata for reconstruction during inference
        return img, mask, {"id": img_id, "x": x, "y": y}
