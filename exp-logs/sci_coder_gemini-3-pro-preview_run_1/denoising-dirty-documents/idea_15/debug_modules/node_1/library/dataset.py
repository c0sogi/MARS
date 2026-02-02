import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import GlobalConfig
from library.utils import pad_image


class DenoisingDataset(Dataset):
    """
    Dataset class for the Denoising Task.
    Handles loading, normalization, padding, and caching of noisy/clean image pairs.
    Implements differential patch extraction and augmentation for training.
    """

    def __init__(self, mode, stream_config=None, load_cached_data=True):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            stream_config: Configuration object (StreamAConfig or StreamBConfig) containing PATCH_SIZE.
                           Required for 'train' mode.
            load_cached_data (bool): Whether to attempt loading from cache.
        """
        self.mode = mode
        self.stream_config = stream_config

        # Patch size is only relevant for training random crops
        if self.mode == "train":
            if self.stream_config is None:
                raise ValueError("stream_config must be provided for training mode.")
            self.patch_size = self.stream_config.PATCH_SIZE
        else:
            self.patch_size = None

        # Define cache paths
        self.cache_dir = GlobalConfig.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, f"{mode}_cache.npz")

        # Load Data
        self.ids, self.noisy_images, self.clean_images = self._load_data(
            load_cached_data
        )

    def _load_data(self, load_cached_data):
        """
        Loads data from cache if available and requested, otherwise processes from scratch
        and saves to cache. Uses a flattened array approach to avoid pickle for variable-sized images.
        """
        # 1. Try Cache
        if load_cached_data and os.path.exists(self.cache_file):
            try:
                data = np.load(self.cache_file)
                ids = data["ids"]
                shapes = data["shapes"]
                noisy_flat = data["noisy_flat"]

                has_clean = "clean_flat" in data
                clean_flat = data["clean_flat"] if has_clean else None

                noisy_images = []
                clean_images = []

                curr_n = 0
                curr_c = 0

                for h, w in shapes:
                    size = h * w
                    # Reconstruct Noisy
                    img_n = noisy_flat[curr_n : curr_n + size].reshape(h, w)
                    noisy_images.append(img_n)
                    curr_n += size

                    # Reconstruct Clean
                    if has_clean:
                        img_c = clean_flat[curr_c : curr_c + size].reshape(h, w)
                        clean_images.append(img_c)
                        curr_c += size
                    else:
                        clean_images.append(None)

                return ids, noisy_images, clean_images
            except Exception:
                # Fallback to processing if cache is corrupt or incompatible
                pass

        # 2. Process from Scratch
        # Determine Metadata File
        if self.mode == "train":
            csv_path = os.path.join(GlobalConfig.METADATA_DIR, "train.csv")
        elif self.mode == "val":
            csv_path = os.path.join(GlobalConfig.METADATA_DIR, "val.csv")
        elif self.mode == "test":
            csv_path = os.path.join(GlobalConfig.METADATA_DIR, "test.csv")
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        ids = []
        noisy_images = []
        clean_images = []

        # For caching flattened arrays
        noisy_flat_list = []
        clean_flat_list = []
        shapes = []

        for _, row in df.iterrows():
            img_id = str(row["id"])
            noisy_path = os.path.join(GlobalConfig.INPUT_DIR, row["noisy_image_path"])

            # Load Noisy Image (Grayscale)
            n_img = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
            if n_img is None:
                continue

            # Normalize [0, 1]
            n_img = n_img.astype(np.float32) / 255.0

            # Apply Reflection Padding (mod 16)
            n_img, _ = pad_image(n_img, GlobalConfig.PAD_MODULUS)

            ids.append(img_id)
            noisy_images.append(n_img)
            noisy_flat_list.append(n_img.flatten())
            shapes.append(n_img.shape)

            # Load Clean Image (if available)
            if "clean_image_path" in row and pd.notna(row["clean_image_path"]):
                clean_path = os.path.join(
                    GlobalConfig.INPUT_DIR, row["clean_image_path"]
                )
                c_img = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
                if c_img is None:
                    # Fallback safety, though metadata check should prevent this
                    c_img = np.zeros_like(n_img)

                c_img = c_img.astype(np.float32) / 255.0
                c_img, _ = pad_image(c_img, GlobalConfig.PAD_MODULUS)

                clean_images.append(c_img)
                clean_flat_list.append(c_img.flatten())
            else:
                clean_images.append(None)

        # Save to Cache
        save_dict = {
            "ids": np.array(ids),
            "shapes": np.array(shapes),
            "noisy_flat": np.concatenate(noisy_flat_list),
        }

        if clean_flat_list:
            save_dict["clean_flat"] = np.concatenate(clean_flat_list)

        np.savez(self.cache_file, **save_dict)

        return ids, noisy_images, clean_images

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        noisy = self.noisy_images[idx]
        clean = self.clean_images[idx]
        img_id = self.ids[idx]

        # --- Training Mode: Crop & Augment ---
        if self.mode == "train":
            h, w = noisy.shape
            p_size = self.patch_size

            # 1. Handle Padding if image is smaller than patch size
            pad_h = max(0, p_size - h)
            pad_w = max(0, p_size - w)

            if pad_h > 0 or pad_w > 0:
                noisy = np.pad(noisy, ((0, pad_h), (0, pad_w)), mode="reflect")
                if clean is not None:
                    clean = np.pad(clean, ((0, pad_h), (0, pad_w)), mode="reflect")
                h, w = noisy.shape

            # 2. Random Crop
            top = np.random.randint(0, h - p_size + 1)
            left = np.random.randint(0, w - p_size + 1)

            noisy = noisy[top : top + p_size, left : left + p_size]
            if clean is not None:
                clean = clean[top : top + p_size, left : left + p_size]

            # 3. Geometric Augmentations
            # Random Rotate 90 deg increments
            k = np.random.randint(0, 4)
            if k > 0:
                noisy = np.rot90(noisy, k)
                if clean is not None:
                    clean = np.rot90(clean, k)

            # Random Horizontal Flip
            if np.random.rand() > 0.5:
                noisy = np.fliplr(noisy)
                if clean is not None:
                    clean = np.fliplr(clean)

            # Random Vertical Flip
            if np.random.rand() > 0.5:
                noisy = np.flipud(noisy)
                if clean is not None:
                    clean = np.flipud(clean)

            # Ensure memory continuity after numpy strides
            noisy = np.ascontiguousarray(noisy)
            if clean is not None:
                clean = np.ascontiguousarray(clean)

        # --- Convert to Tensor ---
        # Add channel dimension (C, H, W) -> (1, H, W)
        noisy_t = torch.from_numpy(noisy).float().unsqueeze(0)

        if clean is not None:
            clean_t = torch.from_numpy(clean).float().unsqueeze(0)
            return noisy_t, clean_t, img_id
        else:
            # For Test set
            return noisy_t, img_id


def get_dataloader(
    mode,
    stream_config=None,
    batch_size=GlobalConfig.BATCH_SIZE,
    num_workers=GlobalConfig.NUM_WORKERS,
    shuffle=True,
):
    """
    Factory function to create DataLoaders.
    Automatically handles batch_size=1 for validation/test due to variable image sizes.
    """
    dataset = DenoisingDataset(mode, stream_config=stream_config)

    # Force batch_size=1 for val/test because images have variable sizes (even after mod 16 padding)
    # and we don't crop them.
    current_batch_size = batch_size if mode == "train" else 1

    loader = DataLoader(
        dataset,
        batch_size=current_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=GlobalConfig.PIN_MEMORY,
    )

    return loader
