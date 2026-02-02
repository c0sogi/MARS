import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import pad_image


class SaltDataset(Dataset):
    def __init__(
        self,
        images,
        depths,
        ids,
        masks=None,
        depth_min=0.0,
        depth_max=1.0,
        mode="train",
    ):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W) in uint8.
            depths (np.ndarray): Array of depths (N,) in int/float.
            ids (np.ndarray): Array of image IDs (N,).
            masks (np.ndarray, optional): Array of masks (N, H, W) in uint8 (0 or 1).
            depth_min (float): Minimum depth value for normalization.
            depth_max (float): Maximum depth value for normalization.
            mode (str): 'train', 'val', or 'test'.
        """
        self.images = images
        self.depths = depths
        self.ids = ids
        self.masks = masks
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.mode = mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Load Image
        # Input images are uint8 (0-255), convert to float [0, 1]
        img = self.images[idx].astype(np.float32) / 255.0

        # 2. Load and Normalize Depth
        # Normalize depth to [0, 1] using training set stats
        z = self.depths[idx]
        z_norm = (z - self.depth_min) / (self.depth_max - self.depth_min + 1e-8)

        # 3. Construct 2-Channel Input
        # Channel 0: Seismic Image
        # Channel 1: Dense Depth Channel
        h, w = img.shape
        input_tensor = np.zeros((2, h, w), dtype=np.float32)
        input_tensor[0, :, :] = img
        input_tensor[1, :, :] = z_norm

        input_tensor = torch.from_numpy(input_tensor)

        # 4. Handle Mask and Augmentation
        if self.mode in ["train", "val"]:
            # Load mask (0 or 1)
            mask = self.masks[idx].astype(np.float32)
            mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)

            # Augmentation: Horizontal Flip (Train only)
            if self.mode == "train" and torch.rand(1).item() > 0.5:
                # Flip along width (dim 2)
                input_tensor = torch.flip(input_tensor, [2])
                mask = torch.flip(mask, [2])

            return input_tensor, mask

        else:
            # Test mode: Return input and ID for submission
            return input_tensor, self.ids[idx]


def preprocess_data(df, input_dir, mode="train"):
    """
    Loads images/masks from disk, pads them, and returns numpy arrays.
    """
    images = []
    masks = []
    depths = []
    ids = []

    for _, row in df.iterrows():
        img_id = row["id"]
        z = row["z"]

        # Construct full paths
        img_path = os.path.join(input_dir, row["image_path"])

        # Load Image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Should not happen given metadata validation, but safety check
            continue

        # Pad Image
        img_padded = pad_image(img)
        images.append(img_padded)
        depths.append(z)
        ids.append(img_id)

        if mode in ["train", "val"]:
            mask_path = os.path.join(input_dir, row["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Normalize mask to 0/1 immediately
            mask = (mask > 127).astype(np.uint8)

            # Pad Mask
            mask_padded = pad_image(mask)
            masks.append(mask_padded)

    images = np.array(images, dtype=np.uint8)
    depths = np.array(depths, dtype=np.float32)
    ids = np.array(ids)

    if mode in ["train", "val"]:
        masks = np.array(masks, dtype=np.uint8)
        return images, masks, depths, ids
    else:
        return images, None, depths, ids


def get_dataloaders(
    train_csv=Config.TRAIN_CSV,
    val_csv=Config.VAL_CSV,
    test_csv=Config.TEST_CSV,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    debug=Config.DEBUG,
):
    """
    Prepares DataLoaders for train, val, and test splits.
    Handles caching of processed numpy arrays.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache filenames
    cache_files = {
        "train": [
            "train_images.npy",
            "train_masks.npy",
            "train_depths.npy",
            "train_ids.npy",
        ],
        "val": ["val_images.npy", "val_masks.npy", "val_depths.npy", "val_ids.npy"],
        "test": ["test_images.npy", "test_depths.npy", "test_ids.npy"],
    }

    data = {}

    # Helper to check if cache exists
    def check_cache(mode):
        return all(
            os.path.exists(os.path.join(cache_dir, f)) for f in cache_files[mode]
        )

    # 1. Load Data (Cache or Process)
    splits = [("train", train_csv), ("val", val_csv), ("test", test_csv)]

    for mode, csv_path in splits:
        is_cached = check_cache(mode)

        if load_cached_data and is_cached:
            # Load from cache
            print(f"Loading {mode} data from cache...")
            data[f"{mode}_images"] = np.load(
                os.path.join(cache_dir, f"{mode}_images.npy")
            )
            data[f"{mode}_depths"] = np.load(
                os.path.join(cache_dir, f"{mode}_depths.npy")
            )
            data[f"{mode}_ids"] = np.load(os.path.join(cache_dir, f"{mode}_ids.npy"))
            if mode != "test":
                data[f"{mode}_masks"] = np.load(
                    os.path.join(cache_dir, f"{mode}_masks.npy")
                )
            else:
                data[f"{mode}_masks"] = None
        else:
            # Process from scratch
            print(f"Processing {mode} data from source...")
            df = pd.read_csv(csv_path)

            # Slice for debug BEFORE processing to save time
            if debug:
                df = df.head(Config.DEBUG_SAMPLE_SIZE)

            images, masks, depths, ids = preprocess_data(
                df, Config.INPUT_DIR, mode=mode
            )

            # Save to cache
            np.save(os.path.join(cache_dir, f"{mode}_images.npy"), images)
            np.save(os.path.join(cache_dir, f"{mode}_depths.npy"), depths)
            np.save(os.path.join(cache_dir, f"{mode}_ids.npy"), ids)
            if masks is not None:
                np.save(os.path.join(cache_dir, f"{mode}_masks.npy"), masks)

            data[f"{mode}_images"] = images
            data[f"{mode}_masks"] = masks
            data[f"{mode}_depths"] = depths
            data[f"{mode}_ids"] = ids

    # 2. Handle Debug Slicing (if loaded from cache)
    if debug and load_cached_data:
        for key in data:
            if data[key] is not None:
                data[key] = data[key][: Config.DEBUG_SAMPLE_SIZE]

    # 3. Calculate Depth Statistics from Training Data
    # We use training stats to normalize all sets to prevent leakage
    depth_min = data["train_depths"].min()
    depth_max = data["train_depths"].max()

    print(f"Depth Statistics (Train): Min={depth_min:.4f}, Max={depth_max:.4f}")

    # 4. Create Datasets
    train_dataset = SaltDataset(
        data["train_images"],
        data["train_depths"],
        data["train_ids"],
        data["train_masks"],
        depth_min=depth_min,
        depth_max=depth_max,
        mode="train",
    )

    val_dataset = SaltDataset(
        data["val_images"],
        data["val_depths"],
        data["val_ids"],
        data["val_masks"],
        depth_min=depth_min,
        depth_max=depth_max,
        mode="val",
    )

    test_dataset = SaltDataset(
        data["test_images"],
        data["test_depths"],
        data["test_ids"],
        masks=None,
        depth_min=depth_min,
        depth_max=depth_max,
        mode="test",
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
