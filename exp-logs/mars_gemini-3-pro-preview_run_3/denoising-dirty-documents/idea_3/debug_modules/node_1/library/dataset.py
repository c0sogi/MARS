import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import normalize_image, get_cached_data


def _generate_patch_data(metadata_path, patch_size, stride, is_train):
    """
    Internal function to compute patches from images listed in metadata.
    Used by get_cached_data.

    Returns:
        np.ndarray:
            - If is_train: Shape (2, N, 1, H, W) containing [inputs, targets]
            - If not is_train: Shape (N, 1, H, W) containing inputs
    """
    df = pd.read_csv(metadata_path)

    inputs_list = []
    targets_list = []

    for _, row in df.iterrows():
        input_path = os.path.join(Config.INPUT_DIR, row["input_path"])

        # Load image in grayscale
        img_in = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img_in is None:
            continue

        img_in = normalize_image(img_in)

        img_tar = None
        if is_train:
            target_path = os.path.join(Config.INPUT_DIR, row["target_path"])
            img_tar = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
            if img_tar is None:
                continue
            img_tar = normalize_image(img_tar)

        h, w = img_in.shape

        # Extract patches using sliding window
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch_in = img_in[y : y + patch_size, x : x + patch_size]
                inputs_list.append(patch_in)

                if is_train and img_tar is not None:
                    patch_tar = img_tar[y : y + patch_size, x : x + patch_size]
                    targets_list.append(patch_tar)

    # Convert to numpy arrays
    # Shape: (N, H, W) -> (N, 1, H, W)
    inputs_array = np.array(inputs_list, dtype=np.float32)
    inputs_array = np.expand_dims(inputs_array, axis=1)

    if is_train:
        targets_array = np.array(targets_list, dtype=np.float32)
        targets_array = np.expand_dims(targets_array, axis=1)
        # Stack to return single object: (2, N, 1, H, W)
        return np.stack([inputs_array, targets_array], axis=0)
    else:
        return inputs_array


class DenoisingPatchDataset(Dataset):
    def __init__(self, data_array, transform=False):
        """
        Args:
            data_array: Numpy array.
                        Shape (2, N, 1, H, W) for train (input, target).
                        Shape (N, 1, H, W) for inference/input only.
            transform: Boolean, whether to apply augmentations.
        """
        self.is_train = False
        # Check if data contains both inputs and targets
        if data_array.ndim == 5 and data_array.shape[0] == 2:
            self.inputs = data_array[0]
            self.targets = data_array[1]
            self.is_train = True
        else:
            self.inputs = data_array
            self.targets = None

        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.inputs[idx])

        if self.is_train:
            y = torch.from_numpy(self.targets[idx])

            if self.transform:
                # Random Horizontal Flip
                if torch.rand(1) < 0.5:
                    x = torch.flip(x, [2])
                    y = torch.flip(y, [2])

                # Random Vertical Flip
                if torch.rand(1) < 0.5:
                    x = torch.flip(x, [1])
                    y = torch.flip(y, [1])

                # Random Rotation (0, 90, 180, 270)
                k = torch.randint(0, 4, (1,)).item()
                if k > 0:
                    x = torch.rot90(x, k, [1, 2])
                    y = torch.rot90(y, k, [1, 2])

            return x, y
        else:
            return x


def get_dataloaders(load_cached_data=True, batch_size=None, debug_limit=None):
    """
    Creates and returns training and validation DataLoaders.

    Args:
        load_cached_data (bool): Whether to attempt loading data from cache.
        batch_size (int, optional): Batch size override. Defaults to Config.BATCH_SIZE.
        debug_limit (int, optional): If provided, limits the dataset size for debugging.

    Returns:
        tuple: (train_loader, val_loader)
    """
    bs = batch_size if batch_size is not None else Config.BATCH_SIZE

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # --- Train Data ---
    train_data = get_cached_data(
        cache_filename="train_patches_packed.npy",
        compute_func=_generate_patch_data,
        load_cached_data=load_cached_data,
        cache_dir=Config.WORKING_DIR,
        metadata_path=Config.TRAIN_METADATA_PATH,
        patch_size=Config.PATCH_SIZE,
        stride=Config.PATCH_STRIDE,
        is_train=True,
    )

    if debug_limit is not None:
        train_data = train_data[:, :debug_limit]

    train_dataset = DenoisingPatchDataset(train_data, transform=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # --- Validation Data ---
    val_data = get_cached_data(
        cache_filename="val_patches_packed.npy",
        compute_func=_generate_patch_data,
        load_cached_data=load_cached_data,
        cache_dir=Config.WORKING_DIR,
        metadata_path=Config.VAL_METADATA_PATH,
        patch_size=Config.PATCH_SIZE,
        stride=Config.PATCH_STRIDE,
        is_train=True,
    )

    if debug_limit is not None:
        val_data = val_data[:, :debug_limit]

    val_dataset = DenoisingPatchDataset(val_data, transform=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader
