import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def load_data(
    metadata_path, path_col, cache_prefix, load_cached_data=True, debug_size=None
):
    """
    Loads images based on metadata CSV. Caches processed arrays to disk as .npy files.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        path_col (str): Column name in CSV containing the relative image paths.
        cache_prefix (str): Prefix for cached filenames.
        load_cached_data (bool): Whether to attempt loading from cache.
        debug_size (int, optional): Number of samples to load for debugging.

    Returns:
        tuple: (ids, data_list) where ids is a numpy array of image IDs and
               data_list is a list of numpy arrays (images).
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    imgs_cache_path = os.path.join(cache_dir, f"{cache_prefix}_imgs.npy")
    shapes_cache_path = os.path.join(cache_dir, f"{cache_prefix}_shapes.npy")
    ids_cache_path = os.path.join(cache_dir, f"{cache_prefix}_ids.npy")

    # 1. Try loading from cache
    if (
        load_cached_data
        and os.path.exists(imgs_cache_path)
        and os.path.exists(shapes_cache_path)
        and os.path.exists(ids_cache_path)
    ):
        try:
            padded_imgs = np.load(imgs_cache_path)
            shapes = np.load(shapes_cache_path)
            ids = np.load(ids_cache_path, allow_pickle=True)

            data_list = []
            for i in range(len(padded_imgs)):
                h, w = shapes[i]
                data_list.append(padded_imgs[i, :h, :w])

            return ids, data_list
        except Exception as e:
            print(
                f"Cache load failed for {cache_prefix}: {e}. Processing from scratch."
            )

    # 2. Process from scratch
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    if debug_size is not None:
        df = df.head(debug_size)

    ids = []
    img_list = []
    max_h, max_w = 0, 0

    for _, row in df.iterrows():
        rel_path = row.get(path_col)

        # Skip if path is missing (e.g., target_path for test set)
        if pd.isna(rel_path) or str(rel_path).strip() == "":
            continue

        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        h, w = img.shape
        max_h = max(max_h, h)
        max_w = max(max_w, w)

        ids.append(row["image_id"])
        img_list.append(img)

    # 3. Save to cache
    count = len(img_list)
    if count > 0:
        padded_storage = np.zeros((count, max_h, max_w), dtype=np.float32)
        shapes_storage = np.zeros((count, 2), dtype=np.int32)

        for i, img in enumerate(img_list):
            h, w = img.shape
            padded_storage[i, :h, :w] = img
            shapes_storage[i] = [h, w]

        np.save(imgs_cache_path, padded_storage)
        np.save(shapes_cache_path, shapes_storage)
        np.save(ids_cache_path, np.array(ids))

    return np.array(ids), img_list


class DenoisingDataset(Dataset):
    def __init__(
        self, inputs, targets=None, patch_size=Config.PATCH_SIZE, train_mode=True
    ):
        """
        Dataset for image denoising.

        Args:
            inputs (list): List of input image arrays (noisy).
            targets (list, optional): List of target image arrays (clean).
            patch_size (int): Size of random crops for training.
            train_mode (bool): If True, applies random cropping and augmentation.
                               If False, returns full images.
        """
        self.inputs = inputs
        self.targets = targets
        self.patch_size = patch_size
        self.train_mode = train_mode

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_img = self.inputs[idx]

        if self.train_mode:
            # Training: Random Crop & Augmentation
            target_img = self.targets[idx]
            h, w = input_img.shape

            # Pad image if it's smaller than the patch size
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)

            if pad_h > 0 or pad_w > 0:
                input_img = np.pad(input_img, ((0, pad_h), (0, pad_w)), mode="reflect")
                target_img = np.pad(
                    target_img, ((0, pad_h), (0, pad_w)), mode="reflect"
                )
                h, w = input_img.shape

            # Random coordinates for crop
            y = np.random.randint(0, h - self.patch_size + 1)
            x = np.random.randint(0, w - self.patch_size + 1)

            patch_in = input_img[y : y + self.patch_size, x : x + self.patch_size]
            patch_tar = target_img[y : y + self.patch_size, x : x + self.patch_size]

            # Random Flips
            if np.random.rand() > 0.5:
                patch_in = np.flipud(patch_in)
                patch_tar = np.flipud(patch_tar)
            if np.random.rand() > 0.5:
                patch_in = np.fliplr(patch_in)
                patch_tar = np.fliplr(patch_tar)

            # Convert to Tensor (C, H, W)
            img_tensor = torch.from_numpy(patch_in.copy()).unsqueeze(0).float()
            target_tensor = torch.from_numpy(patch_tar.copy()).unsqueeze(0).float()

            return img_tensor, target_tensor

        else:
            # Validation/Test: Full Image
            img_tensor = torch.from_numpy(input_img.copy()).unsqueeze(0).float()

            if self.targets is not None:
                target_tensor = (
                    torch.from_numpy(self.targets[idx].copy()).unsqueeze(0).float()
                )
                return img_tensor, target_tensor
            else:
                return img_tensor


def get_dataloaders(
    train_metadata_path=Config.TRAIN_METADATA_PATH,
    val_metadata_path=Config.VAL_METADATA_PATH,
    test_metadata_path=Config.TEST_METADATA_PATH,
    batch_size=Config.BATCH_SIZE,
    patch_size=Config.PATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Returns:
        tuple: (train_loader, val_loader, test_loader, test_ids)
    """
    # --- Load Training Data ---
    _, train_inputs = load_data(
        train_metadata_path,
        "input_path",
        "train_in",
        load_cached_data,
        debug_sample_size,
    )
    _, train_targets = load_data(
        train_metadata_path,
        "target_path",
        "train_target",
        load_cached_data,
        debug_sample_size,
    )

    # --- Load Validation Data ---
    _, val_inputs = load_data(
        val_metadata_path, "input_path", "val_in", load_cached_data, debug_sample_size
    )
    _, val_targets = load_data(
        val_metadata_path,
        "target_path",
        "val_target",
        load_cached_data,
        debug_sample_size,
    )

    # --- Load Test Data ---
    test_ids, test_inputs = load_data(
        test_metadata_path, "input_path", "test_in", load_cached_data, debug_sample_size
    )

    # --- Create Datasets ---
    train_dataset = DenoisingDataset(
        train_inputs, train_targets, patch_size=patch_size, train_mode=True
    )
    val_dataset = DenoisingDataset(val_inputs, val_targets, train_mode=False)
    test_dataset = DenoisingDataset(test_inputs, train_mode=False)

    # --- Create DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Validation and Test use batch_size=1 to handle variable image dimensions
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, test_ids
