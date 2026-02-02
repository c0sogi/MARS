import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def load_and_cache_data(metadata_path, cache_path, load_cached_data, is_test=False):
    """
    Loads data from metadata/disk, applies preprocessing (Inversion), and caches it.
    Uses a flattened array strategy to avoid pickle in .npz files, ensuring strict
    compliance with requirements while handling variable image sizes.
    """

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            # Reconstruct noisy images from flattened arrays
            noisy_flat = data["noisy_flat"]
            shapes = data["shapes"]
            ids = data["ids"]

            noisy_imgs = []
            start = 0
            for h, w in shapes:
                end = start + h * w
                img = noisy_flat[start:end].reshape(h, w)
                noisy_imgs.append(img)
                start = end

            # Reconstruct clean images if not test
            clean_imgs = []
            if not is_test:
                clean_flat = data["clean_flat"]
                start = 0
                for h, w in shapes:
                    end = start + h * w
                    img = clean_flat[start:end].reshape(h, w)
                    clean_imgs.append(img)
                    start = end
            else:
                clean_imgs = None

            print(f"Loaded {len(ids)} samples from cache: {cache_path}")
            return ids, noisy_imgs, clean_imgs

        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Reloading from source.")

    # 2. Load from Source
    df = pd.read_csv(metadata_path)
    ids = df["id"].astype(str).values
    noisy_paths = df["noisy_image_path"].values
    clean_paths = df["clean_image_path"].values if not is_test else None

    noisy_imgs = []
    clean_imgs = []
    shapes = []

    for i in range(len(df)):
        # Load Noisy Image
        n_path = os.path.join(Config.INPUT_DIR, noisy_paths[i])
        n_img = cv2.imread(n_path, cv2.IMREAD_GRAYSCALE)
        if n_img is None:
            raise ValueError(f"Could not load image: {n_path}")

        # Preprocessing: Normalize and Invert (1.0 - I)
        # Inversion aligns sparse text (black) to 1.0, background (white) to 0.0
        n_img = n_img.astype(np.float32) / 255.0
        n_img = 1.0 - n_img

        noisy_imgs.append(n_img)
        shapes.append(n_img.shape)

        # Load Clean Image (if train/val)
        if not is_test:
            c_path = os.path.join(Config.INPUT_DIR, clean_paths[i])
            c_img = cv2.imread(c_path, cv2.IMREAD_GRAYSCALE)
            if c_img is None:
                raise ValueError(f"Could not load image: {c_path}")

            # Preprocessing: Normalize and Invert
            c_img = c_img.astype(np.float32) / 255.0
            c_img = 1.0 - c_img
            clean_imgs.append(c_img)

    # 3. Save to Cache (Flattened)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    noisy_flat = np.concatenate([img.flatten() for img in noisy_imgs])
    shapes_arr = np.array(shapes)

    save_dict = {"noisy_flat": noisy_flat, "shapes": shapes_arr, "ids": ids}

    if not is_test:
        clean_flat = np.concatenate([img.flatten() for img in clean_imgs])
        save_dict["clean_flat"] = clean_flat

    np.savez(cache_path, **save_dict)
    print(f"Saved {len(ids)} samples to cache: {cache_path}")

    return ids, noisy_imgs, (clean_imgs if not is_test else None)


class DenoisingDataset(Dataset):
    def __init__(self, ids, noisy_imgs, clean_imgs=None, transform=None):
        self.ids = ids
        self.noisy_imgs = noisy_imgs
        self.clean_imgs = clean_imgs
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        noisy = self.noisy_imgs[idx]

        # Albumentations pipeline
        if self.clean_imgs is not None:
            clean = self.clean_imgs[idx]
            if self.transform:
                # Apply consistent geometric transforms to both image and mask
                augmented = self.transform(image=noisy, mask=clean)
                noisy_aug = augmented["image"]
                clean_aug = augmented["mask"]

                # Ensure channel dimension (C, H, W) -> (1, H, W)
                if noisy_aug.ndim == 2:
                    noisy_aug = noisy_aug.unsqueeze(0)
                if clean_aug.ndim == 2:
                    clean_aug = clean_aug.unsqueeze(0)

                return noisy_aug, clean_aug, img_id
            else:
                # Fallback without transforms
                noisy_t = torch.from_numpy(noisy).float().unsqueeze(0)
                clean_t = torch.from_numpy(clean).float().unsqueeze(0)
                return noisy_t, clean_t, img_id
        else:
            # Test mode (no ground truth)
            if self.transform:
                augmented = self.transform(image=noisy)
                noisy_aug = augmented["image"]

                if noisy_aug.ndim == 2:
                    noisy_aug = noisy_aug.unsqueeze(0)

                # Return dummy target for consistency
                return noisy_aug, torch.zeros_like(noisy_aug), img_id
            else:
                noisy_t = torch.from_numpy(noisy).float().unsqueeze(0)
                return noisy_t, torch.zeros_like(noisy_t), img_id


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline.

    Train: Random Crops and Flips/Rotations.
    Val/Test: Reflection Padding to multiples of 16 (required for U-Net).
    """
    if mode == "train":
        return A.Compose(
            [
                A.PadIfNeeded(
                    min_height=Config.IMG_SIZE,
                    min_width=Config.IMG_SIZE,
                    pad_height_divisor=None,
                    pad_width_divisor=None,
                    border_mode=cv2.BORDER_REFLECT_101,
                ),
                A.RandomCrop(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        # Reflection padding extends the signal statistics, avoiding edge artifacts
        return A.Compose(
            [
                A.PadIfNeeded(
                    min_height=None,
                    min_width=None,
                    pad_height_divisor=16,
                    pad_width_divisor=16,
                    border_mode=cv2.BORDER_REFLECT_101,
                ),
                ToTensorV2(),
            ]
        )


def get_train_val_datasets(load_cached_data=True):
    """
    Factory function to create Train and Validation datasets.
    """
    # Train
    train_ids, train_noisy, train_clean = load_and_cache_data(
        Config.TRAIN_METADATA_PATH,
        Config.TRAIN_CACHE_PATH,
        load_cached_data,
        is_test=False,
    )
    train_ds = DenoisingDataset(
        train_ids, train_noisy, train_clean, transform=get_transforms("train")
    )

    # Validation
    val_ids, val_noisy, val_clean = load_and_cache_data(
        Config.VAL_METADATA_PATH, Config.VAL_CACHE_PATH, load_cached_data, is_test=False
    )
    val_ds = DenoisingDataset(
        val_ids, val_noisy, val_clean, transform=get_transforms("val")
    )

    return train_ds, val_ds


def get_test_dataset(load_cached_data=True):
    """
    Factory function to create Test dataset.
    """
    test_ids, test_noisy, _ = load_and_cache_data(
        Config.TEST_METADATA_PATH,
        Config.TEST_CACHE_PATH,
        load_cached_data,
        is_test=True,
    )
    test_ds = DenoisingDataset(
        test_ids, test_noisy, None, transform=get_transforms("test")
    )
    return test_ds
