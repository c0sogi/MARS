import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the mixup loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_transforms(data_type="train"):
    """
    Returns the augmentation pipeline based on the data type.

    Strategy:
    - Train: Resize, Horizontal Shift (Zero Pad), Brightness/Contrast, Normalize.
    - Val/Test: Resize, Normalize.

    Strictly NO Horizontal Flip.
    """
    if data_type == "train":
        return A.Compose(
            [
                # Resize is handled in preprocessing/loading, but good to ensure consistency
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                # Horizontal Translation via ShiftScaleRotate
                # shift_limit=0.1 (10%), scale=0, rotate=0.
                # border_mode=cv2.BORDER_CONSTANT (Zero Padding)
                A.ShiftScaleRotate(
                    shift_limit_x=Config.SHIFT_LIMIT,
                    shift_limit_y=0.0,
                    scale_limit=0.0,
                    rotate_limit=0.0,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
                # Photometric Augmentations
                A.RandomBrightnessContrast(
                    brightness_limit=Config.BRIGHTNESS_LIMIT,
                    contrast_limit=Config.CONTRAST_LIMIT,
                    p=0.5,
                ),
                # Normalize (ImageNet stats)
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def load_data(mode="train", load_cached_data=True):
    """
    Loads data from metadata, processes images (Resize, 3-Channel), and returns arrays.
    Implements caching using .npy files.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        images (np.ndarray): Shape (N, H, W, 3)
        labels (np.ndarray): Shape (N, Num_Classes)
    """
    # Define Cache Paths
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    images_cache_path = os.path.join(cache_dir, f"{mode}_images.npy")
    labels_cache_path = os.path.join(cache_dir, f"{mode}_labels.npy")

    # 1. Try Loading from Cache
    if load_cached_data:
        if os.path.exists(images_cache_path) and os.path.exists(labels_cache_path):
            print(f"Loading {mode} data from cache...")
            try:
                images = np.load(images_cache_path)
                labels = np.load(labels_cache_path)
                return images, labels
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from Scratch
    print(f"Processing {mode} data from scratch...")

    # Select Metadata File
    if mode == "train":
        meta_path = Config.TRAIN_METADATA
    elif mode == "val":
        meta_path = Config.VAL_METADATA
    elif mode == "test":
        meta_path = Config.TEST_METADATA
    else:
        raise ValueError(f"Invalid mode: {mode}")

    df = pd.read_csv(meta_path)

    image_list = []
    label_list = []

    for idx, row in df.iterrows():
        # Path Resolution
        # row['file_path'] is like 'essential_data/src_wavs/filename.wav'
        # We need 'supplemental_data/spectrograms/filename.bmp'
        wav_filename = os.path.basename(row["file_path"])
        bmp_filename = wav_filename.replace(".wav", ".bmp")
        bmp_path = os.path.join(Config.SPECTROGRAM_DIR, bmp_filename)

        # Load Image
        if not os.path.exists(bmp_path):
            print(f"Warning: File not found {bmp_path}")
            img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            # Read as Grayscale
            img_gray = cv2.imread(bmp_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                print(f"Warning: Failed to read {bmp_path}")
                img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
            else:
                # Resize
                img_resized = cv2.resize(img_gray, (Config.IMG_SIZE, Config.IMG_SIZE))
                # Replicate to 3 Channels (Pseudo-RGB)
                img = np.stack([img_resized] * 3, axis=-1)

        image_list.append(img)

        # Parse Labels
        # Labels are space-separated strings of ints. Test labels are '?'
        label_vec = np.zeros(Config.NUM_CLASSES, dtype=np.float32)
        label_str = str(row["labels"])

        if label_str != "?" and label_str.lower() != "nan":
            try:
                indices = [int(x) for x in label_str.split()]
                label_vec[indices] = 1.0
            except ValueError:
                pass

        label_list.append(label_vec)

    images = np.array(image_list, dtype=np.uint8)
    labels = np.array(label_list, dtype=np.float32)

    # 3. Save to Cache
    np.save(images_cache_path, images)
    np.save(labels_cache_path, labels)
    print(f"Saved {mode} data to cache.")

    return images, labels


class BirdDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, 3).
            labels (np.ndarray, optional): Array of labels (N, Num_Classes).
            transform (albumentations.Compose, optional): Augmentation pipeline.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.labels is not None:
            label = self.labels[idx]
            return image, torch.tensor(label, dtype=torch.float32)
        else:
            return image
