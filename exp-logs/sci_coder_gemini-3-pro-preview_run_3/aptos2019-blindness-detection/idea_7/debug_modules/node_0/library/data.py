import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config
from library.utils import seed_everything


def crop_image_from_gray(img, tol=7):
    """
    Crops the black borders from a fundus image to isolate the circular ROI.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img  # Return original if crop fails (image too dark)
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def get_transforms(phase="train"):
    """
    Returns Albumentations transforms for the specified phase.
    Note: Input images are already resized to the target resolution by the caching step.
    """
    # specific stats from data analysis
    mean = (0.4018, 0.2190, 0.0852)
    std = (0.2653, 0.1452, 0.0837)

    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def process_and_cache_data(df, img_size, set_name, load_cached_data=True):
    """
    Loads images, crops, resizes, and caches them as .npy files.

    Args:
        df: DataFrame containing metadata.
        img_size: Target image size (e.g., 512, 1024).
        set_name: Name of the dataset split ('train', 'val', 'test').
        load_cached_data: Whether to attempt loading from cache.

    Returns:
        images: Numpy array of shape (N, img_size, img_size, 3)
        labels: Numpy array of labels (if 'diagnosis' in df) or None
    """
    # Ensure working directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    cache_path_imgs = os.path.join(
        Config.CACHE_DIR, f"{set_name}_images_{img_size}.npy"
    )
    cache_path_labels = os.path.join(
        Config.CACHE_DIR, f"{set_name}_labels_{img_size}.npy"
    )

    # Try to load from cache
    if load_cached_data and os.path.exists(cache_path_imgs):
        print(f"Loading {set_name} data (size={img_size}) from cache...")
        images = np.load(cache_path_imgs)
        if os.path.exists(cache_path_labels):
            labels = np.load(cache_path_labels)
        else:
            labels = df["diagnosis"].values if "diagnosis" in df.columns else None
        return images, labels

    print(f"Processing {set_name} data (size={img_size})...")

    img_list = []

    for _, row in df.iterrows():
        # Construct full path. Metadata contains relative path e.g., "train_images/xxx.png"
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        try:
            img = cv2.imread(full_path)
            if img is None:
                # Placeholder for missing images (black image)
                img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = crop_image_from_gray(img)
                img = cv2.resize(img, (img_size, img_size))
        except Exception as e:
            print(f"Warning: Error processing {full_path}: {e}")
            img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        img_list.append(img)

    images = np.array(img_list, dtype=np.uint8)

    # Save to cache
    np.save(cache_path_imgs, images)

    labels = None
    if "diagnosis" in df.columns:
        labels = df["diagnosis"].values
        np.save(cache_path_labels, labels)

    print(f"Saved {set_name} data to cache at {cache_path_imgs}")
    return images, labels


class RetinaDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
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
            # Return float label for regression (MSE Loss)
            label = torch.tensor(self.labels[idx], dtype=torch.float)
            return image, label

        return image


def get_dataloaders(img_size, batch_size, load_cached_data=True, debug=False):
    """
    Prepares DataLoaders for training and validation.
    """
    seed_everything(Config.SEED)

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    if debug:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Process Data
    train_imgs, train_labels = process_and_cache_data(
        train_df, img_size, "train", load_cached_data
    )
    val_imgs, val_labels = process_and_cache_data(
        val_df, img_size, "val", load_cached_data
    )

    # Create Datasets
    train_dataset = RetinaDataset(
        train_imgs, train_labels, transform=get_transforms(phase="train")
    )

    val_dataset = RetinaDataset(
        val_imgs, val_labels, transform=get_transforms(phase="val")
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(img_size, batch_size, load_cached_data=True, debug=False):
    """
    Prepares DataLoader for testing/inference.
    """
    seed_everything(Config.SEED)

    test_df = pd.read_csv(Config.TEST_CSV)

    if debug:
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    test_imgs, _ = process_and_cache_data(test_df, img_size, "test", load_cached_data)

    test_dataset = RetinaDataset(
        test_imgs, labels=None, transform=get_transforms(phase="test")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader, test_df
