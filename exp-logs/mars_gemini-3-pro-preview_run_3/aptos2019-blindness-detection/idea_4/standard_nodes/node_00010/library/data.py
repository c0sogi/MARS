import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.utils import seed_everything


def crop_image_from_gray(img, tol=7):
    """
    Crops the black borders from a fundus image.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # image is too dark so that we crop out everything
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img
    return img


def process_and_cache_images(
    df, image_size, cache_dir, load_cached_data, set_name, input_dir="./input"
):
    """
    Loads, crops, resizes, and caches images as .npy files.
    Strictly follows the caching logic:
    1. If load_cached_data is True, try to load.
    2. If fails or False, compute and save.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{set_name}_images_{image_size}.npy")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached images from {cache_path}...")
        try:
            images = np.load(cache_path)
            if len(images) == len(df):
                return images
            else:
                print("Cached data size mismatch. Recomputing...")
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(
        f"Processing {set_name} images (Crop + Resize to {image_size}x{image_size})..."
    )
    images = []

    # Ensure we use a fixed order based on the dataframe
    for idx, row in df.iterrows():
        # Metadata contains relative paths, e.g., "train_images/id.png"
        file_path = os.path.join(input_dir, row["file_path"])

        try:
            img = cv2.imread(file_path)
            if img is None:
                # Fallback for missing/corrupt images: black image
                print(f"Warning: Could not read image {file_path}. Using black image.")
                img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = crop_image_from_gray(img)
                img = cv2.resize(img, (image_size, image_size))
        except Exception as e:
            print(f"Error processing {file_path}: {e}. Using black image.")
            img = np.zeros((image_size, image_size, 3), dtype=np.uint8)

        images.append(img)

    images = np.array(images, dtype=np.uint8)

    print(f"Saving processed images to {cache_path}...")
    np.save(cache_path, images)

    return images


class RetinopathyDataset(Dataset):
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
            label = self.labels[idx]
            # Return float label for regression, or long for classification
            # The task description suggests regression is better, but PyTorch datasets
            # usually return the raw type. We'll return float for MSELoss compatibility
            # or let the model cast it. Given the regression task, float is safer.
            return image, torch.tensor(label, dtype=torch.float32)
        else:
            return image


def get_transforms(image_size, phase="train"):
    """
    Returns Albumentations transforms for train or validation/test phases.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Note: Resize is handled during caching, but we can add it here for safety
                # or if we were loading raw images. Since we cache resized images,
                # we skip resize here to avoid double interpolation artifacts.
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def get_dataloaders(
    image_size: int = 512,
    batch_size: int = 32,
    num_workers: int = 4,
    load_cached_data: bool = True,
    base_path: str = "./input",
    metadata_dir: str = "./metadata",
    cache_dir: str = "./working/idea_4",
):
    """
    Creates DataLoaders for train, validation, and test sets.
    Handles caching of pre-processed images.
    """
    seed_everything(42)

    # Load Metadata
    train_csv = os.path.join(metadata_dir, "train.csv")
    val_csv = os.path.join(metadata_dir, "val.csv")
    test_csv = os.path.join(metadata_dir, "test.csv")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # Process Images
    train_images = process_and_cache_images(
        train_df, image_size, cache_dir, load_cached_data, "train", base_path
    )
    val_images = process_and_cache_images(
        val_df, image_size, cache_dir, load_cached_data, "val", base_path
    )
    test_images = process_and_cache_images(
        test_df, image_size, cache_dir, load_cached_data, "test", base_path
    )

    # Extract Labels
    train_labels = train_df["diagnosis"].values
    val_labels = val_df["diagnosis"].values
    # Test labels don't exist

    # Create Datasets
    train_dataset = RetinopathyDataset(
        train_images, train_labels, transform=get_transforms(image_size, phase="train")
    )

    val_dataset = RetinopathyDataset(
        val_images, val_labels, transform=get_transforms(image_size, phase="valid")
    )

    test_dataset = RetinopathyDataset(
        test_images, labels=None, transform=get_transforms(image_size, phase="test")
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
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
