import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.utils import seed_everything


def crop_image_from_gray(img, tol=7):
    """
    Crops the black borders from the fundus image to focus on the retina.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img
    return img


def get_transforms(image_size=768, phase="train"):
    """
    Returns the Albumentations transform pipeline.
    """
    if phase == "train":
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


class RetinopathyDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, C).
            labels (np.ndarray, optional): Array of labels (N,).
            transform (callable, optional): Transform pipeline.
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
            # Regression expects float target
            return image, torch.tensor(label, dtype=torch.float32)
        else:
            # Dummy label for inference
            return image, torch.tensor(0.0, dtype=torch.float32)


def load_and_process_images(df, input_dir, image_size=768):
    """
    Loads images from disk, crops, resizes, and returns as numpy array.
    """
    processed_images = []
    # Ensure deterministic behavior
    seed_everything(42)

    print(f"Processing {len(df)} images...")
    for idx, row in df.iterrows():
        file_path = os.path.join(input_dir, row["file_path"])

        # Read image
        img = cv2.imread(file_path)
        if img is None:
            # Fallback for missing files (should not happen based on metadata check)
            img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Crop black borders
            img = crop_image_from_gray(img)

            # Resize to target size
            img = cv2.resize(img, (image_size, image_size))

        processed_images.append(img)

    return np.array(processed_images, dtype=np.uint8)


def get_cached_dataset(
    metadata_path,
    cache_dir,
    cache_name,
    input_dir="./input",
    image_size=768,
    load_cached_data=True,
):
    """
    Handles caching logic: Load from npy if exists and requested, else process and save.
    """
    os.makedirs(cache_dir, exist_ok=True)

    images_cache_path = os.path.join(cache_dir, f"{cache_name}_images_{image_size}.npy")
    labels_cache_path = os.path.join(cache_dir, f"{cache_name}_labels_{image_size}.npy")

    df = pd.read_csv(metadata_path)

    # Logic:
    # 1. IF load_cached_data is True: Try to load.
    # 2. IF loading fails OR load_cached_data is False: Compute and Save.

    images = None
    labels = None
    loaded = False

    if load_cached_data:
        if os.path.exists(images_cache_path):
            try:
                print(f"Loading cached images from {images_cache_path}...")
                images = np.load(images_cache_path)
                loaded = True

                # Load labels if they exist and are expected
                if "diagnosis" in df.columns:
                    if os.path.exists(labels_cache_path):
                        labels = np.load(labels_cache_path)
                    else:
                        # If labels cache missing but images found, just reload labels from df
                        labels = df["diagnosis"].values
                else:
                    labels = None
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")
                loaded = False
        else:
            print(f"Cache file {images_cache_path} not found.")

    if not loaded:
        print(f"Processing data for {cache_name} from scratch...")
        images = load_and_process_images(df, input_dir, image_size)

        print(f"Saving images to {images_cache_path}...")
        np.save(images_cache_path, images)

        if "diagnosis" in df.columns:
            labels = df["diagnosis"].values
            print(f"Saving labels to {labels_cache_path}...")
            np.save(labels_cache_path, labels)
        else:
            labels = None

    return images, labels


def prepare_datasets(image_size=768, load_cached_data=True):
    """
    Main entry point to get datasets.
    """
    cache_dir = "./working/idea_5"
    input_dir = "./input"

    # Train
    train_imgs, train_lbls = get_cached_dataset(
        "./metadata/train.csv",
        cache_dir,
        "train",
        input_dir,
        image_size,
        load_cached_data,
    )
    train_dataset = RetinopathyDataset(
        train_imgs, train_lbls, transform=get_transforms(image_size, "train")
    )

    # Validation
    val_imgs, val_lbls = get_cached_dataset(
        "./metadata/val.csv", cache_dir, "val", input_dir, image_size, load_cached_data
    )
    val_dataset = RetinopathyDataset(
        val_imgs, val_lbls, transform=get_transforms(image_size, "val")
    )

    # Test
    test_imgs, _ = get_cached_dataset(
        "./metadata/test.csv",
        cache_dir,
        "test",
        input_dir,
        image_size,
        load_cached_data,
    )
    test_dataset = RetinopathyDataset(
        test_imgs, None, transform=get_transforms(image_size, "test")
    )

    return train_dataset, val_dataset, test_dataset
