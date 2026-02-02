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
    Crops the image to the region of interest (fundus) by removing black borders.
    Detects the circular mask of the eye and crops the bounding box around it.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            # Image is too dark or empty, return original
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def get_transforms(phase, size=768):
    """
    Returns Albumentations transforms for the specified phase.
    Implements geometric-only augmentations for training to preserve color signals.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


class RetinopathyDataset(Dataset):
    """
    Dataset class for Diabetic Retinopathy classification.
    Operates on pre-loaded numpy arrays for maximum throughput.
    """

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
            # Return float label for regression (MSELoss)
            return image, torch.tensor(label, dtype=torch.float32)
        else:
            return image


def process_images(df, input_dir, size=768):
    """
    Loads, crops, and resizes images from the dataframe.
    """
    images = []
    for _, row in df.iterrows():
        # Metadata paths are relative to input_dir
        path = os.path.join(input_dir, row["file_path"])
        try:
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Image not found at {path}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = crop_image_from_gray(img)
            img = cv2.resize(img, (size, size))
            images.append(img)
        except Exception as e:
            print(
                f"Warning: Could not process image {path}. Using black image. Error: {e}"
            )
            images.append(np.zeros((size, size, 3), dtype=np.uint8))

    return np.array(images)


def get_datasets(
    input_dir,
    metadata_dir,
    cache_dir="./working/idea_6",
    img_size=768,
    load_cached_data=True,
):
    """
    Prepares and returns training, validation, and test datasets.
    Implements caching mechanism using .npy files to speed up subsequent runs.
    """
    os.makedirs(cache_dir, exist_ok=True)

    phases = ["train", "val", "test"]
    datasets = {}

    for phase in phases:
        cache_img_path = os.path.join(cache_dir, f"{phase}_images_{img_size}.npy")
        cache_lbl_path = os.path.join(cache_dir, f"{phase}_labels.npy")

        # Determine if we need to process data
        data_exists = os.path.exists(cache_img_path)
        if phase != "test":
            data_exists = data_exists and os.path.exists(cache_lbl_path)

        if load_cached_data and data_exists:
            print(f"Loading cached {phase} data from {cache_dir}...")
            images = np.load(cache_img_path)
            if phase != "test":
                labels = np.load(cache_lbl_path)
            else:
                labels = None
        else:
            print(f"Processing {phase} data from scratch...")
            meta_path = os.path.join(metadata_dir, f"{phase}.csv")
            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"Metadata file not found: {meta_path}")

            df = pd.read_csv(meta_path)

            # Process images
            images = process_images(df, input_dir, size=img_size)
            np.save(cache_img_path, images)

            # Process labels
            if phase != "test":
                labels = df["diagnosis"].values.astype(float)
                np.save(cache_lbl_path, labels)
            else:
                labels = None

        transform = get_transforms(phase, size=img_size)
        datasets[phase] = RetinopathyDataset(images, labels, transform)

    return datasets["train"], datasets["val"], datasets["test"]
