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
    Crops the black borders from a fundus image to focus on the circular ROI.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # Image is too dark
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


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
            # Return float for regression (MSELoss)
            label = torch.tensor(self.labels[idx], dtype=torch.float)
            return image, label
        else:
            # Dummy label for test set
            return image, torch.tensor(0.0, dtype=torch.float)


def process_dataset(df, input_dir, cache_name, image_size=512, load_cached_data=True):
    """
    Loads images from disk, crops ROI, resizes, and caches the result to numpy arrays.
    """
    cache_dir = "./working/idea_3"
    os.makedirs(cache_dir, exist_ok=True)

    img_cache_path = os.path.join(cache_dir, f"{cache_name}_images_{image_size}.npy")
    lbl_cache_path = os.path.join(cache_dir, f"{cache_name}_labels_{image_size}.npy")

    # Try to load from cache
    if load_cached_data and os.path.exists(img_cache_path):
        print(f"Loading cached {cache_name} data from {cache_dir}...")
        images = np.load(img_cache_path)
        if os.path.exists(lbl_cache_path):
            labels = np.load(lbl_cache_path)
        else:
            labels = None
        return images, labels

    print(f"Processing {cache_name} data (Crop + Resize)...")
    images = []
    labels = []

    has_labels = "diagnosis" in df.columns

    for _, row in df.iterrows():
        # Construct full path
        # Metadata file_path is relative to input_dir (e.g., train_images/xxx.png)
        img_path = os.path.join(input_dir, row["file_path"])

        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                # Create a black image as placeholder to maintain alignment
                img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = crop_image_from_gray(img)
                img = cv2.resize(img, (image_size, image_size))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            img = np.zeros((image_size, image_size, 3), dtype=np.uint8)

        images.append(img)
        if has_labels:
            labels.append(row["diagnosis"])

    images = np.array(images, dtype=np.uint8)

    # Save to cache
    print(f"Saving {cache_name} data to {cache_dir}...")
    np.save(img_cache_path, images)

    if has_labels:
        labels = np.array(labels, dtype=np.float32)
        np.save(lbl_cache_path, labels)
    else:
        labels = None

    return images, labels


def get_dataloaders(
    batch_size=32,
    image_size=512,
    num_workers=4,
    load_cached_data=True,
    input_dir="./input",
    metadata_dir="./metadata",
):
    """
    Main function to create DataLoaders for train, val, and test sets.
    """
    seed_everything(42)

    # Load Metadata
    train_df = pd.read_csv(os.path.join(metadata_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(metadata_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(metadata_dir, "test.csv"))

    # Process Data (Load/Cache)
    train_images, train_labels = process_dataset(
        train_df, input_dir, "train", image_size, load_cached_data
    )
    val_images, val_labels = process_dataset(
        val_df, input_dir, "val", image_size, load_cached_data
    )
    test_images, _ = process_dataset(
        test_df, input_dir, "test", image_size, load_cached_data
    )

    # Define Transforms
    # Train: Geometric Augmentations (Flip/Rotate) + Color + Normalize
    # Cite solution_lesson_node_00002: Augmentation increases task complexity, handled by increased epochs in runfile.
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    # Val/Test: Normalize only
    val_transform = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    # Create Datasets
    train_dataset = RetinopathyDataset(
        train_images, train_labels, transform=train_transform
    )
    val_dataset = RetinopathyDataset(val_images, val_labels, transform=val_transform)
    test_dataset = RetinopathyDataset(test_images, labels=None, transform=val_transform)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
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
