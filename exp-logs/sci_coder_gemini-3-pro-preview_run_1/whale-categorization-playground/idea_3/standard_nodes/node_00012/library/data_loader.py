import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.configuration import Config


class WhaleDataset(Dataset):
    """
    Custom Dataset for loading Whale images and corresponding labels.
    """

    def __init__(self, df, root_dir, transform=None, class_to_idx=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (Image, Id, file_path).
            root_dir (str): Root directory for images (input directory).
            transform (callable, optional): Albumentations transforms.
            class_to_idx (dict, optional): Mapping from class ID string to integer index.
            is_test (bool): Flag to indicate if this is the test set (no labels).
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # metadata file_path is relative to input_dir (e.g., "train/image.jpg")
        img_path = os.path.join(self.root_dir, row["file_path"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing/corrupt images: create a black image
            # This prevents crashing during training
            image = np.zeros((Config.img_size, Config.img_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Minimal transform if none provided
            base_transform = A.Compose(
                [
                    A.Resize(Config.img_size, Config.img_size),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
            image = base_transform(image=image)["image"]

        # Return image only for test set
        if self.is_test:
            return image, row["Image"]  # Return filename for submission

        # Return image and label for train/val
        label_str = row["Id"]
        label = self.class_to_idx.get(label_str, -1)

        # Safety check for unknown labels (should not happen with correct splitting)
        if label == -1:
            # Map to new_whale (class 0) if unknown
            label = 0

        return image, torch.tensor(label, dtype=torch.long)


def get_transforms(phase):
    """
    Returns Albumentations transforms based on the phase (train/val/test).

    Strategy:
    - Resize to 320x320.
    - Train: HorizontalFlip, Affine (Rotate, Shear, Scale), RandomBrightnessContrast.
    - Exclude Hue/Saturation jitter.
    - Normalize using ImageNet defaults.
    """
    if phase == "train":
        return A.Compose(
            [
                A.Resize(height=Config.img_size, width=Config.img_size),
                A.HorizontalFlip(p=0.5),
                # Affine transformations: Rotation, Shear, Scaling
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=None,
                    rotate=(-15, 15),
                    shear=(-5, 5),
                    p=0.5,
                ),
                # Photometric distortions (excluding color jitter/hue)
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Resize(height=Config.img_size, width=Config.img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def process_classes(df_train, load_cached_data=True):
    """
    Determines the class mapping. Caches the class list to ensure consistency.

    Logic:
    1. If load_cached_data is True, try to load classes.npy.
    2. If not found or load_cached_data is False, compute classes from df_train.
       - Ensure 'new_whale' is class 0.
       - Sort remaining classes alphabetically.
    3. Save to cache if computed.
    """
    cache_path = os.path.join(Config.working_dir, "classes.npy")

    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    classes = None

    if load_cached_data and os.path.exists(cache_path):
        try:
            classes = np.load(cache_path)
            # print(f"Loaded {len(classes)} classes from cache.")
        except Exception:
            classes = None

    if classes is None:
        # Compute classes
        unique_ids = set(df_train["Id"].unique())

        # Remove new_whale to handle it specifically
        if "new_whale" in unique_ids:
            unique_ids.remove("new_whale")

        # Sort known whales
        sorted_ids = sorted(list(unique_ids))

        # Prepend new_whale to be index 0
        classes = np.array(["new_whale"] + sorted_ids)

        # Save to cache
        np.save(cache_path, classes)
        # print(f"Computed and saved {len(classes)} classes to cache.")

    return classes


def get_dataloaders(load_cached_data=True):
    """
    Main function to prepare DataLoaders.

    Args:
        load_cached_data (bool): Whether to use cached class encoding.

    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    # 1. Load Metadata
    df_train = pd.read_csv(Config.train_csv)
    df_val = pd.read_csv(Config.val_csv)
    df_test = pd.read_csv(Config.test_csv)

    # 2. Debug Mode: Subset data
    if Config.debug:
        df_train = df_train.head(100)
        df_val = df_val.head(50)
        df_test = df_test.head(50)

    # 3. Process Classes (Label Encoding)
    classes = process_classes(df_train, load_cached_data=load_cached_data)
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    num_classes = len(classes)

    # Update Config with actual number of classes
    # (Though Config is a class, we can update the instance or the class attribute if needed by the caller)
    # The caller typically uses the returned num_classes.

    # 4. Create Datasets
    train_dataset = WhaleDataset(
        df_train,
        Config.input_dir,
        transform=get_transforms("train"),
        class_to_idx=class_to_idx,
        is_test=False,
    )

    val_dataset = WhaleDataset(
        df_val,
        Config.input_dir,
        transform=get_transforms("val"),
        class_to_idx=class_to_idx,
        is_test=False,
    )

    test_dataset = WhaleDataset(
        df_test,
        Config.input_dir,
        transform=get_transforms("test"),
        class_to_idx=None,
        is_test=True,
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, num_classes
