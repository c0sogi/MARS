import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


class RetinaDataset(Dataset):
    """
    Custom Dataset for Diabetic Retinopathy detection.
    Handles loading images from disk, resizing, caching to RAM/Disk, and applying augmentations.
    """

    def __init__(self, df, phase="train", transform=None, load_cached_data=True):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata (id_code, file_path, diagnosis).
            phase (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transforms.
            load_cached_data (bool): Whether to attempt loading from numpy cache.
        """
        self.df = df
        self.phase = phase
        self.transform = transform
        self.input_dir = Config.input_dir

        # Ensure working directory exists for caching
        os.makedirs(Config.working_dir, exist_ok=True)

        # Define cache filename based on phase and debug status
        debug_suffix = "_debug" if Config.debug else ""
        cache_filename = f"cached_images_{phase}{debug_suffix}.npy"
        cache_path = os.path.join(Config.working_dir, cache_filename)

        self.images = None

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                # print(f"Loading cached images from {cache_path}...")
                self.images = np.load(cache_path)
            except Exception:
                # If loading fails, fall back to processing
                self.images = None

        # 2. If not loaded, process from scratch
        if self.images is None:
            # print(f"Processing images for {phase} (Cache miss or force reload)...")
            img_list = []

            # Iterate through dataframe
            for _, row in self.df.iterrows():
                # Construct full path
                full_path = os.path.join(self.input_dir, row["file_path"])

                # Read image
                img = cv2.imread(full_path)

                if img is None:
                    # Handle missing images by creating a black image (should not happen based on metadata check)
                    img = np.zeros(
                        (Config.image_size, Config.image_size, 3), dtype=np.uint8
                    )
                else:
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Resize ("Squash") to target size
                    img = cv2.resize(img, (Config.image_size, Config.image_size))

                img_list.append(img)

            # Convert to numpy array
            self.images = np.array(img_list, dtype=np.uint8)

            # Save to cache
            np.save(cache_path, self.images)
            # print(f"Saved processed images to {cache_path}")

        # Extract targets
        if self.phase != "test":
            self.targets = self.df["diagnosis"].values
        else:
            self.targets = np.zeros(len(self.df), dtype=int)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        target = self.targets[idx]

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        # Return image and target
        # Target is returned as a long tensor (integer index)
        # Conversion to ordinal vectors (if needed) happens in the training loop/loss
        return img, torch.tensor(target, dtype=torch.long)


def get_transforms(phase="train"):
    """
    Returns Albumentations transforms for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.
    """
    # Normalization parameters for ImageNet
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if phase == "train":
        return A.Compose(
            [
                # Geometric Augmentations (No photometric changes)
                A.HorizontalFlip(p=Config.aug_hflip_prob),
                A.VerticalFlip(p=Config.aug_vflip_prob),
                A.RandomRotate90(p=Config.aug_rotate90_prob),
                # Normalize and Convert to Tensor
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: Only Normalize and ToTensor
        return A.Compose(
            [
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """
    Performs Mixup on the input batch.

    Args:
        x (torch.Tensor): Input images.
        y (torch.Tensor): Target labels.
        alpha (float): Mixup beta distribution parameter.
        device (str): Device to perform operations on.

    Returns:
        mixed_x (torch.Tensor): Mixed images.
        y_a (torch.Tensor): Original targets.
        y_b (torch.Tensor): Permuted targets.
        lam (float): Mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def get_loaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached numpy arrays for images.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load Metadata DataFrames
    df_train = pd.read_csv(Config.train_metadata_path)
    df_val = pd.read_csv(Config.val_metadata_path)
    df_test = pd.read_csv(Config.test_metadata_path)

    # Handle Debug Mode: Subsample data
    if Config.debug:
        # print("Debug mode enabled: Subsampling datasets...")
        df_train = df_train.sample(
            n=min(100, len(df_train)), random_state=Config.seed
        ).reset_index(drop=True)
        df_val = df_val.sample(
            n=min(50, len(df_val)), random_state=Config.seed
        ).reset_index(drop=True)
        df_test = df_test.sample(
            n=min(20, len(df_test)), random_state=Config.seed
        ).reset_index(drop=True)

    # Initialize Datasets
    train_dataset = RetinaDataset(
        df_train,
        phase="train",
        transform=get_transforms("train"),
        load_cached_data=load_cached_data,
    )

    val_dataset = RetinaDataset(
        df_val,
        phase="val",
        transform=get_transforms("val"),
        load_cached_data=load_cached_data,
    )

    test_dataset = RetinaDataset(
        df_test,
        phase="test",
        transform=get_transforms("test"),
        load_cached_data=load_cached_data,
    )

    # Create DataLoaders
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
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
