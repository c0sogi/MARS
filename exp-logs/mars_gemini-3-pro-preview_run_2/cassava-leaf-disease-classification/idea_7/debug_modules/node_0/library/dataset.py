import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T
from typing import Tuple, Optional
from sklearn.model_selection import StratifiedKFold

from library.config import Config


class CassavaDataset(Dataset):
    """
    Custom Dataset for Cassava Leaf Disease Classification.
    Uses PIL for image loading and supports torchvision transforms.
    """

    def __init__(
        self, df: pd.DataFrame, input_dir: str, transform: Optional[T.Compose] = None
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'file_path' and 'label' columns.
            input_dir (str): Root directory where images are stored.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df
        self.input_dir = input_dir
        self.transform = transform

        # Pre-extract values for faster access
        self.file_paths = df["file_path"].values
        self.labels = df["label"].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Construct full file path
        # metadata file_path is relative (e.g., "train_images/xyz.jpg")
        img_path = os.path.join(self.input_dir, self.file_paths[idx])

        try:
            # Load image using PIL and convert to RGB
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback for potential read errors (though dataset is verified)
            print(f"Warning: Could not load image {img_path}. Error: {e}")
            # Return a blank image to prevent crash
            image = Image.new("RGB", (600, 800))

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, label


def get_transforms(data_split: str, img_size: int) -> T.Compose:
    """
    Generates the transformation pipeline based on the data split and target resolution.

    Args:
        data_split (str): 'train', 'val', or 'test'.
        img_size (int): Target resolution (e.g., 224 or 384).

    Returns:
        T.Compose: Composed transforms.
    """
    # ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if data_split == "train":
        return T.Compose(
            [
                # Geometric Augmentations
                T.RandomResizedCrop(img_size),
                T.RandomHorizontalFlip(),
                # Photometric Augmentations
                T.RandAugment(num_ops=2, magnitude=9),
                # Conversion and Normalization
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
    else:
        # Validation/Test: Resize then CenterCrop
        # This preserves aspect ratio better than direct resize
        return T.Compose(
            [
                T.Resize(img_size),  # Resizes smaller edge to img_size
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )


def get_loaders(
    config: Config, phase: int, fold_idx: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepares DataLoaders for training, validation, and testing.
    Handles Progressive Resolution (via phase) and Cross-Validation (via fold_idx).

    Args:
        config (Config): Configuration object.
        phase (int): Current training phase (1 or 2).
        fold_idx (int, optional): Index of the fold for CV. If None, uses default split.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: train_loader, val_loader, test_loader
    """

    # 1. Determine Image Size based on Phase
    if phase == 1:
        img_size = config.phase1_image_size
    elif phase == 2:
        img_size = config.phase2_image_size
    else:
        raise ValueError(f"Invalid phase {phase}. Expected 1 or 2.")

    # 2. Load Metadata
    df_train_meta = pd.read_csv(config.train_metadata_path)
    df_val_meta = pd.read_csv(config.val_metadata_path)
    df_test = pd.read_csv(config.test_metadata_path)

    # 3. Handle Cross-Validation Split
    if fold_idx is not None:
        # Concatenate original train and val to reform the full training set
        df_full = pd.concat([df_train_meta, df_val_meta], ignore_index=True)

        # Create Stratified K-Fold
        skf = StratifiedKFold(
            n_splits=config.n_folds, shuffle=True, random_state=config.seed
        )

        # Get indices for the specific fold
        # We iterate to find the specific fold indices
        splits = list(skf.split(df_full, df_full["label"]))
        train_idx, val_idx = splits[fold_idx]

        df_train = df_full.iloc[train_idx].reset_index(drop=True)
        df_val = df_full.iloc[val_idx].reset_index(drop=True)
    else:
        # Use the default split provided in metadata
        df_train = df_train_meta
        df_val = df_val_meta

    # 4. Handle Debug Mode (Subset Data)
    if config.debug:
        subset_size = config.subset_size if config.subset_size else 100
        df_train = df_train.iloc[:subset_size]
        df_val = df_val.iloc[:subset_size]
        df_test = df_test.iloc[:subset_size]

    # 5. Define Transforms
    train_transform = get_transforms("train", img_size)
    val_transform = get_transforms("val", img_size)
    test_transform = get_transforms("test", img_size)

    # 6. Instantiate Datasets
    train_dataset = CassavaDataset(df_train, config.input_dir, train_transform)
    val_dataset = CassavaDataset(df_val, config.input_dir, val_transform)
    test_dataset = CassavaDataset(df_test, config.input_dir, test_transform)

    # 7. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
