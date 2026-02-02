import os
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from library.config import Config

# Standard ImageNet normalization statistics
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transforms(phase: str):
    """
    Constructs the data augmentation and transformation pipeline.

    Args:
        phase (str): One of 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The transformation pipeline.
    """
    if phase == "train":
        # Training pipeline: RandomResizedCrop -> Flip -> RandAugment
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(Config.IMG_SIZE, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )
    else:
        # Validation/Test pipeline: Resize -> CenterCrop
        # We resize to slightly larger than the target size before cropping
        resize_dim = int(Config.IMG_SIZE * 256 / 224)
        return transforms.Compose(
            [
                transforms.Resize(resize_dim),
                transforms.CenterCrop(Config.IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )


class DogDataset(Dataset):
    """
    Custom Dataset for loading Dog images.
    """

    def __init__(
        self, df, transform=None, return_label=True, input_dir=Config.INPUT_DIR
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (file_path, breed/id).
            transform (callable, optional): Optional transform to be applied on a sample.
            return_label (bool): Whether to return the label (for training) or ID (for testing).
            input_dir (str): Root directory for images.
        """
        self.df = df
        self.transform = transform
        self.return_label = return_label
        self.input_dir = input_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Load image using OpenCV
        img = cv2.imread(img_path)

        # Handle potential loading errors (though analysis showed 0 missing files)
        if img is None:
            # Return a black image as fallback to prevent crashing
            img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for torchvision compatibility
        img = Image.fromarray(img)

        # Apply transformations
        if self.transform:
            img = self.transform(img)

        if self.return_label:
            # Return image and integer label
            label = row["label_idx"]
            return img, torch.tensor(label, dtype=torch.long)
        else:
            # Return image and ID (for submission file creation)
            return img, row["id"]


def process_and_cache_data(load_cached_data=True):
    """
    Loads metadata, combines train/val splits, performs label encoding,
    generates stratified K-Folds, and caches the result.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        tuple: (pd.DataFrame with 'fold' and 'label_idx' columns, list of class names)
    """
    cache_path = os.path.join(Config.WORKING_DIR, "folds.parquet")
    classes_path = os.path.join(Config.WORKING_DIR, "classes.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path) and os.path.exists(classes_path):
        df = pd.read_parquet(cache_path)
        classes_df = pd.read_parquet(classes_path)
        classes = classes_df["breed"].tolist()
        return df, classes

    # 2. Process from scratch
    # Load existing metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    # Combine to use all labeled data for CV
    df = pd.concat([df_train, df_val], ignore_index=True)

    # Encode Labels
    # Sort classes alphabetically to ensure index 0 matches submission file order
    classes = sorted(df["breed"].unique().tolist())
    le = LabelEncoder()
    le.fit(classes)
    df["label_idx"] = le.transform(df["breed"])

    # Create Stratified K-Folds
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    df["fold"] = -1

    # Assign fold indices
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label_idx"])):
        df.loc[val_idx, "fold"] = fold

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    pd.DataFrame({"breed": classes}).to_parquet(classes_path, index=False)

    return df, classes


def get_dataloaders(fold_idx, load_cached_data=True):
    """
    Creates DataLoaders for a specific fold of the Cross-Validation.

    Args:
        fold_idx (int): The fold index to use for validation (0 to N_FOLDS-1).
        load_cached_data (bool): Whether to use cached DataFrame.

    Returns:
        tuple: (train_loader, val_loader, class_names)
    """
    df, classes = process_and_cache_data(load_cached_data=load_cached_data)

    # Split dataframe based on fold index
    train_df = df[df["fold"] != fold_idx].reset_index(drop=True)
    val_df = df[df["fold"] == fold_idx].reset_index(drop=True)

    # Create Datasets
    train_dataset = DogDataset(
        train_df, transform=get_transforms("train"), return_label=True
    )
    val_dataset = DogDataset(val_df, transform=get_transforms("val"), return_label=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, classes


def get_test_dataloader():
    """
    Creates a DataLoader for the test set.

    Returns:
        DataLoader: The test data loader.
    """
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    test_dataset = DogDataset(
        df_test, transform=get_transforms("test"), return_label=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
