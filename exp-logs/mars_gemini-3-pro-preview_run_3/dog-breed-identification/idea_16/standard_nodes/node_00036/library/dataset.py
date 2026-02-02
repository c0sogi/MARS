import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sklearn.model_selection import StratifiedKFold
from PIL import Image

# Import from provided library files
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("dataset", os.path.join(Config.working_dir, "dataset.log"))


class DogDataset(Dataset):
    """
    Dataset class for loading dog images and corresponding labels.
    """

    def __init__(
        self,
        df,
        transform=None,
        class_to_idx=None,
        is_test=False,
        input_dir=Config.input_dir,
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (id, file_path, breed).
            transform (callable, optional): Optional transform to be applied on a sample.
            class_to_idx (dict, optional): Mapping from breed name to integer index.
            is_test (bool): Flag to indicate if this is the test set (no labels).
            input_dir (str): Root directory for images.
        """
        self.df = df
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.is_test = is_test
        self.input_dir = input_dir

        # Pre-compute full paths to avoid doing it in __getitem__
        # The metadata contains relative paths like 'train/id.jpg'
        self.file_paths = [
            os.path.join(self.input_dir, fp) for fp in df["file_path"].values
        ]

        if not self.is_test:
            self.breeds = df["breed"].values
            if self.class_to_idx is None:
                raise ValueError(
                    "class_to_idx must be provided for training/validation sets."
                )
            self.labels = [self.class_to_idx[breed] for breed in self.breeds]
        else:
            self.ids = df["id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        # Load image using PIL (compatible with torchvision transforms)
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image at {path}: {e}")
            # Return a black image as fallback to prevent crashing
            image = Image.new("RGB", (Config.image_size, Config.image_size))

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, self.ids[idx]
        else:
            label = self.labels[idx]
            return image, torch.tensor(label, dtype=torch.long)


def get_transforms(mode="train"):
    """
    Returns the specific preprocessing pipeline required.

    Args:
        mode (str): 'train' or 'val'/'test'.

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    # ImageNet statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train":
        # Proposed Solution: RandomResizedCrop -> RandomHorizontalFlip -> RandAugment
        return T.Compose(
            [
                T.RandomResizedCrop(
                    Config.image_size, scale=(0.8, 1.0)
                ),  # Conservative scale to avoid losing too much context
                T.RandomHorizontalFlip(),
                T.RandAugment(num_ops=2, magnitude=9),  # Standard RandAugment settings
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )
    else:
        # Validation/Test: Deterministic resizing
        # We use Resize then CenterCrop to maintain aspect ratio logic standard in classification
        # However, prompt says "Inputs will be fixed at 224x224".
        # To match training distribution best without distortion, we resize short edge to 256 then crop.
        return T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(Config.image_size),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )


def _get_folds(df, n_folds=5, seed=42, load_cached_data=True):
    """
    Generates or loads stratified K-fold splits.

    Args:
        df (pd.DataFrame): The full labeled dataframe.
        n_folds (int): Number of folds.
        seed (int): Random seed.
        load_cached_data (bool): Whether to use cached folds.

    Returns:
        pd.DataFrame: DataFrame with a 'fold' column.
    """
    cache_path = os.path.join(Config.working_dir, "folds.parquet")

    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached folds from {cache_path}")
        return pd.read_parquet(cache_path)

    logger.info("Generating new stratified folds...")
    df["fold"] = -1
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold, (_, val_idx) in enumerate(skf.split(df, df["breed"])):
        df.loc[val_idx, "fold"] = fold

    # Cache the result
    df.to_parquet(cache_path, index=False)
    logger.info(f"Saved folds to {cache_path}")

    return df


def _get_classes(df, load_cached_data=True):
    """
    Generates or loads class-to-index mapping.

    Args:
        df (pd.DataFrame): DataFrame containing 'breed' column.
        load_cached_data (bool): Whether to use cached mapping.

    Returns:
        dict: class_to_idx mapping.
        list: list of class names sorted by index.
    """
    cache_path = os.path.join(Config.working_dir, "classes.parquet")

    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached classes from {cache_path}")
        classes_df = pd.read_parquet(cache_path)
        classes = classes_df["breed"].tolist()
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        return class_to_idx, classes

    logger.info("Generating class mappings...")
    classes = sorted(df["breed"].unique().tolist())
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    # Cache
    pd.DataFrame({"breed": classes}).to_parquet(cache_path, index=False)

    return class_to_idx, classes


def get_dataloaders(fold=0, load_cached_data=True):
    """
    Constructs training and validation dataloaders for a specific fold.

    Args:
        fold (int): The fold index (0 to n_folds-1) to use for validation.
        load_cached_data (bool): Whether to use cached metadata/folds.

    Returns:
        train_loader (DataLoader)
        val_loader (DataLoader)
        classes (list): List of class names.
    """
    # 1. Load Metadata
    # We combine train and val metadata to perform proper N-fold CV
    df_train_meta = pd.read_csv(Config.train_metadata_path)
    df_val_meta = pd.read_csv(Config.val_metadata_path)
    df_full = pd.concat([df_train_meta, df_val_meta], ignore_index=True)

    # 2. Get Class Mappings
    class_to_idx, classes = _get_classes(df_full, load_cached_data=load_cached_data)

    # 3. Get Folds
    df_folds = _get_folds(
        df_full,
        n_folds=Config.n_folds,
        seed=Config.seed,
        load_cached_data=load_cached_data,
    )

    # 4. Split Data
    train_df = df_folds[df_folds["fold"] != fold].reset_index(drop=True)
    val_df = df_folds[df_folds["fold"] == fold].reset_index(drop=True)

    if Config.debug:
        logger.info(
            f"Debug mode: Subsampling data to {Config.debug_sample_size} samples."
        )
        train_df = train_df.head(Config.debug_sample_size)
        val_df = val_df.head(Config.debug_sample_size)

    logger.info(
        f"Fold {fold}: Train samples: {len(train_df)}, Val samples: {len(val_df)}"
    )

    # 5. Create Datasets
    train_dataset = DogDataset(
        train_df,
        transform=get_transforms("train"),
        class_to_idx=class_to_idx,
        is_test=False,
    )

    val_dataset = DogDataset(
        val_df,
        transform=get_transforms("val"),
        class_to_idx=class_to_idx,
        is_test=False,
    )

    # 6. Create Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,  # Useful for Batch Norm stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, classes


def get_test_dataloader(load_cached_data=True):
    """
    Constructs the test dataloader.

    Args:
        load_cached_data (bool): Used to retrieve cached class mappings.

    Returns:
        test_loader (DataLoader)
        test_df (pd.DataFrame): The test metadata dataframe (for IDs).
    """
    df_test = pd.read_csv(Config.test_metadata_path)

    # We need class mappings just to ensure consistency if needed,
    # though test dataset doesn't use them for labels.
    # We load them to ensure the cache exists.
    # We use train metadata to establish the classes.
    df_train_meta = pd.read_csv(Config.train_metadata_path)
    df_val_meta = pd.read_csv(Config.val_metadata_path)
    df_full = pd.concat([df_train_meta, df_val_meta], ignore_index=True)
    _, _ = _get_classes(df_full, load_cached_data=load_cached_data)

    test_dataset = DogDataset(
        df_test,
        transform=get_transforms("val"),  # Use validation transforms (deterministic)
        class_to_idx=None,
        is_test=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return test_loader, df_test
