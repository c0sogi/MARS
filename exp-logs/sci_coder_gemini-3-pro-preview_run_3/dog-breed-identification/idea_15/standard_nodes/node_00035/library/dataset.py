import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from library.config import Config

# ImageNet Normalization Constants
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transforms(mode="train"):
    """
    Returns the torchvision transformations for the specified mode.

    Args:
        mode (str): 'train' or 'val'/'test'.

    Returns:
        torchvision.transforms.Compose: The composition of transforms.
    """
    if mode == "train":
        # Strict augmentation policy: RandomResizedCrop -> HorizontalFlip -> RandAugment
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(Config.IMG_SIZE, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )
    else:
        # Standard validation: Resize to slightly larger, then CenterCrop
        # 256 is the standard resize dim for 224 crop in ImageNet models
        resize_dim = int(Config.IMG_SIZE * (256 / 224))
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
    PyTorch Dataset for Dog Breed Classification.
    """

    def __init__(self, df, transform=None, mode="train"):
        self.df = df
        self.transform = transform
        self.mode = mode

        # Pre-compute full paths to avoid overhead in __getitem__
        # Metadata contains relative paths (e.g., 'train/id.jpg')
        self.file_paths = [
            os.path.join(Config.INPUT_DIR, fp) for fp in df["file_path"].values
        ]

        # Handle labels
        self.labels = None
        if "label_idx" in df.columns:
            self.labels = df["label_idx"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            # Fallback for corrupt images (unlikely in this dataset but good practice)
            print(f"Warning: Error loading image {path}: {e}")
            image = Image.new("RGB", (Config.IMG_SIZE, Config.IMG_SIZE))

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label
        else:
            return image


def prepare_folds(load_cached_data=True):
    """
    Loads metadata, merges train/val splits, encodes labels based on submission header,
    creates stratified folds, and caches the result.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (df_with_folds, class_to_idx_mapping)
    """
    folds_path = os.path.join(Config.OUTPUT_DIR, "folds.parquet")
    classes_path = os.path.join(Config.OUTPUT_DIR, "classes.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(folds_path) and os.path.exists(classes_path):
        print(f"Loading cached folds from {folds_path}")
        df = pd.read_parquet(folds_path)
        classes_df = pd.read_parquet(classes_path)
        class_to_idx = {row["breed"]: row["idx"] for _, row in classes_df.iterrows()}
        return df, class_to_idx

    print("Creating folds from scratch...")

    # 2. Load and Merge Metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)

    # 3. Determine Class Mapping from Sample Submission
    # This guarantees that our model output indices match the submission columns
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION)
    # Submission format: id, breed_1, breed_2, ...
    class_names = sample_sub.columns[1:].tolist()
    class_to_idx = {breed: idx for idx, breed in enumerate(class_names)}

    # Validation: Ensure all training breeds are in the submission columns
    train_breeds = set(df["breed"].unique())
    sub_breeds = set(class_names)
    if not train_breeds.issubset(sub_breeds):
        missing = train_breeds - sub_breeds
        raise ValueError(
            f"Training data contains breeds not in sample submission: {missing}"
        )

    # Map breeds to indices
    df["label_idx"] = df["breed"].map(class_to_idx)

    # 4. Create Stratified Folds
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    df["fold"] = -1

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label_idx"])):
        df.loc[val_idx, "fold"] = fold

    # 5. Debug Mode Handling
    if Config.DEBUG:
        print("DEBUG mode: Subsampling data...")
        df = df.sample(n=min(len(df), 500), random_state=Config.SEED).reset_index(
            drop=True
        )
        # Re-split for the subset
        df["fold"] = -1
        try:
            skf_debug = StratifiedKFold(
                n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
            )
            for fold, (train_idx, val_idx) in enumerate(
                skf_debug.split(df, df["label_idx"])
            ):
                df.loc[val_idx, "fold"] = fold
        except ValueError:
            # Fallback if classes have too few samples in debug subset
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
            for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
                df.loc[val_idx, "fold"] = fold

    # 6. Cache Data
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    df.to_parquet(folds_path)

    classes_df = pd.DataFrame(list(class_to_idx.items()), columns=["breed", "idx"])
    classes_df.to_parquet(classes_path)

    print(f"Folds created and saved to {folds_path}")
    return df, class_to_idx


def get_dataloaders(fold_idx, load_cached_data=True):
    """
    Creates training and validation DataLoaders for a specific fold.

    Args:
        fold_idx (int): The fold index (0 to N_FOLDS-1).
        load_cached_data (bool): Whether to use cached fold definitions.

    Returns:
        tuple: (train_loader, val_loader)
    """
    df, _ = prepare_folds(load_cached_data=load_cached_data)

    train_df = df[df["fold"] != fold_idx].reset_index(drop=True)
    val_df = df[df["fold"] == fold_idx].reset_index(drop=True)

    train_dataset = DogDataset(
        train_df, transform=get_transforms("train"), mode="train"
    )
    val_dataset = DogDataset(val_df, transform=get_transforms("val"), mode="val")

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

    return train_loader, val_loader


def get_test_dataloader():
    """
    Creates the test DataLoader and returns the test metadata.

    Returns:
        tuple: (test_loader, test_df)
    """
    test_df = pd.read_csv(Config.TEST_CSV)

    if Config.DEBUG:
        test_df = test_df.head(100)

    test_dataset = DogDataset(test_df, transform=get_transforms("val"), mode="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader, test_df
