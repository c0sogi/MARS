import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from library.config import CFG


class CassavaDataset(Dataset):
    """
    Custom Dataset for Cassava Leaf Disease Classification.
    Loads images using PIL and applies transformations.
    """

    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Construct full file path
        # Metadata file_path is relative, e.g., "train_images/1000015157.jpg"
        img_path = os.path.join(CFG.input_dir, row["file_path"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback for corrupt images
            print(f"Warning: Could not load image {img_path}. Error: {e}")
            image = Image.new("RGB", (CFG.image_size, CFG.image_size), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        # Label is required for training/validation
        # For test data, the label in metadata is a placeholder (4), which is fine to return
        label = torch.tensor(row["label"], dtype=torch.long)

        return image, label


def get_transforms(data="train"):
    """
    Returns the torchvision transformations for the specified data split.
    """
    if data == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(CFG.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    elif data == "valid" or data == "test":
        return transforms.Compose(
            [
                # Resize the shortest edge to image_size, then center crop
                transforms.Resize(CFG.image_size),
                transforms.CenterCrop(CFG.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        raise ValueError(f"Unknown data split: {data}")


def prepare_folds(load_cached_data=True):
    """
    Loads metadata, combines train/val splits, and creates 5 Stratified Folds.
    Caches the result to a parquet file.
    """
    cache_path = os.path.join(CFG.output_dir, "folds.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cached folds: {e}. Recomputing...")

    # 2. Compute from scratch
    # Load existing metadata
    train_meta_path = os.path.join(CFG.metadata_dir, "train.csv")
    val_meta_path = os.path.join(CFG.metadata_dir, "val.csv")

    if not os.path.exists(train_meta_path) or not os.path.exists(val_meta_path):
        raise FileNotFoundError("Metadata CSV files not found in ./metadata/")

    df_train = pd.read_csv(train_meta_path)
    df_val = pd.read_csv(val_meta_path)

    # Combine to create a full dataset for 5-fold CV
    df = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)

    # Create Stratified Folds
    skf = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    df["fold"] = -1

    for fold, (_, val_idx) in enumerate(skf.split(df, df["label"])):
        df.loc[val_idx, "fold"] = fold

    # 3. Save to cache
    os.makedirs(CFG.output_dir, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


def get_loaders(fold, load_cached_data=True):
    """
    Creates DataLoaders for the specified fold.
    """
    df = prepare_folds(load_cached_data=load_cached_data)

    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)

    train_dataset = CassavaDataset(train_df, transform=get_transforms(data="train"))
    valid_dataset = CassavaDataset(valid_df, transform=get_transforms(data="valid"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, valid_loader


def get_test_loader():
    """
    Creates DataLoader for the test set.
    """
    test_meta_path = os.path.join(CFG.metadata_dir, "test.csv")
    if not os.path.exists(test_meta_path):
        raise FileNotFoundError(f"Test metadata not found at {test_meta_path}")

    df_test = pd.read_csv(test_meta_path)

    test_dataset = CassavaDataset(df_test, transform=get_transforms(data="test"))

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
