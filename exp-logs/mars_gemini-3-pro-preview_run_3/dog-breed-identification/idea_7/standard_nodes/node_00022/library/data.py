import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from library.config import Config

# Standard ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(mode="train"):
    """
    Returns the strict preprocessing pipeline defined in Idea 7.

    Args:
        mode (str): 'train', 'val', or 'test'.
    """
    if mode == "train":
        return transforms.Compose(
            [
                # Explicitly retain RandomResizedCrop to prevent overfitting
                transforms.RandomResizedCrop(Config.IMG_SIZE, scale=(0.08, 1.0)),
                # RandomHorizontalFlip BEFORE RandAugment
                transforms.RandomHorizontalFlip(p=0.5),
                # RandAugment for geometric and photometric diversity
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    else:
        # Standard evaluation pipeline: Resize short edge to 256, then CenterCrop
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(Config.IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )


class DogDataset(Dataset):
    """
    PyTorch Dataset for Dog Breed Classification.
    """

    def __init__(self, df, transform=None, class_to_idx=None, is_test=False):
        self.df = df
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.is_test = is_test

        # Pre-compute full paths to avoid overhead in __getitem__
        # Metadata file_path is relative to input dir
        self.file_paths = [
            os.path.join(Config.INPUT_DIR, fp) for fp in df["file_path"].values
        ]

        if not self.is_test:
            self.breeds = df["breed"].values
            self.labels = [self.class_to_idx[b] for b in self.breeds]
        else:
            self.ids = df["id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        # Load image (PIL for torchvision compatibility)
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            # Fallback for corrupt images (though dataset analysis showed none)
            # Return a black image or raise. Here we raise to fail fast during dev.
            raise IOError(f"Error loading image {path}: {e}")

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            # Return image and ID for submission mapping
            return image, self.ids[idx]
        else:
            # Return image and label index
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label


def get_class_mapping(df_train, load_cached_data=True):
    """
    Generates or loads the class-to-index mapping.
    Ensures determinism across runs by caching the mapping.
    """
    cache_path = os.path.join(Config.WORK_DIR, "classes.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading class mapping from {cache_path}")
        classes_df = pd.read_parquet(cache_path)
        # Convert back to dict
        class_to_idx = dict(zip(classes_df["breed"], classes_df["idx"]))
        idx_to_class = dict(zip(classes_df["idx"], classes_df["breed"]))
        return class_to_idx, idx_to_class

    print("Generating new class mapping...")
    # Extract unique breeds and sort alphabetically
    unique_breeds = sorted(df_train["breed"].unique().tolist())

    class_to_idx = {breed: i for i, breed in enumerate(unique_breeds)}
    idx_to_class = {i: breed for i, breed in enumerate(unique_breeds)}

    # Save to cache
    classes_df = pd.DataFrame(
        {"breed": list(class_to_idx.keys()), "idx": list(class_to_idx.values())}
    )
    classes_df.to_parquet(cache_path, index=False)
    print(f"Saved class mapping to {cache_path}")

    return class_to_idx, idx_to_class


def get_dataloaders(load_cached_data=True):
    """
    Main function to prepare DataLoaders.

    Args:
        load_cached_data (bool): Whether to use cached class mappings.

    Returns:
        train_loader, val_loader, test_loader, class_to_idx
    """
    # 1. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    df_val = pd.read_csv(Config.VAL_METADATA)
    df_test = pd.read_csv(Config.TEST_METADATA)

    # 2. Handle Debug Mode
    if Config.DEBUG:
        print(f"DEBUG MODE: Subsampling {Config.DEBUG_SAMPLE_SIZE} samples.")
        df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    # 3. Get Class Mapping
    # We use the full training set to define classes even in debug mode
    # to ensure consistency, but here we pass the loaded df.
    # If debug is on, we might miss classes, so strictly we should load
    # classes from full metadata if possible.
    # For safety, we'll reload full train metadata just for class mapping if in debug.
    if Config.DEBUG:
        full_train_df = pd.read_csv(Config.TRAIN_METADATA)
        class_to_idx, _ = get_class_mapping(full_train_df, load_cached_data)
    else:
        class_to_idx, _ = get_class_mapping(df_train, load_cached_data)

    # 4. Create Datasets
    train_dataset = DogDataset(
        df_train,
        transform=get_transforms(mode="train"),
        class_to_idx=class_to_idx,
        is_test=False,
    )

    val_dataset = DogDataset(
        df_val,
        transform=get_transforms(mode="val"),
        class_to_idx=class_to_idx,
        is_test=False,
    )

    test_dataset = DogDataset(
        df_test, transform=get_transforms(mode="test"), class_to_idx=None, is_test=True
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Data Loaded Successfully:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    print(f"  Classes: {len(class_to_idx)}")

    return train_loader, val_loader, test_loader, class_to_idx
