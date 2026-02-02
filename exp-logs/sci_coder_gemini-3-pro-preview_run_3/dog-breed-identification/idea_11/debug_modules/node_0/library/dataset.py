import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from library.config import Config


def get_class_mapping(load_cached_data=True):
    """
    Generates or loads the mapping between class names (breeds) and indices.

    Args:
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        list: Sorted list of class names.
        dict: Mapping from class name to index.
    """
    cache_path = os.path.join(Config.OUTPUT_DIR, "classes.parquet")

    if load_cached_data and os.path.exists(cache_path):
        df_classes = pd.read_parquet(cache_path)
        classes = df_classes["breed"].tolist()
    else:
        # Load training metadata to determine classes
        if not os.path.exists(Config.TRAIN_CSV):
            raise FileNotFoundError(
                f"Training metadata not found at {Config.TRAIN_CSV}"
            )

        df_train = pd.read_csv(Config.TRAIN_CSV)
        classes = sorted(df_train["breed"].unique().tolist())

        # Save to cache
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        pd.DataFrame({"breed": classes}).to_parquet(cache_path)

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    return classes, class_to_idx


def get_transforms(data_type="train"):
    """
    Returns the transformation pipeline based on the data type.

    Args:
        data_type (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    # ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if data_type == "train":
        # Strict adherence to the strategy:
        # RandomResizedCrop -> RandomHorizontalFlip -> RandAugment
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(Config.IMAGE_SIZE, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        # Validation/Test: Resize -> CenterCrop
        # Typically resize to slightly larger than target, then center crop
        resize_dim = int(Config.IMAGE_SIZE * 256 / 224)
        return transforms.Compose(
            [
                transforms.Resize(resize_dim),
                transforms.CenterCrop(Config.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


class DogDataset(Dataset):
    """
    Custom Dataset for loading Dog images and labels.
    """

    def __init__(self, df, transform=None, class_to_idx=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata.
            transform (callable, optional): Transform to be applied on a sample.
            class_to_idx (dict, optional): Mapping from breed name to index.
            is_test (bool): If True, returns (image, id). If False, returns (image, label).
        """
        self.df = df
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.is_test = is_test

        # Pre-compute full paths
        self.image_paths = [
            os.path.join(Config.INPUT_DIR, path) for path in df["file_path"]
        ]
        self.ids = df["id"].values

        if not self.is_test:
            self.labels = df["breed"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback for corrupted images (though analysis showed 0 missing)
            # Return a black image or raise error.
            # Given competition context, we'll create a blank image to avoid crashing.
            print(f"Warning: Could not load image {img_path}. Error: {e}")
            image = Image.new("RGB", (Config.IMAGE_SIZE, Config.IMAGE_SIZE))

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.is_test:
            # Return image and ID for submission creation
            return image, self.ids[idx]
        else:
            # Return image and label index
            label_name = self.labels[idx]
            label_idx = self.class_to_idx[label_name]
            return image, torch.tensor(label_idx, dtype=torch.long)


def get_dataloaders(fold_idx=None, load_cached_data=True):
    """
    Creates DataLoaders for training and validation (or testing).

    Args:
        fold_idx (int, optional): If provided, splits training data for K-Fold CV.
                                  However, the metadata provided is already split into
                                  train.csv (80%) and val.csv (20%).
                                  The task description implies we should use the provided splits
                                  or re-split if we want 5-fold on the full train set.

                                  Given the metadata generation script in the prompt:
                                  - train.csv is 80%
                                  - val.csv is 20%

                                  The Config.N_FOLDS = 5 implies we might want to do Cross Validation.
                                  However, the metadata is static.

                                  To support the 5-Fold Stratified Ensemble strategy described in "Idea",
                                  we should ideally combine train.csv and val.csv and perform our own splitting
                                  based on fold_idx.

    Returns:
        train_loader, val_loader, classes
    """

    # 1. Get Class Mapping
    classes, class_to_idx = get_class_mapping(load_cached_data=load_cached_data)

    # 2. Load Metadata
    df_train_meta = pd.read_csv(Config.TRAIN_CSV)
    df_val_meta = pd.read_csv(Config.VAL_CSV)

    # Combine for K-Fold logic
    df_full = pd.concat([df_train_meta, df_val_meta], ignore_index=True)

    # 3. Handle Debug Mode
    if Config.DEBUG:
        df_full = df_full.sample(
            n=min(len(df_full), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        print(f"DEBUG MODE: Reduced dataset to {len(df_full)} samples.")

    # 4. Stratified K-Fold Split
    # We use a deterministic shuffle based on seed to ensure folds are consistent
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Create a generator to get the specific fold
    # We iterate to find the indices for the requested fold_idx
    # If fold_idx is None, we default to Fold 0
    target_fold = fold_idx if fold_idx is not None else 0

    splits = list(skf.split(df_full, df_full["breed"]))
    train_idx, val_idx = splits[target_fold]

    df_train = df_full.iloc[train_idx].reset_index(drop=True)
    df_val = df_full.iloc[val_idx].reset_index(drop=True)

    # 5. Create Datasets
    train_dataset = DogDataset(
        df_train, transform=get_transforms("train"), class_to_idx=class_to_idx
    )

    val_dataset = DogDataset(
        df_val, transform=get_transforms("val"), class_to_idx=class_to_idx
    )

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,  # Drop last incomplete batch for stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    return train_loader, val_loader, classes


def get_test_loader():
    """
    Creates DataLoader for the test set.
    """
    df_test = pd.read_csv(Config.TEST_CSV)

    if Config.DEBUG:
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    test_dataset = DogDataset(df_test, transform=get_transforms("test"), is_test=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    return test_loader
