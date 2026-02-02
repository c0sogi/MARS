import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def get_transforms(mode="train"):
    """
    Returns the torchvision transforms for the specified mode.
    Strictly implements the geometric pipeline: Resize(274) -> CenterCrop(256).
    """
    # ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train":
        return transforms.Compose(
            [
                # Resize smaller edge to resize_target, preserving aspect ratio
                transforms.Resize(Config.resize_target),
                # Center crop to the final input size
                transforms.CenterCrop(Config.img_size),
                # Augmentation
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        # Validation and Test (Deterministic)
        return transforms.Compose(
            [
                transforms.Resize(Config.resize_target),
                transforms.CenterCrop(Config.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


class DogDataset(Dataset):
    """
    PyTorch Dataset for Dog Breed Classification.
    Handles loading images from disk and applying transforms.
    """

    def __init__(self, df, transform=None, mode="train", label_map=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (id, file_path, breed).
            transform (callable, optional): Transform to be applied on a sample.
            mode (str): 'train', 'val', or 'test'.
            label_map (dict, optional): Mapping from breed name to integer index.
        """
        self.df = df
        self.transform = transform
        self.mode = mode
        self.label_map = label_map

        # Pre-compute full paths to avoid overhead in __getitem__
        # Config.input_dir is "./input". file_path in df is relative, e.g., "train/xxx.jpg"
        self.file_paths = [
            os.path.join(Config.input_dir, fp) for fp in df["file_path"].values
        ]

        if self.mode != "test":
            # Ensure we have labels for train/val
            if "breed" not in df.columns and "label_idx" not in df.columns:
                raise ValueError(
                    "DataFrame must contain 'breed' or 'label_idx' for train/val modes."
                )

            # If label_idx is already in df (from caching), use it.
            # Otherwise map 'breed' using label_map.
            if "label_idx" in df.columns:
                self.labels = df["label_idx"].values
            else:
                if label_map is None:
                    raise ValueError(
                        "label_map must be provided if 'label_idx' is not in DataFrame."
                    )
                self.labels = [label_map[b] for b in df["breed"].values]
        else:
            self.labels = None
            self.ids = df["id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]

        # Load image using PIL (RGB)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image in case of error to prevent crash,
            # though dataset should be verified beforehand.
            image = Image.new("RGB", (Config.img_size, Config.img_size))

        if self.transform:
            image = self.transform(image)

        if self.mode == "test":
            # For test, return image and the ID (for submission file)
            return image, self.ids[idx]
        else:
            # For train/val, return image and integer label
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label


def process_metadata(load_cached_data=True):
    """
    Loads and processes metadata. Implements caching mechanism.

    Logic:
    1. If load_cached_data is True, try to load processed parquets and label_map.npy.
    2. If not found or load_cached_data is False, load raw CSVs from ./metadata.
    3. Generate label_map from training data.
    4. Map breeds to integers.
    5. Save processed data to ./working/idea_6/.

    Returns:
        train_df, val_df, test_df, label_map (list of class names)
    """
    os.makedirs(Config.working_dir, exist_ok=True)

    train_cache = os.path.join(Config.working_dir, "train_processed.parquet")
    val_cache = os.path.join(Config.working_dir, "val_processed.parquet")
    test_cache = os.path.join(Config.working_dir, "test_processed.parquet")
    label_map_cache = os.path.join(Config.working_dir, "label_map.npy")

    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
            and os.path.exists(label_map_cache)
        ):

            train_df = pd.read_parquet(train_cache)
            val_df = pd.read_parquet(val_cache)
            test_df = pd.read_parquet(test_cache)
            classes = np.load(label_map_cache, allow_pickle=True).tolist()

            return train_df, val_df, test_df, classes

    # Load raw metadata
    train_df = pd.read_csv(Config.train_metadata_path)
    val_df = pd.read_csv(Config.val_metadata_path)
    test_df = pd.read_csv(Config.test_metadata_path)

    # Generate Label Map (sorted unique breeds from training data)
    classes = sorted(train_df["breed"].unique().tolist())
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    # Map labels to integers
    train_df["label_idx"] = train_df["breed"].map(class_to_idx)
    val_df["label_idx"] = val_df["breed"].map(class_to_idx)

    # Save to cache
    train_df.to_parquet(train_cache)
    val_df.to_parquet(val_cache)
    test_df.to_parquet(test_cache)
    np.save(label_map_cache, np.array(classes))

    return train_df, val_df, test_df, classes


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        debug (bool): If True, subsets the data for quick debugging.
        load_cached_data (bool): Whether to use cached metadata.

    Returns:
        train_loader, val_loader, test_loader, classes
    """
    # 1. Process Metadata
    train_df, val_df, test_df, classes = process_metadata(
        load_cached_data=load_cached_data
    )

    # 2. Handle Debug Mode
    if debug:
        train_df = train_df.head(Config.batch_size * 2)
        val_df = val_df.head(Config.batch_size)
        test_df = test_df.head(Config.batch_size)

    # 3. Create Datasets
    # Pass class_to_idx mapping implicitly via the 'label_idx' column in processed DFs
    # But we also pass the map dict for safety if raw DFs were used (though process_metadata handles this)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_dataset = DogDataset(
        train_df,
        transform=get_transforms(mode="train"),
        mode="train",
        label_map=class_to_idx,
    )

    val_dataset = DogDataset(
        val_df, transform=get_transforms(mode="val"), mode="val", label_map=class_to_idx
    )

    test_dataset = DogDataset(
        test_df, transform=get_transforms(mode="test"), mode="test"
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,  # Good for BatchNorm stability
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

    return train_loader, val_loader, test_loader, classes
