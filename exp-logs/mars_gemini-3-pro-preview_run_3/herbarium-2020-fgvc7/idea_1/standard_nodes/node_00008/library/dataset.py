import os
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import library.config as config


def get_label_encoder(train_df=None):
    """
    Retrieves or creates the label encoder mapping.
    Returns:
        raw_to_idx (dict): Mapping from category_id to index (0..N-1)
        idx_to_raw (dict): Mapping from index to category_id
    """
    if os.path.exists(config.LABEL_ENCODER_PATH):
        with open(config.LABEL_ENCODER_PATH, "r") as f:
            data = json.load(f)
            # JSON keys are strings, convert back to int
            raw_to_idx = {int(k): v for k, v in data["raw_to_idx"].items()}
            idx_to_raw = {int(k): v for k, v in data["idx_to_raw"].items()}
            return raw_to_idx, idx_to_raw

    if train_df is None:
        # If file doesn't exist and no df provided, we can't create it.
        # Try loading the full train csv to create it.
        if os.path.exists(config.TRAIN_CSV):
            train_df = pd.read_csv(config.TRAIN_CSV)
        else:
            raise ValueError(
                "Label encoder not found and cannot load train.csv to create it."
            )

    # Create mapping
    unique_ids = sorted(train_df["category_id"].unique().tolist())
    raw_to_idx = {raw: idx for idx, raw in enumerate(unique_ids)}
    idx_to_raw = {idx: raw for idx, raw in enumerate(unique_ids)}

    # Save mapping
    os.makedirs(os.path.dirname(config.LABEL_ENCODER_PATH), exist_ok=True)
    with open(config.LABEL_ENCODER_PATH, "w") as f:
        json.dump({"raw_to_idx": raw_to_idx, "idx_to_raw": idx_to_raw}, f)

    return raw_to_idx, idx_to_raw


class PlantDataset(Dataset):
    """
    Custom Dataset for Plant Species Classification.
    """

    def __init__(self, df, root_dir, transform=None, is_test=False, raw_to_idx=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.raw_to_idx = raw_to_idx

        if not self.is_test and self.raw_to_idx is None:
            raise ValueError(
                "raw_to_idx mapping must be provided for training/validation sets"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Construct full path. Metadata paths are relative to input root.
        img_path = os.path.join(self.root_dir, row["file_path"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback for corrupt images (though analysis showed none)
            # Return a black image to prevent training crash
            image = Image.new("RGB", (config.IMG_SIZE, config.IMG_SIZE))

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            # Return image and ID for submission file generation
            return image, row["image_id"]
        else:
            # Return image and class label for training/validation
            raw_label = row["category_id"]
            # Map raw label to index
            label = self.raw_to_idx[raw_label]
            return image, torch.tensor(label, dtype=torch.long)


def _get_train_weights(train_df, load_cached_data=True):
    """
    Computes or loads sample weights for WeightedRandomSampler.

    Args:
        train_df (pd.DataFrame): Training metadata.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        torch.Tensor: Weights for each sample in train_df.
    """
    # Cite solution_lesson_node_00007: Dampened resampling (sqrt) to prevent overfitting on tail.
    cache_path = os.path.join(config.WORKING_DIR, "train_weights_sqrt.npy")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            weights = np.load(cache_path)
            if len(weights) == len(train_df):
                return torch.from_numpy(weights).double()
        except Exception:
            pass  # Fallback to computation if load fails

    # 2. Compute weights from scratch
    # Count frequency of each class
    class_counts = train_df["category_id"].value_counts().sort_index()

    # Weight = 1 / sqrt(frequency) - Dampened resampling
    # We create a mapping from category_id to weight
    weight_map = 1.0 / np.sqrt(class_counts)

    # Map weights to each sample in the dataframe
    # map() is efficient for this operation
    sample_weights = train_df["category_id"].map(weight_map).values.astype(np.float64)

    # 3. Save to cache
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    np.save(cache_path, sample_weights)

    return torch.from_numpy(sample_weights).double()


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached artifacts for data processing.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # --- Load Metadata ---
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)
    test_df = pd.read_csv(config.TEST_CSV)

    # --- Ensure Label Encoder Exists (using FULL train set) ---
    raw_to_idx, _ = get_label_encoder(train_df)

    # --- Apply Debug Limits ---
    if config.MAX_TRAIN_SAMPLES is not None:
        train_df = train_df.iloc[: config.MAX_TRAIN_SAMPLES]

    if config.MAX_VAL_SAMPLES is not None:
        val_df = val_df.iloc[: config.MAX_VAL_SAMPLES]
        # Usually we don't limit test set unless specifically requested,
        # but for consistency with debug modes, we can leave it full or limit if needed.
        # Here we keep test set full as it's for submission.

    # --- Define Transforms ---
    # Standard ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Training: Resize slightly larger -> Random Crop -> Flip -> Norm
    # Cite solution_lesson_node_00007: Aggressive augmentation for long-tail generalization.
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # Evaluation: Resize -> Center Crop -> Norm
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # --- Prepare Sampler (Class Balancing) ---
    # If we are debugging with a subset, we must recompute weights on the fly
    # to avoid shape mismatch or incorrect statistics.
    if config.MAX_TRAIN_SAMPLES is not None:
        # Compute locally without caching to file
        class_counts = train_df["category_id"].value_counts()
        # Cite solution_lesson_node_00007: Dampened resampling here as well for consistency
        weight_map = 1.0 / np.sqrt(class_counts)
        weights_np = train_df["category_id"].map(weight_map).values.astype(np.float64)
        train_weights = torch.from_numpy(weights_np).double()
    else:
        # Use caching logic for full dataset
        train_weights = _get_train_weights(train_df, load_cached_data=load_cached_data)

    sampler = WeightedRandomSampler(
        weights=train_weights, num_samples=len(train_weights), replacement=True
    )

    # --- Create Datasets ---
    train_dataset = PlantDataset(
        train_df, config.INPUT_ROOT, transform=train_transform, raw_to_idx=raw_to_idx
    )
    val_dataset = PlantDataset(
        val_df, config.INPUT_ROOT, transform=eval_transform, raw_to_idx=raw_to_idx
    )
    test_dataset = PlantDataset(
        test_df, config.INPUT_ROOT, transform=eval_transform, is_test=True
    )

    # --- Create DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,  # Sampler is mutually exclusive with shuffle
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
