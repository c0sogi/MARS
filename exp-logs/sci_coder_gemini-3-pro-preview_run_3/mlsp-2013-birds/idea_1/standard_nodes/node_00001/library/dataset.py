import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library import config


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_histogram_features(load_cached_data=True):
    """
    Parses the histogram_of_segments.txt file into a DataFrame.
    Implements caching using Parquet to avoid re-parsing.

    Args:
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        pd.DataFrame: DataFrame containing 'rec_id' and 100 feature columns.
    """
    cache_path = os.path.join(config.WORKING_DIR, "histogram_features.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            pass  # Fallback to processing from scratch

    # 2. Process from scratch
    feature_data = []
    with open(config.HISTOGRAM_PATH, "r") as f:
        # Skip potential header if it exists (the file format desc says "rec_id,[histogram...]")
        # Checking first line
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split(",")

        # Check if header
        if parts[0] == "rec_id":
            continue

        try:
            rec_id = int(parts[0])
            # The rest are features
            features = [float(x) for x in parts[1:]]

            # Ensure we have exactly 100 features
            if len(features) != config.INPUT_DIM:
                # In case of trailing commas or issues, truncate or pad?
                # Assuming data is clean based on description, but let's be safe.
                # If strictly 100 dim is required.
                features = features[: config.INPUT_DIM]

            row = {"rec_id": rec_id}
            for i, val in enumerate(features):
                row[f"feat_{i}"] = val
            feature_data.append(row)
        except ValueError:
            continue

    df = pd.DataFrame(feature_data)

    # 3. Save to cache
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


class BirdDataset(Dataset):
    """
    PyTorch Dataset for Bird Species Classification.
    """

    def __init__(self, df, feature_cols, phase="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata and features.
            feature_cols (list): List of column names corresponding to features.
            phase (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.phase = phase
        self.num_classes = config.NUM_CLASSES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Extract features
        features = row[self.feature_cols].values.astype(np.float32)
        features_tensor = torch.tensor(features, dtype=torch.float32)

        rec_id = int(row["rec_id"])

        # Extract Labels
        if self.phase in ["train", "val"]:
            label_str = str(row["labels"])
            label_vec = np.zeros(self.num_classes, dtype=np.float32)

            if label_str != "?" and label_str.lower() != "nan":
                try:
                    indices = [int(x) for x in label_str.split()]
                    for cls_idx in indices:
                        if 0 <= cls_idx < self.num_classes:
                            label_vec[cls_idx] = 1.0
                except ValueError:
                    pass

            labels_tensor = torch.tensor(label_vec, dtype=torch.float32)
            return features_tensor, labels_tensor

        else:
            # Test phase: return rec_id to help mapping predictions later if needed,
            # but standard pytorch loaders usually just need inputs.
            # We will return dummy labels to keep signature consistent or just features.
            # Returning (features, rec_id) is common for inference.
            return features_tensor, torch.tensor(rec_id, dtype=torch.long)


def get_dataloaders(load_cached_data=True):
    """
    Prepares DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached feature processing.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    set_seed(config.RANDOM_SEED)

    # 1. Load Metadata
    train_meta = pd.read_csv(config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(config.VAL_METADATA_PATH)
    test_meta = pd.read_csv(config.TEST_METADATA_PATH)

    # 2. Load Features
    features_df = load_histogram_features(load_cached_data=load_cached_data)

    # Identify feature columns (feat_0 to feat_99)
    feature_cols = [c for c in features_df.columns if c.startswith("feat_")]

    # 3. Merge Metadata with Features
    # Inner join ensures we only use records that have both metadata and features
    train_df = train_meta.merge(features_df, on="rec_id", how="inner")
    val_df = val_meta.merge(features_df, on="rec_id", how="inner")
    test_df = test_meta.merge(features_df, on="rec_id", how="inner")

    # 4. Create Datasets
    train_dataset = BirdDataset(train_df, feature_cols, phase="train")
    val_dataset = BirdDataset(val_df, feature_cols, phase="val")
    test_dataset = BirdDataset(test_df, feature_cols, phase="test")

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
