import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from library.config import Config
from library.utils import set_seed

# Set seed for reproducibility
set_seed(Config.SEED)


def process_metadata(load_cached_data=True):
    """
    Loads, processes, and caches metadata for train, val, and test sets.
    Handles categorical encoding and numerical normalization.
    """
    cache_dir = Config.WORK_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "processed_train.parquet")
    val_cache = os.path.join(cache_dir, "processed_val.parquet")
    test_cache = os.path.join(cache_dir, "processed_test.parquet")
    meta_cache = os.path.join(cache_dir, "feature_meta.npy")

    # 1. Try Loading Cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
            and os.path.exists(meta_cache)
        ):
            print("Loading cached processed metadata...")
            train_df = pd.read_parquet(train_cache)
            val_df = pd.read_parquet(val_cache)
            test_df = pd.read_parquet(test_cache)
            feature_meta = np.load(meta_cache, allow_pickle=True).item()
            return train_df, val_df, test_df, feature_meta

    # 2. Process from Scratch
    print("Processing metadata from scratch...")

    # Load raw metadata
    train_df = pd.read_csv(Config.TRAIN_META)
    val_df = pd.read_csv(Config.VAL_META)
    test_df = pd.read_csv(Config.TEST_META)

    # Define feature groups
    cat_cols = ["site_id", "laterality", "view", "machine_id"]
    num_cols = ["age"]
    bin_cols = ["implant"]

    feature_meta = {
        "cat_dims": {},  # Number of classes for each categorical feature
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "bin_cols": bin_cols,
    }

    # -- Numerical Processing --
    # Fit scaler on Train only
    scaler = StandardScaler()

    # Fill NaNs in Age with Train mean
    age_mean = train_df["age"].mean()
    train_df["age"] = train_df["age"].fillna(age_mean)
    val_df["age"] = val_df["age"].fillna(age_mean)
    test_df["age"] = test_df["age"].fillna(age_mean)

    train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
    val_df[num_cols] = scaler.transform(val_df[num_cols])
    test_df[num_cols] = scaler.transform(test_df[num_cols])

    # -- Categorical Processing --
    for col in cat_cols:
        le = LabelEncoder()

        # Convert to string to handle mixed types/NaNs gracefully
        train_vals = train_df[col].astype(str).fillna("Unknown")
        val_vals = val_df[col].astype(str).fillna("Unknown")
        test_vals = test_df[col].astype(str).fillna("Unknown")

        # Fit on all available data to ensure all categories are covered (or handle unknown)
        # In a strict setting, we fit on train and handle unknown, but for this competition
        # knowing the closed set of IDs (like machine_id) is acceptable.
        # To be safe and robust: Fit on Train + Val + Test unique values
        all_vals = pd.concat([train_vals, val_vals, test_vals]).unique()
        le.fit(all_vals)

        train_df[f"{col}_idx"] = le.transform(train_vals)
        val_df[f"{col}_idx"] = le.transform(val_vals)
        test_df[f"{col}_idx"] = le.transform(test_vals)

        feature_meta["cat_dims"][col] = len(le.classes_)

    # -- Binary Processing --
    for col in bin_cols:
        train_df[col] = train_df[col].fillna(0).astype(int)
        val_df[col] = val_df[col].fillna(0).astype(int)
        test_df[col] = test_df[col].fillna(0).astype(int)

    # Save to cache
    print(f"Saving processed metadata to {cache_dir}...")
    train_df.to_parquet(train_cache)
    val_df.to_parquet(val_cache)
    test_df.to_parquet(test_cache)
    np.save(meta_cache, feature_meta)

    return train_df, val_df, test_df, feature_meta


class BreastCancerDataset(Dataset):
    def __init__(
        self,
        df,
        feature_meta,
        img_size=Config.IMG_SIZE,
        input_dir=Config.INPUT_DIR,
        mode="train",
    ):
        """
        Args:
            df (pd.DataFrame): Processed metadata dataframe.
            feature_meta (dict): Metadata about features (column names, vocab sizes).
            img_size (tuple): Target image size (H, W).
            input_dir (str): Root directory for images.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.feature_meta = feature_meta
        self.img_size = img_size
        self.input_dir = input_dir
        self.mode = mode

        # Pre-extract feature arrays for speed
        self.cat_cols = [f"{c}_idx" for c in feature_meta["cat_cols"]]
        self.num_cols = feature_meta["num_cols"]
        self.bin_cols = feature_meta["bin_cols"]

        self.cat_data = self.df[self.cat_cols].values.astype(np.int64)
        self.num_data = self.df[self.num_cols].values.astype(np.float32)
        self.bin_data = self.df[self.bin_cols].values.astype(np.float32)

        self.file_paths = self.df["file_path"].values

        # Labels are only available in train/val
        if self.mode != "test":
            self.labels = self.df["cancer"].values.astype(np.float32)
        else:
            self.labels = np.zeros(len(self.df), dtype=np.float32)  # Dummy labels

    def __len__(self):
        return len(self.df)

    def _load_image(self, rel_path):
        """
        Reads DICOM images using byte-level extraction with cv2.
        Handles JPEG/JPEG2000 encapsulation.
        """
        full_path = os.path.join(self.input_dir, rel_path)

        if not os.path.exists(full_path):
            # Return black image if file missing (robustness)
            return np.zeros((self.img_size[0], self.img_size[1]), dtype=np.float32)

        try:
            with open(full_path, "rb") as f:
                file_bytes = np.frombuffer(f.read(), dtype=np.uint8)

            # Attempt 1: Direct decode
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

            # Attempt 2: Find JPEG/JPEG2000 header if direct decode fails
            if img is None:
                # JPEG SOI: FF D8
                # JPEG2000 SOC: FF 4F

                # Simple search for JPEG start
                jpeg_start = -1
                for i in range(min(len(file_bytes), 4096)):  # Search first 4KB
                    if file_bytes[i] == 0xFF:
                        if i + 1 < len(file_bytes):
                            if file_bytes[i + 1] == 0xD8:  # JPEG
                                jpeg_start = i
                                break
                            elif file_bytes[i + 1] == 0x4F:  # JPEG2000
                                jpeg_start = i
                                break

                if jpeg_start != -1:
                    img = cv2.imdecode(file_bytes[jpeg_start:], cv2.IMREAD_GRAYSCALE)

            if img is None:
                raise ValueError("Could not decode image")

            # Resize
            if (img.shape[0] != self.img_size[0]) or (img.shape[1] != self.img_size[1]):
                img = cv2.resize(
                    img,
                    (self.img_size[1], self.img_size[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            # Normalize to 0-1
            img = img.astype(np.float32) / 255.0

            return img

        except Exception as e:
            # Fallback for corrupt files
            return np.zeros((self.img_size[0], self.img_size[1]), dtype=np.float32)

    def __getitem__(self, idx):
        # Image
        img = self._load_image(self.file_paths[idx])
        # Add channel dimension: (H, W) -> (1, H, W)
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        # Tabular
        cat_feats = torch.tensor(self.cat_data[idx], dtype=torch.long)
        num_feats = torch.tensor(self.num_data[idx], dtype=torch.float32)
        bin_feats = torch.tensor(self.bin_data[idx], dtype=torch.float32)

        # Combine numerical and binary for the continuous branch
        cont_feats = torch.cat([num_feats, bin_feats], dim=0)

        # Label
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return {
            "image": img_tensor,
            "categorical": cat_feats,
            "continuous": cont_feats,
            "label": label,
            "idx": idx,  # Useful for tracking
        }


def get_dataloaders(
    load_cached_data=True, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
):
    """
    Factory function to generate dataloaders.

    Args:
        load_cached_data (bool): Whether to use cached metadata.
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes.

    Returns:
        train_loader, val_loader, test_loader, feature_meta
    """
    train_df, val_df, test_df, feature_meta = process_metadata(load_cached_data)

    # Debug: Print dataset sizes
    print(
        f"Dataset Sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )

    # Create Datasets
    train_dataset = BreastCancerDataset(train_df, feature_meta, mode="train")
    val_dataset = BreastCancerDataset(val_df, feature_meta, mode="val")
    test_dataset = BreastCancerDataset(test_df, feature_meta, mode="test")

    # Create DataLoaders
    # Note: Using standard shuffle=True for train. No WeightedRandomSampler as per strategy.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches to stabilize BatchNorm
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, feature_meta
