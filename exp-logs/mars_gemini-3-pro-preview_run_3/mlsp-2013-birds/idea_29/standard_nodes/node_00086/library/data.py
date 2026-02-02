import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from skmultilearn.model_selection import IterativeStratification
from library.config import Config
from library.utils import seed_everything

# Ensure reproducibility across the module
seed_everything(Config.SEED)


def load_histograms(load_cached_data=True):
    """
    Loads the histogram features from the text file.
    Implements caching using Parquet to speed up subsequent loads.

    Args:
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        pd.DataFrame: DataFrame containing 'rec_id' and 'features' (list of floats).
    """
    cache_path = os.path.join(Config.CACHE_DIR, "histogram_features.parquet")

    if load_cached_data and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    # Parse the text file
    # Format: rec_id,feat0,feat1,...
    data = []
    if os.path.exists(Config.HISTOGRAM_FILE):
        with open(Config.HISTOGRAM_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("rec_id"):
                    continue
                parts = line.split(",")
                try:
                    rec_id = int(parts[0])
                    features = [float(x) for x in parts[1:]]
                    data.append({"rec_id": rec_id, "features": features})
                except ValueError:
                    continue

    df = pd.DataFrame(data)

    # Save to cache if data was found
    if not df.empty:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_parquet(cache_path, index=False)

    return df


def get_data_splits(load_cached_data=True):
    """
    Loads training and validation metadata, combines them, and generates
    Iterative Stratified K-Fold splits.

    Args:
        load_cached_data (bool): If True, attempts to load fold assignments from cache.

    Returns:
        pd.DataFrame: Combined metadata with a 'fold' column.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "folds.parquet")

    # Load provided metadata
    train_meta = pd.read_csv(Config.TRAIN_CSV)
    val_meta = pd.read_csv(Config.VAL_CSV)

    # Combine to perform fresh 5-fold CV
    df = pd.concat([train_meta, val_meta], ignore_index=True).reset_index(drop=True)

    if load_cached_data and os.path.exists(cache_path):
        fold_info = pd.read_parquet(cache_path)
        # Merge fold info back to df
        if "fold" in df.columns:
            df = df.drop(columns=["fold"])
        df = df.merge(fold_info, on="rec_id", how="left")
        return df

    # Prepare labels for stratification
    # Labels are space-separated strings "0 4 10"
    num_samples = len(df)
    num_classes = Config.NUM_CLASSES
    y = np.zeros((num_samples, num_classes))

    for idx, row in df.iterrows():
        if pd.notna(row["labels"]) and row["labels"] != "?":
            try:
                lbls = [int(x) for x in str(row["labels"]).split()]
                for l in lbls:
                    if 0 <= l < num_classes:
                        y[idx, l] = 1
            except ValueError:
                pass

    # Perform Iterative Stratified K-Fold
    # This ensures balanced multi-label distribution across folds
    k_fold = IterativeStratification(n_splits=Config.N_FOLDS, order=1)

    # We need dummy X for the splitter
    X = df["rec_id"].values.reshape(-1, 1)

    df["fold"] = -1

    # skmultilearn returns indices
    for fold_idx, (train_indices, val_indices) in enumerate(k_fold.split(X, y)):
        df.loc[val_indices, "fold"] = fold_idx

    # Save fold info to cache
    fold_info = df[["rec_id", "fold"]]
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    fold_info.to_parquet(cache_path, index=False)

    return df


class SpectrogramDataset(Dataset):
    def __init__(self, df, mode="train"):
        """
        Dataset for loading and processing spectrogram images.

        Args:
            df (pd.DataFrame): DataFrame containing 'file_path', 'labels', 'rec_id'.
            mode (str): 'train', 'val', or 'test'. Controls augmentation.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.img_size = Config.IMG_SIZE

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct path to spectrogram BMP
        # Metadata file_path example: essential_data/src_wavs/PC10_20090513_054500_0020.wav
        # We need to map this to: supplemental_data/spectrograms/PC10_20090513_054500_0020.bmp
        wav_path = row["file_path"]
        filename = os.path.basename(wav_path)
        bmp_filename = filename.replace(".wav", ".bmp")
        full_path = os.path.join(Config.SPECTROGRAM_DIR, bmp_filename)

        # Load Image (BMP)
        # Load as grayscale first
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            # Fallback for missing files (should not happen based on metadata check)
            img = np.zeros(self.img_size, dtype=np.uint8)

        # Resize to fixed dimensions (224x224)
        img = cv2.resize(img, self.img_size)

        # Normalize to 0-1 float
        img = img.astype(np.float32) / 255.0

        # Augmentation (Train only)
        if self.mode == "train":
            # 1. Safe-Zone Horizontal Translation (Zero-Padding)
            # Shift limit < 10% of width (e.g., 22 pixels for 224 width)
            shift_limit = int(self.img_size[1] * Config.AUG_SHIFT_LIMIT)
            shift = np.random.randint(-shift_limit, shift_limit + 1)

            if shift != 0:
                shifted_img = np.zeros_like(img)
                if shift > 0:
                    # Shift right: pad left with 0, crop right
                    shifted_img[:, shift:] = img[:, :-shift]
                else:
                    # Shift left: pad right with 0, crop left
                    shifted_img[:, :shift] = img[:, -shift:]
                img = shifted_img

            # 2. Photometric Jitter (Brightness & Contrast)
            # Brightness
            brightness = np.random.uniform(
                -Config.AUG_BRIGHTNESS_LIMIT, Config.AUG_BRIGHTNESS_LIMIT
            )
            # Contrast
            contrast = np.random.uniform(
                1 - Config.AUG_CONTRAST_LIMIT, 1 + Config.AUG_CONTRAST_LIMIT
            )

            img = img * contrast + brightness
            img = np.clip(img, 0.0, 1.0)

        # 3-Channel Rule: Replicate grayscale to RGB
        # Shape becomes (3, H, W)
        img = np.stack([img, img, img], axis=0)

        # Convert to Tensor
        img_tensor = torch.from_numpy(img).float()

        # Labels
        labels = torch.zeros(Config.NUM_CLASSES, dtype=torch.float32)
        if "labels" in row and pd.notna(row["labels"]) and row["labels"] != "?":
            try:
                lbl_indices = [int(x) for x in str(row["labels"]).split()]
                for l in lbl_indices:
                    if 0 <= l < Config.NUM_CLASSES:
                        labels[l] = 1.0
            except ValueError:
                pass

        return {"image": img_tensor, "labels": labels, "rec_id": row["rec_id"]}


class HistogramDataset(Dataset):
    def __init__(self, df, hist_df, mode="train"):
        """
        Dataset for loading Bag-of-Audio-Words histogram features.

        Args:
            df (pd.DataFrame): Metadata DataFrame.
            hist_df (pd.DataFrame): DataFrame containing 'rec_id' and 'features'.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        # Create a fast lookup map
        self.hist_map = dict(zip(hist_df["rec_id"], hist_df["features"]))
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rec_id = row["rec_id"]

        # Retrieve features
        if rec_id in self.hist_map:
            features = np.array(self.hist_map[rec_id], dtype=np.float32)
        else:
            # Fallback (should be rare)
            features = np.zeros(Config.MLP_INPUT_DIM, dtype=np.float32)

        # Augmentation (Train only)
        if self.mode == "train":
            # Feature Noise (Gaussian)
            noise = np.random.normal(0, Config.MLP_FEATURE_NOISE, features.shape)
            features = features + noise

        features_tensor = torch.from_numpy(features).float()

        # Labels
        labels = torch.zeros(Config.NUM_CLASSES, dtype=torch.float32)
        if "labels" in row and pd.notna(row["labels"]) and row["labels"] != "?":
            try:
                lbl_indices = [int(x) for x in str(row["labels"]).split()]
                for l in lbl_indices:
                    if 0 <= l < Config.NUM_CLASSES:
                        labels[l] = 1.0
            except ValueError:
                pass

        return {"features": features_tensor, "labels": labels, "rec_id": rec_id}


def get_dataloaders(fold_idx, load_cached_data=True):
    """
    Creates DataLoaders for a specific fold for both CNN and MLP streams.

    Args:
        fold_idx (int): The fold index to use for validation (0 to 4).
        load_cached_data (bool): Whether to use cached splits/histograms.

    Returns:
        dict: Dictionary containing 'train_cnn', 'val_cnn', 'train_mlp', 'val_mlp' DataLoaders.
    """
    # Load Data
    full_df = get_data_splits(load_cached_data=load_cached_data)
    hist_df = load_histograms(load_cached_data=load_cached_data)

    # Split DataFrames based on fold index
    train_df = full_df[full_df["fold"] != fold_idx].copy()
    val_df = full_df[full_df["fold"] == fold_idx].copy()

    # Initialize Datasets
    train_cnn_ds = SpectrogramDataset(train_df, mode="train")
    val_cnn_ds = SpectrogramDataset(val_df, mode="val")

    train_mlp_ds = HistogramDataset(train_df, hist_df, mode="train")
    val_mlp_ds = HistogramDataset(val_df, hist_df, mode="val")

    # Initialize Loaders
    train_cnn_loader = DataLoader(
        train_cnn_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Useful for Mixup stability
    )

    val_cnn_loader = DataLoader(
        val_cnn_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    train_mlp_loader = DataLoader(
        train_mlp_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_mlp_loader = DataLoader(
        val_mlp_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return {
        "train_cnn": train_cnn_loader,
        "val_cnn": val_cnn_loader,
        "train_mlp": train_mlp_loader,
        "val_mlp": val_mlp_loader,
    }


def get_test_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for the test set.

    Args:
        load_cached_data (bool): Whether to use cached histograms.

    Returns:
        tuple: (cnn_loader, mlp_loader)
    """
    test_df = pd.read_csv(Config.TEST_CSV)
    hist_df = load_histograms(load_cached_data=load_cached_data)

    cnn_ds = SpectrogramDataset(test_df, mode="test")
    mlp_ds = HistogramDataset(test_df, hist_df, mode="test")

    cnn_loader = DataLoader(
        cnn_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    mlp_loader = DataLoader(
        mlp_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return cnn_loader, mlp_loader
