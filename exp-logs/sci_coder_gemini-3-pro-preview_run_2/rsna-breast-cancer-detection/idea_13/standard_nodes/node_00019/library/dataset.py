import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config
from library.image_utils import generate_tri_spectral_tensor


def process_tabular_features(df: pd.DataFrame) -> np.ndarray:
    """
    Generates a consistent tabular feature vector for the dataset.
    Features: Age, Implant, Laterality (One-Hot), View (One-Hot).
    Output dimension: 10
    """
    # 1. Age: Normalize (approx 0-1 range based on 25-90)
    # Fill NaN with mean approx 58.0
    age = df["age"].fillna(58.0).values
    age_norm = (age - 25.0) / 65.0  # Min 26, Max 89
    age_norm = np.clip(age_norm, 0.0, 1.0)

    # 2. Implant: Binary
    implant = df["implant"].fillna(0).astype(int).values

    # 3. Laterality: One-Hot [L, R]
    lat_l = (df["laterality"] == "L").astype(int).values
    lat_r = (df["laterality"] == "R").astype(int).values

    # 4. View: One-Hot [CC, MLO, AT, LM, ML, LMO]
    # We explicitly define the views to ensure column consistency across splits
    views = ["CC", "MLO", "AT", "LM", "ML", "LMO"]
    view_feats = []
    for v in views:
        view_feats.append((df["view"] == v).astype(int).values)

    # Stack features
    # Shape: (N, 1 + 1 + 2 + 6) = (N, 10)
    features = np.stack([age_norm, implant, lat_l, lat_r] + view_feats, axis=1)
    return features.astype(np.float32)


def get_metadata(mode: str, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads metadata for the specified mode ('train', 'val', 'test').
    Implements caching for the processed dataframe to satisfy deterministic processing requirements.
    """
    # Determine source file
    if mode == "train":
        src_path = Config.TRAIN_METADATA_PATH
    elif mode == "val":
        src_path = Config.VAL_METADATA_PATH
    elif mode == "test":
        src_path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Determine cache path
    cache_filename = f"processed_metadata_{mode}.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If load fails, proceed to compute
            pass

    # 2. Compute (Load Raw)
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Metadata file not found: {src_path}")

    df = pd.read_csv(src_path)

    # Ensure prediction_id exists for test set if not present (defensive)
    if mode == "test" and "prediction_id" not in df.columns:
        df["prediction_id"] = df["patient_id"].astype(str) + "_" + df["laterality"]

    # 3. Save Cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


class BreastCancerDataset(Dataset):
    def __init__(self, mode: str = "train", load_cached_data: bool = True):
        """
        Args:
            mode: 'train', 'val', or 'test'.
            load_cached_data: Whether to use cached metadata.
        """
        self.mode = mode
        self.df = get_metadata(mode, load_cached_data)

        # Precompute Tabular Features
        # We do this once at initialization to avoid overhead in __getitem__
        self.tabular_features = process_tabular_features(self.df)

        # Precompute Labels (if available)
        if "cancer" in self.df.columns:
            self.labels = self.df["cancer"].values.astype(np.float32)
        else:
            self.labels = np.zeros(len(self.df), dtype=np.float32)

        # Precompute Full File Paths
        # The metadata contains relative paths (e.g., "train_images/...")
        # We prepend the INPUT_DIR
        self.file_paths = (
            self.df["file_path"]
            .apply(lambda x: os.path.join(Config.INPUT_DIR, x))
            .values
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Image Processing
        img_path = self.file_paths[idx]

        # Generate Tri-Spectral Tensor (H, W, 3)
        # Channels: Linear, CLAHE, Gamma
        img_tensor = generate_tri_spectral_tensor(
            img_path,
            size=Config.IMAGE_SIZE,
            gamma=Config.GAMMA_VALUE,
            clahe_clip=Config.CLAHE_CLIP_LIMIT,
            clahe_grid=Config.CLAHE_TILE_GRID_SIZE,
        )

        # Convert to PyTorch format (C, H, W)
        img_tensor = np.transpose(img_tensor, (2, 0, 1))
        img_tensor = torch.from_numpy(img_tensor)

        # 2. Tabular Features
        tab_vector = torch.from_numpy(self.tabular_features[idx])

        # 3. Label
        label = torch.tensor(self.labels[idx])

        return img_tensor, tab_vector, label
