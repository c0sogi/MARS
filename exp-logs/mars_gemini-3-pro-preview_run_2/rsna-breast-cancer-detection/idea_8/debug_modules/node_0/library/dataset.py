import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# ==========================================
# Constants & Mappings
# ==========================================
# Normalization constants (ImageNet defaults)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Mappings for categorical variables
VIEW_MAP = {"CC": 0, "MLO": 1, "ML": 2, "LM": 2, "AT": 2, "LMO": 2}
DENSITY_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}
# Machine IDs found in analysis: [29, 48, 49, 93, 170, 190, 197, 210, 216, 21]
# We map them to 0-N. Any unknown machine in test will be mapped to a default.
MACHINE_ID_MAP = {
    29: 0,
    48: 1,
    49: 2,
    93: 3,
    170: 4,
    190: 5,
    197: 6,
    210: 7,
    216: 8,
    21: 9,
}


def get_transforms(mode="train"):
    """
    Returns albumentations transforms for the given mode.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE[0], width=Config.IMAGE_SIZE[1]),
                A.HorizontalFlip(p=0.5),
                # Random brightness/contrast can help with scanner variations
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE[0], width=Config.IMAGE_SIZE[1]),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]
        )


def preprocess_metadata(df, mode="train", load_cached_data=True):
    """
    Preprocesses metadata: fills NaNs, maps categoricals, and caches to Parquet.

    Args:
        df (pd.DataFrame): Raw metadata dataframe.
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Processed dataframe.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"processed_{mode}.parquet")

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached {mode} metadata from {cache_path}...")
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print(f"Processing {mode} metadata...")

    # 2. Process Data
    # Handle Age: Fill NaN with median (approx 58)
    df["age"] = df["age"].fillna(58.0)
    # Normalize Age (simple scaling based on dataset stats)
    df["age_norm"] = (df["age"] - 58.0) / 10.0

    # Handle Implant: Fill NaN with 0, ensure int
    if "implant" in df.columns:
        df["implant"] = df["implant"].fillna(0).astype(int)
    else:
        df["implant"] = 0

    # Handle View: Map to integers
    df["view_enc"] = df["view"].map(VIEW_MAP).fillna(2).astype(int)

    # Handle Machine ID: Map to integers
    df["machine_enc"] = (
        df["machine_id"].map(MACHINE_ID_MAP).fillna(len(MACHINE_ID_MAP)).astype(int)
    )

    # Handle Targets (only for train/val)
    if mode in ["train", "val"]:
        # Cancer
        df["cancer"] = df["cancer"].fillna(0).astype(int)

        # BIRADS (0-2). Fill NaN with -1 (ignore index)
        if "BIRADS" in df.columns:
            df["birads_enc"] = df["BIRADS"].fillna(-1).astype(int)
        else:
            df["birads_enc"] = -1

        # Density (A-D -> 0-3). Fill NaN with -1 (ignore index)
        if "density" in df.columns:
            df["density_enc"] = df["density"].map(DENSITY_MAP).fillna(-1).astype(int)
        else:
            df["density_enc"] = -1

    # 3. Save to Cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


class MammographyDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Processed metadata.
            transforms (albumentations.Compose): Image transformations.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.input_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- 1. Load Image ---
        # Construct full path
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Robust loading using byte stream
        try:
            with open(img_path, "rb") as f:
                file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            # Try loading as grayscale
            image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

            if image is None:
                raise ValueError("Decode failed")

        except Exception:
            # Fallback for corrupt/missing images: Black image
            # print(f"Warning: Could not load {img_path}. Using black image.")
            image = np.zeros(Config.IMAGE_SIZE, dtype=np.uint8)

        # Convert to RGB (EfficientNet expects 3 channels)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Basic to tensor if no transforms provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # --- 2. Prepare Metadata Vector ---
        # Vector: [Age, Implant, View_OneHot(3), Machine_OneHot(11)]
        # However, to keep it compatible with a simple Linear layer,
        # we will pass specific features and let the model handle embedding/fusion.
        # Here we construct a numerical vector of shape (N, )

        # For this implementation, we'll return a fixed size vector containing:
        # [age_norm, implant, view_enc, machine_enc]
        # The model's MLP will need to interpret these.
        # Alternatively, we can do one-hot here. Let's do a simple float vector.

        meta_vec = torch.tensor(
            [
                row["age_norm"],
                float(row["implant"]),
                float(row["view_enc"]),
                float(row["machine_enc"]),
            ],
            dtype=torch.float32,
        )

        # --- 3. Return Data ---
        if self.mode in ["train", "val"]:
            # Targets
            targets = {
                "cancer": torch.tensor(row["cancer"], dtype=torch.float32),
                "birads": torch.tensor(row["birads_enc"], dtype=torch.long),
                "density": torch.tensor(row["density_enc"], dtype=torch.long),
            }
            return image, meta_vec, targets
        else:
            # For inference, we might need prediction_id to aggregate later,
            # but the DataLoader usually just returns tensors.
            # We'll rely on the order being preserved or pass ID if needed.
            return image, meta_vec


def get_dataloaders(
    train_metadata_path=Config.TRAIN_METADATA_PATH,
    val_metadata_path=Config.VAL_METADATA_PATH,
    test_metadata_path=Config.TEST_METADATA_PATH,
    load_cached_data=True,
    debug=Config.DEBUG,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
):
    """
    Creates DataLoaders for train, val, and test sets.

    Args:
        train_metadata_path (str): Path to train CSV.
        val_metadata_path (str): Path to val CSV.
        test_metadata_path (str): Path to test CSV.
        load_cached_data (bool): Use parquet cache.
        debug (bool): If True, subsample data.
        debug_sample_size (int): Number of samples for debug.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # --- Load DataFrames ---
    if os.path.exists(train_metadata_path):
        train_df = pd.read_csv(train_metadata_path)
        val_df = pd.read_csv(val_metadata_path)

        if debug:
            train_df = train_df.iloc[:debug_sample_size]
            val_df = val_df.iloc[:debug_sample_size]

        # Process Metadata
        train_df = preprocess_metadata(
            train_df, mode="train", load_cached_data=load_cached_data
        )
        val_df = preprocess_metadata(
            val_df, mode="val", load_cached_data=load_cached_data
        )

        # Datasets
        train_dataset = MammographyDataset(
            train_df, transforms=get_transforms("train"), mode="train"
        )
        val_dataset = MammographyDataset(
            val_df, transforms=get_transforms("val"), mode="val"
        )

        # DataLoaders
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
        )
    else:
        train_loader = None
        val_loader = None

    # --- Test Loader ---
    if os.path.exists(test_metadata_path):
        test_df = pd.read_csv(test_metadata_path)
        if debug:
            test_df = test_df.iloc[:debug_sample_size]

        test_df = preprocess_metadata(
            test_df, mode="test", load_cached_data=load_cached_data
        )

        test_dataset = MammographyDataset(
            test_df, transforms=get_transforms("test"), mode="test"
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
    else:
        test_loader = None

    return train_loader, val_loader, test_loader
