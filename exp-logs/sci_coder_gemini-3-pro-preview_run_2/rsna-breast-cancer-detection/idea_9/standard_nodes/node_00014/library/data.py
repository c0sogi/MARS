import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import StandardScaler, LabelEncoder

from library.config import Config
from library.utils import seed_everything

# ==========================================
# Metadata Processing & Caching
# ==========================================


def process_metadata(load_cached_data=True):
    """
    Loads and processes metadata. Implements caching mechanism using Parquet.

    Args:
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        tuple: (train_df, val_df, test_df, feature_cols)
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "processed_train.parquet")
    val_cache = os.path.join(cache_dir, "processed_val.parquet")
    test_cache = os.path.join(cache_dir, "processed_test.parquet")
    meta_cache = os.path.join(cache_dir, "feature_meta.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
            and os.path.exists(meta_cache)
        ):
            print("Loading processed metadata from cache...")
            train_df = pd.read_parquet(train_cache)
            val_df = pd.read_parquet(val_cache)
            test_df = pd.read_parquet(test_cache)
            feature_cols = np.load(meta_cache, allow_pickle=True).tolist()
            return train_df, val_df, test_df, feature_cols
        else:
            print("Cache not found or incomplete. Processing from scratch...")

    # 2. Process from scratch
    print("Loading raw metadata...")
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # --- Feature Engineering ---

    # 1. Handle Age (Numerical)
    # Fill missing age with median from train
    age_median = train_df["age"].median()
    train_df["age"] = train_df["age"].fillna(age_median)
    val_df["age"] = val_df["age"].fillna(age_median)
    test_df["age"] = test_df["age"].fillna(age_median)

    # Normalize Age
    scaler = StandardScaler()
    train_df["age_norm"] = scaler.fit_transform(train_df[["age"]])
    val_df["age_norm"] = scaler.transform(val_df[["age"]])
    test_df["age_norm"] = scaler.transform(test_df[["age"]])

    # 2. Categorical Features
    # We will use One-Hot Encoding for low cardinality, Label Encoding for others if needed.
    # Features: laterality (L/R), view (CC/MLO/etc), implant (0/1), machine_id

    # Map binary/simple columns
    map_lat = {"L": 0, "R": 1}
    map_implant = {0: 0, 1: 1}  # Ensure int

    for df in [train_df, val_df, test_df]:
        df["lat_enc"] = df["laterality"].map(map_lat).fillna(0).astype(int)
        df["implant_enc"] = df["implant"].fillna(0).astype(int)

    # One-Hot Encode 'view'
    # Get common views from train
    top_views = (
        train_df["view"].value_counts().index[:4].tolist()
    )  # usually CC, MLO, ML, LM

    def encode_view(df_in):
        for v in top_views:
            df_in[f"view_{v}"] = (df_in["view"] == v).astype(int)
        return df_in

    train_df = encode_view(train_df)
    val_df = encode_view(val_df)
    test_df = encode_view(test_df)

    # Machine ID - Label Encode
    # Handle unknown machine_ids in test by assigning a default
    le_machine = LabelEncoder()
    # Fit on all available machine IDs to cover as much as possible, or just train
    all_machines = (
        pd.concat([train_df["machine_id"], val_df["machine_id"], test_df["machine_id"]])
        .astype(str)
        .unique()
    )
    le_machine.fit(all_machines)

    train_df["machine_enc"] = le_machine.transform(train_df["machine_id"].astype(str))
    val_df["machine_enc"] = le_machine.transform(val_df["machine_id"].astype(str))
    test_df["machine_enc"] = le_machine.transform(test_df["machine_id"].astype(str))

    # Define Feature Columns for MLP input
    feature_cols = ["age_norm", "lat_enc", "implant_enc", "machine_enc"] + [
        f"view_{v}" for v in top_views
    ]

    # --- Target Processing (Train/Val only) ---
    # Handle missing auxiliary targets by filling with -1

    density_map = {"A": 0, "B": 1, "C": 2, "D": 3}

    for df in [train_df, val_df]:
        if "density" in df.columns:
            df["density_label"] = df["density"].map(density_map).fillna(-1).astype(int)
        if "BIRADS" in df.columns:
            df["birads_label"] = df["BIRADS"].fillna(-1).astype(float)

    # Test df doesn't have these, but we can add dummy columns for consistency if needed
    # (Dataset class handles missing cols gracefully usually, but let's be safe)

    # 3. Save to Cache
    print("Saving processed metadata to cache...")
    train_df.to_parquet(train_cache)
    val_df.to_parquet(val_cache)
    test_df.to_parquet(test_cache)
    np.save(meta_cache, np.array(feature_cols))

    return train_df, val_df, test_df, feature_cols


# ==========================================
# Dataset Class
# ==========================================


class MammographyDataset(Dataset):
    def __init__(self, df, feature_cols, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            feature_cols (list): List of column names for auxiliary features.
            transforms (A.Compose): Albumentations transforms.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.feature_cols = feature_cols
        self.transforms = transforms
        self.mode = mode
        self.input_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def read_dicom_bytes(self, rel_path):
        """
        Reads DICOM file as bytes and decodes using OpenCV.
        Robust to JPEG2000 and other compressions embedded in DICOM.
        """
        full_path = os.path.join(self.input_dir, rel_path)

        if not os.path.exists(full_path):
            # Return black image if file missing (should not happen with valid metadata)
            return np.zeros(
                (Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], 3), dtype=np.uint8
            )

        try:
            # Read file as byte stream
            with open(full_path, "rb") as f:
                bytes_data = f.read()

            # Convert to numpy array
            np_arr = np.frombuffer(bytes_data, np.uint8)

            # Decode
            img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

            if img is None:
                raise ValueError("cv2.imdecode returned None")

            return img

        except Exception as e:
            # Fallback or error logging
            # print(f"Error reading {rel_path}: {e}")
            return np.zeros(
                (Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]), dtype=np.uint8
            )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Image
        img = self.read_dicom_bytes(row["file_path"])

        # Resize if raw image is not target size (cv2.imdecode returns original size)
        if img.shape[:2] != Config.IMAGE_SIZE:
            img = cv2.resize(img, Config.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)

        # Convert to 3 channels (EfficientNet expects RGB)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # 2. Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=img)
            img = augmented["image"]
        else:
            # Basic ToTensor if no transforms provided
            img = ToTensorV2()(image=img)["image"]

        # 3. Extract Aux Features
        aux_feats = torch.tensor(
            row[self.feature_cols].values.astype(np.float32), dtype=torch.float32
        )

        # 4. Prepare Output
        sample = {
            "image": img,
            "aux_features": aux_feats,
            "patient_id": row["patient_id"],
            "image_id": row["image_id"],
        }

        # Add prediction_id for test/submission
        if "prediction_id" in row:
            sample["prediction_id"] = row["prediction_id"]
        else:
            # Construct it for train/val if needed for consistency
            sample["prediction_id"] = f"{row['patient_id']}_{row['laterality']}"

        # 5. Extract Targets (Train/Val only)
        if self.mode != "test":
            # Cancer (Binary)
            cancer = torch.tensor(row["cancer"], dtype=torch.float32)

            # BIRADS (Regression) - Masked if -1
            birads = torch.tensor(row.get("birads_label", -1), dtype=torch.float32)

            # Density (Classification) - Masked if -1
            density = torch.tensor(row.get("density_label", -1), dtype=torch.long)

            sample["targets"] = {"cancer": cancer, "birads": birads, "density": density}

        return sample


# ==========================================
# Transforms
# ==========================================


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for the specified mode.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.15, rotate_limit=20, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


# ==========================================
# Data Loaders
# ==========================================


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Prepares DataLoaders for Train, Val, and Test.

    Args:
        debug (bool): If True, subsets data for quick debugging.
        load_cached_data (bool): Whether to use cached metadata.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    seed_everything(Config.SEED)

    # 1. Process Metadata
    train_df, val_df, test_df, feature_cols = process_metadata(
        load_cached_data=load_cached_data
    )

    # 2. Debug Mode - Subset Data
    if debug:
        print(f"DEBUG MODE: Subsetting data to {Config.DEBUG_SAMPLE_SIZE} samples.")
        train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        test_df = test_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    # 3. Create Datasets
    train_dataset = MammographyDataset(
        train_df, feature_cols, transforms=get_transforms("train"), mode="train"
    )

    val_dataset = MammographyDataset(
        val_df, feature_cols, transforms=get_transforms("val"), mode="val"
    )

    test_dataset = MammographyDataset(
        test_df, feature_cols, transforms=get_transforms("test"), mode="test"
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    print(
        f"DataLoaders created: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}"
    )
    print(f"Auxiliary Features ({len(feature_cols)}): {feature_cols}")

    return train_loader, val_loader, test_loader
