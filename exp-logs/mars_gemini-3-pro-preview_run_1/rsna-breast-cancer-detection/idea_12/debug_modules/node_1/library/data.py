import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import StandardScaler

# Import provided library modules
from library.config import Config
from library.utils import robust_image_loader, set_seed


class SiameseMammographyDataset(Dataset):
    """
    PyTorch Dataset for Flow-Aligned Pyramid Siamese Network.
    Yields pairs of (Target, Contralateral) images with spatially broadcasted metadata.
    """

    def __init__(self, df, transforms=None, age_scaler=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            transforms (albumentations.Compose): Synchronized transforms.
            age_scaler (StandardScaler): Fitted scaler for age normalization.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.age_scaler = age_scaler
        self.mode = mode

        # Pre-compute normalized age if scaler is provided
        if self.age_scaler:
            # Fill NaN age with mean (0.0 after scaling usually, but here we fill before)
            # We assume the scaler was fit on non-NaN data or handled externally.
            # Here we just handle the transform.
            ages = self.df["age"].fillna(self.df["age"].mean()).values.reshape(-1, 1)
            self.normalized_ages = self.age_scaler.transform(ages).flatten()
        else:
            self.normalized_ages = np.zeros(len(df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Target Image (from preprocessed .npy)
        # Format: [path_replace_/_with__]_512x512.npy
        base_name = row["file_path"].replace("/", "_").replace(".", "_")
        target_path = os.path.join(Config.PREPROCESSED_DIR, f"{base_name}_512x512.npy")

        # Load directly using numpy (Cite debug_lesson_4: Decouple Complex Data Ingestion)
        target_img = np.load(target_path)

        # 2. Load Contralateral Image
        contra_path_rel = row.get("contra_file_path")

        if (
            contra_path_rel
            and isinstance(contra_path_rel, str)
            and str(contra_path_rel).lower() != "nan"
        ):
            c_base_name = contra_path_rel.replace("/", "_").replace(".", "_")
            contra_path = os.path.join(
                Config.PREPROCESSED_DIR, f"{c_base_name}_512x512.npy"
            )

            if os.path.exists(contra_path):
                contra_img = np.load(contra_path)
            else:
                # Missing contralateral file -> Zero pad
                contra_img = np.zeros_like(target_img)
        else:
            # No contralateral pair exists
            contra_img = np.zeros_like(target_img)

        # Ensure images are 2D (H, W) for albumentations (grayscale)
        if len(target_img.shape) == 3:
            target_img = target_img[:, :, 0]
        if len(contra_img.shape) == 3:
            contra_img = contra_img[:, :, 0]

        # Handle shape mismatch (rare, but possible if contra is zero-like from a different source)
        if target_img.shape != contra_img.shape:
            contra_img = np.zeros_like(target_img)

        # 3. Synchronized Augmentation
        if self.transforms:
            # Pass both images to albumentations to ensure identical geometric transforms
            transformed = self.transforms(image=target_img, image_contra=contra_img)
            target_img = transformed["image"]
            contra_img = transformed["image_contra"]

        # 4. Normalization & Channel Expansion
        # Convert to float 0-1
        target_img = target_img.astype(np.float32) / 255.0
        contra_img = contra_img.astype(np.float32) / 255.0

        # Create Metadata Maps (Spatially Broadcasted)
        # Dimensions: (H, W)
        h, w = target_img.shape

        # Age Map
        age_val = self.normalized_ages[idx]
        age_map = np.full((h, w), age_val, dtype=np.float32)

        # Implant Map
        implant_val = 1.0 if row["implant"] == 1 else 0.0
        implant_map = np.full((h, w), implant_val, dtype=np.float32)

        # Stack Channels: (3, H, W) -> [Image, Age, Implant]
        # Note: Albumentations ToTensorV2 converts (H, W, C) to (C, H, W) usually,
        # but here we have (H, W) arrays. We stack them first.

        target_tensor = np.stack([target_img, age_map, implant_map], axis=0)
        contra_tensor = np.stack([contra_img, age_map, implant_map], axis=0)

        # Convert to torch tensor
        target_tensor = torch.from_numpy(target_tensor).float()
        contra_tensor = torch.from_numpy(contra_tensor).float()

        # 5. Prepare Output
        output = {
            "target": target_tensor,
            "contra": contra_tensor,
            "prediction_id": (
                str(row["prediction_id"])
                if "prediction_id" in row
                else f"{row['patient_id']}_{row['laterality']}"
            ),
        }

        if self.mode != "test":
            output["label"] = torch.tensor(row["cancer"], dtype=torch.float32)

        return output


def prepare_metadata(load_cached_data=True):
    """
    Loads metadata and performs pairing logic to find contralateral images.
    Caches the result to parquet.
    """
    # Define cache paths
    cache_train = os.path.join(Config.WORKING_DIR, "processed_train.parquet")
    cache_val = os.path.join(Config.WORKING_DIR, "processed_val.parquet")
    cache_test = os.path.join(Config.WORKING_DIR, "processed_test.parquet")

    # Check cache
    if (
        load_cached_data
        and os.path.exists(cache_train)
        and os.path.exists(cache_val)
        and os.path.exists(cache_test)
    ):
        print("Loading processed metadata from cache...")
        df_train = pd.read_parquet(cache_train)
        df_val = pd.read_parquet(cache_val)
        df_test = pd.read_parquet(cache_test)
        return df_train, df_val, df_test

    print("Processing metadata from scratch...")

    # Load raw metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Helper to add contralateral path
    def add_contra_path(df):
        # Create a lookup dictionary: (patient_id, view, laterality) -> file_path
        # We handle duplicates by taking the first one found
        lookup = {}
        for idx, row in df.iterrows():
            key = (row["patient_id"], row["view"], row["laterality"])
            if key not in lookup:
                lookup[key] = row["file_path"]

        contra_paths = []
        for idx, row in df.iterrows():
            # Target: (P, V, L) -> Contra: (P, V, Opposite(L))
            opp_lat = "R" if row["laterality"] == "L" else "L"
            target_key = (row["patient_id"], row["view"], opp_lat)

            if target_key in lookup:
                contra_paths.append(lookup[target_key])
            else:
                contra_paths.append(None)

        df["contra_file_path"] = contra_paths
        return df

    # Apply pairing logic
    # Note: Train and Val are split by patient, so contralateral is guaranteed to be in the same split if it exists.
    df_train = add_contra_path(df_train)
    df_val = add_contra_path(df_val)
    # Test set also contains all images for a patient usually
    df_test = add_contra_path(df_test)

    # Cache results
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df_train.to_parquet(cache_train)
    df_val.to_parquet(cache_val)
    df_test.to_parquet(cache_test)

    return df_train, df_val, df_test


def get_transforms(phase):
    """
    Returns Albumentations transforms for the given phase.
    """
    if phase == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE[0], width=Config.IMG_SIZE[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=0, p=0.5
                ),
            ],
            additional_targets={"image_contra": "image"},
        )
    else:
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE[0], width=Config.IMG_SIZE[1]),
            ],
            additional_targets={"image_contra": "image"},
        )


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Factory function to create DataLoaders.

    Args:
        load_cached_data (bool): Whether to use cached metadata.
        debug (bool): If True, subsamples data for rapid testing.

    Returns:
        train_loader, val_loader, test_loader
    """
    # 1. Prepare Metadata
    df_train, df_val, df_test = prepare_metadata(load_cached_data=load_cached_data)

    # Debug Subsampling
    if debug or Config.DEBUG:
        print(f"Debug mode: Subsampling to {Config.DEBUG_DATA_SIZE} samples.")
        df_train = df_train.iloc[: Config.DEBUG_DATA_SIZE]
        df_val = df_val.iloc[: Config.DEBUG_DATA_SIZE]
        df_test = df_test.iloc[: Config.DEBUG_DATA_SIZE]

    # 2. Fit Age Scaler
    # We fit only on training data to avoid leakage
    print("Fitting Age Scaler...")
    age_scaler = StandardScaler()
    # Handle NaNs in age for fitting
    train_ages = df_train["age"].fillna(df_train["age"].mean()).values.reshape(-1, 1)
    age_scaler.fit(train_ages)

    # 3. Create Datasets
    train_dataset = SiameseMammographyDataset(
        df_train,
        transforms=get_transforms("train"),
        age_scaler=age_scaler,
        mode="train",
    )

    val_dataset = SiameseMammographyDataset(
        df_val, transforms=get_transforms("val"), age_scaler=age_scaler, mode="val"
    )

    test_dataset = SiameseMammographyDataset(
        df_test, transforms=get_transforms("test"), age_scaler=age_scaler, mode="test"
    )

    # 4. Create DataLoaders
    # Cite debug_lesson_9: Disable pin_memory to Resolve Data Loader Initialization OOMs
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
        f"DataLoaders created. Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}"
    )

    return train_loader, val_loader, test_loader
