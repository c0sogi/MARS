import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config

# =========================================================================
# Image Loading Utilities
# =========================================================================


def read_dicom_as_image(path, target_size=None):
    """
    Reads a DICOM file by scanning for embedded JPEG/JPEG2000 streams.
    Bypasses pydicom dependency.
    """
    if not os.path.exists(path):
        # Return a black image if file missing
        if target_size:
            return np.zeros((target_size[0], target_size[1]), dtype=np.uint8)
        return np.zeros((640, 640), dtype=np.uint8)

    with open(path, "rb") as f:
        data = f.read()

    file_bytes = np.frombuffer(data, dtype=np.uint8)

    # Attempt 1: Try decoding the whole buffer (rarely works for DICOM but good sanity check)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        # Attempt 2: Scan for JPEG/JPEG2000 headers
        # JPEG Magic: FF D8
        # JPEG2000 Magic: FF 4F (SOC) or 00 00 00 0C 6A 50 (JP2 Header)

        # We search for potential start indices
        # This is a heuristic: find all FF bytes, check if next is D8 or 4F
        # To be efficient, we use numpy

        candidates = []

        # Find 0xFF
        indices_ff = np.where(file_bytes == 0xFF)[0]

        # Check for JPEG (FF D8)
        for idx in indices_ff:
            if idx + 1 < len(file_bytes) and file_bytes[idx + 1] == 0xD8:
                # Try decoding from here
                try:
                    # We pass the buffer from this point
                    # cv2.imdecode usually handles trailing garbage well
                    candidate = cv2.imdecode(file_bytes[idx:], cv2.IMREAD_GRAYSCALE)
                    if candidate is not None:
                        candidates.append(candidate)
                except:
                    pass

        # Check for JPEG2000 Codestream (FF 4F)
        for idx in indices_ff:
            if idx + 1 < len(file_bytes) and file_bytes[idx + 1] == 0x4F:
                try:
                    candidate = cv2.imdecode(file_bytes[idx:], cv2.IMREAD_GRAYSCALE)
                    if candidate is not None:
                        candidates.append(candidate)
                except:
                    pass

        # If we found candidates, pick the largest one (pixel count)
        if candidates:
            # Sort by area
            candidates.sort(key=lambda x: x.shape[0] * x.shape[1], reverse=True)
            img = candidates[0]

    # Fallback: If still failed, return black image
    if img is None:
        if target_size:
            return np.zeros((target_size[0], target_size[1]), dtype=np.uint8)
        return np.zeros((640, 640), dtype=np.uint8)

    return img


# =========================================================================
# Metadata Processing
# =========================================================================


def process_metadata(load_cached_data=True):
    """
    Loads metadata, processes features, and caches results.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "processed_train.parquet")
    val_cache = os.path.join(cache_dir, "processed_val.parquet")
    test_cache = os.path.join(cache_dir, "processed_test.parquet")

    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    ):
        print("Loading cached metadata...")
        train_df = pd.read_parquet(train_cache)
        val_df = pd.read_parquet(val_cache)
        test_df = pd.read_parquet(test_cache)
        return train_df, val_df, test_df

    print("Processing metadata from scratch...")
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # 1. Handle Missing Values & Normalization

    # Age: Fill NaN with mean from train, normalize by 100
    mean_age = train_df["age"].mean()
    for df in [train_df, val_df, test_df]:
        df["age"] = df["age"].fillna(mean_age)
        df["age_norm"] = df["age"] / 100.0

    # Implant: Fill NaN with 0
    for df in [train_df, val_df, test_df]:
        df["implant"] = df["implant"].fillna(0).astype(int)

    # 2. Categorical Encoding

    # View
    for df in [train_df, val_df, test_df]:
        df["view_idx"] = df["view"].map(Config.VIEW_MAP).fillna(0).astype(int)

    # Density (Train/Val only usually, but we handle robustly)
    # Map A->0, B->1, C->2, D->3. Missing -> -100 (for loss) or 1 (B - mode) for input?
    # For auxiliary task target, we use -100. For input feature (if used), we might need a value.
    # The Config says Density is an Aux task, so it's a target.
    # However, if we use it as input, we can't use the target.
    # We will NOT use density as an input feature to avoid leakage, only as target.

    # Laterality: L->0, R->1
    lat_map = {"L": 0, "R": 1}
    for df in [train_df, val_df, test_df]:
        df["lat_idx"] = df["laterality"].map(lat_map).fillna(0).astype(int)

    # Machine ID: Need a consistent mapping
    # We build map from train
    unique_machines = sorted(train_df["machine_id"].dropna().unique())
    machine_map = {m: i for i, m in enumerate(unique_machines)}

    def map_machine(x):
        return machine_map.get(x, 0)  # Default to 0 if unseen

    for df in [train_df, val_df, test_df]:
        df["machine_idx"] = df["machine_id"].apply(map_machine).astype(int)

    # 3. Targets (Train/Val only)
    # BIRADS: 0, 1, 2. NaN -> -100
    # Density: A, B, C, D -> 0, 1, 2, 3. NaN -> -100

    for df in [train_df, val_df]:
        if "BIRADS" in df.columns:
            df["target_birads"] = df["BIRADS"].fillna(-100).astype(int)
        else:
            df["target_birads"] = -100

        if "density" in df.columns:
            df["target_density"] = (
                df["density"].map(Config.DENSITY_MAP).fillna(-100).astype(int)
            )
        else:
            df["target_density"] = -100

        if "cancer" in df.columns:
            df["target_cancer"] = df["cancer"].astype(float)

    # Save to cache
    train_df.to_parquet(train_cache)
    val_df.to_parquet(val_cache)
    test_df.to_parquet(test_cache)

    return train_df, val_df, test_df


# =========================================================================
# Dataset Class
# =========================================================================


class BreastCancerDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train"):
        self.df = df
        self.transforms = transforms
        self.mode = mode

        # Pre-extract columns to arrays for speed
        self.paths = df["file_path"].values
        self.patient_ids = df["patient_id"].values
        self.image_ids = df["image_id"].values

        # Meta features: Age, Implant, View, Laterality, Machine
        # We stack them into a float tensor.
        # Categoricals are passed as floats (indices).
        # The model is expected to handle them (e.g. via embedding layers or MLP).
        self.meta_features = np.stack(
            [
                df["age_norm"].values,
                df["implant"].values,
                df["view_idx"].values,
                df["lat_idx"].values,
                df["machine_idx"].values,
            ],
            axis=1,
        ).astype(np.float32)

        # Targets
        if self.mode != "test":
            self.cancer = df["target_cancer"].values.astype(np.float32)
            self.birads = df["target_birads"].values.astype(np.int64)
            self.density = df["target_density"].values.astype(np.int64)

        # For test submission
        if self.mode == "test":
            self.prediction_ids = df["prediction_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load Image
        # Construct full path
        full_path = os.path.join(Config.INPUT_DIR, self.paths[idx])
        img = read_dicom_as_image(full_path, Config.IMAGE_SIZE)

        # 2. Augmentations
        if self.transforms:
            augmented = self.transforms(image=img)
            img = augmented["image"]
        else:
            # Basic to tensor if no transforms provided (fallback)
            img = torch.from_numpy(img).float().unsqueeze(0) / 255.0

        # 3. Metadata
        meta = torch.tensor(self.meta_features[idx], dtype=torch.float32)

        result = {
            "image": img,
            "meta": meta,
            "patient_id": self.patient_ids[idx],
            "image_id": self.image_ids[idx],
        }

        # 4. Targets
        if self.mode != "test":
            result["targets"] = {
                "cancer": torch.tensor(self.cancer[idx], dtype=torch.float32).unsqueeze(
                    0
                ),  # (1,)
                "BIRADS": torch.tensor(self.birads[idx], dtype=torch.long),
                "density": torch.tensor(self.density[idx], dtype=torch.long),
            }

        if self.mode == "test":
            result["prediction_id"] = self.prediction_ids[idx]

        return result


# =========================================================================
# DataLoaders
# =========================================================================


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for Train, Val, and Test.
    """
    # 1. Get DataFrames
    train_df, val_df, test_df = process_metadata(load_cached_data=load_cached_data)

    if Config.DEBUG:
        print("DEBUG Mode: Using subset of data")
        train_df = train_df.head(200)
        val_df = val_df.head(100)
        test_df = test_df.head(100)

    # 2. Define Transforms
    # ImageNet normalization mean/std for grayscale?
    # Usually we replicate mean/std across channels or just use 0.5/0.5 for grayscale.
    # Since we are using pre-trained EfficientNet (RGB), we might need to repeat channels in model.
    # Here we just normalize to 0-1 range roughly or standard.
    # Let's use simple 0-1 scaling via ToFloat(max_value=255) implicitly in ToTensor logic or explicit Normalize.
    # Albumentations Normalize default is mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225).
    # Since we read grayscale, we can just use 0.5, 0.5 or keep it simple.
    # We will output (1, H, W). The model should handle channel expansion.

    train_transforms = A.Compose(
        [
            A.Resize(height=Config.IMAGE_SIZE[0], width=Config.IMAGE_SIZE[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            A.Normalize(mean=(0.485,), std=(0.229,)),  # Single channel normalization
            ToTensorV2(),
        ]
    )

    val_test_transforms = A.Compose(
        [
            A.Resize(height=Config.IMAGE_SIZE[0], width=Config.IMAGE_SIZE[1]),
            A.Normalize(mean=(0.485,), std=(0.229,)),
            ToTensorV2(),
        ]
    )

    # 3. Create Datasets
    train_dataset = BreastCancerDataset(
        train_df, transforms=train_transforms, mode="train"
    )
    val_dataset = BreastCancerDataset(
        val_df, transforms=val_test_transforms, mode="val"
    )
    test_dataset = BreastCancerDataset(
        test_df, transforms=val_test_transforms, mode="test"
    )

    # 4. Create DataLoaders
    # Note: We use standard RandomSampler (shuffle=True), not WeightedRandomSampler,
    # to preserve probability calibration as per strategy.

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
