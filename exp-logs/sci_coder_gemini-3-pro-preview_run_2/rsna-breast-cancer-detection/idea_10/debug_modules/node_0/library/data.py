import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import load_image

# ==========================================
# Mappings & Encoders
# ==========================================
# Fixed mappings to ensure consistency across Train/Val/Test
VIEW_MAP = {"CC": 0, "MLO": 1, "AT": 2, "LM": 3, "ML": 4, "LMO": 5}
MACHINE_ID_MAP = {
    29: 0,
    21: 1,
    210: 2,
    49: 3,
    48: 4,
    93: 5,
    170: 6,
    216: 7,
    190: 8,
    197: 9,
}
DENSITY_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}


def preprocess_metadata(df_path, mode, load_cached_data=True):
    """
    Loads and processes metadata with caching mechanism.
    """
    filename = os.path.basename(df_path).replace(".csv", "_processed.parquet")
    cache_path = os.path.join(Config.WORKING_DIR, filename)

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            pass  # Fallback to processing

    # 2. Process from scratch
    if not os.path.exists(df_path):
        # If original file doesn't exist (e.g. test.csv in some environments), return empty or raise
        if mode == "test":
            # In submission environment, test.csv is at Config.INPUT_DIR/test.csv
            # But Config.TEST_META_PATH points to metadata/test.csv
            # We assume the metadata generation script ran correctly.
            pass

    df = pd.read_csv(df_path)

    # --- Feature Engineering ---

    # Encode View
    # Fill unknown views with a safe default (e.g., 'MLO' or new class) if any
    df["view_idx"] = df["view"].map(VIEW_MAP).fillna(0).astype(int)

    # Encode Machine ID
    # Map unknown machines to 0 or handle gracefully
    df["machine_idx"] = df["machine_id"].map(MACHINE_ID_MAP).fillna(0).astype(int)

    # Normalize Age
    # Fill missing age with median (approx 58)
    df["age"] = df["age"].fillna(58.0)
    df["age_norm"] = df["age"] / 100.0

    # Implant
    df["implant"] = df["implant"].fillna(0).astype(int)

    # --- Target Processing (Train/Val only) ---
    if mode in ["train", "val"]:
        # Cancer (Primary Target)
        df["cancer"] = df["cancer"].fillna(0).astype(float)

        # Density (Auxiliary) - Map A-D to 0-3, Missing to -1
        df["density_idx"] = df["density"].map(DENSITY_MAP).fillna(-1).astype(int)

        # BIRADS (Auxiliary) - Keep 0,1,2, Missing to -1
        # Assuming BIRADS column contains 0, 1, 2.
        df["birads_idx"] = df["BIRADS"].fillna(-1).astype(int)

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


class MammographyDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train"):
        self.df = df
        self.transforms = transforms
        self.mode = mode

        # Pre-extract columns to arrays for faster access
        self.file_paths = self.df["file_path"].values
        self.view_indices = self.df["view_idx"].values
        self.machine_indices = self.df["machine_idx"].values
        self.age_values = self.df["age_norm"].values
        self.implant_values = self.df["implant"].values

        if self.mode in ["train", "val"]:
            self.cancer_labels = self.df["cancer"].values
            self.density_labels = self.df["density_idx"].values
            self.birads_labels = self.df["birads_idx"].values

        if self.mode == "test":
            self.prediction_ids = self.df["prediction_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load Image
        # Note: file_path in metadata is relative (e.g., "train_images/...")
        # We need to join with INPUT_DIR
        rel_path = self.file_paths[idx]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Load and resize to Config.IMG_SIZE
        img = load_image(full_path, size=Config.IMG_SIZE)

        # 2. Channel Expansion (Simulated Windowing)
        # Input img is uint8 [0, 255]

        # Channel 0: Original
        c0 = img

        # Channel 1: CLAHE (Clip Limit 2.0)
        clahe_1 = cv2.createCLAHE(
            clipLimit=Config.CLAHE_CLIP_LIMITS[0], tileGridSize=(8, 8)
        )
        c1 = clahe_1.apply(img)

        # Channel 2: CLAHE (Clip Limit 4.0)
        clahe_2 = cv2.createCLAHE(
            clipLimit=Config.CLAHE_CLIP_LIMITS[1], tileGridSize=(8, 8)
        )
        c2 = clahe_2.apply(img)

        # Stack channels: (H, W, 3)
        img_stacked = np.stack([c0, c1, c2], axis=-1)

        # 3. Augmentations
        if self.transforms:
            augmented = self.transforms(image=img_stacked)
            img_tensor = augmented["image"]
        else:
            # Fallback if no transforms provided (shouldn't happen with get_dataloaders)
            img_tensor = (
                torch.from_numpy(img_stacked.transpose(2, 0, 1)).float() / 255.0
            )

        # 4. Metadata Vector
        # Construct a vector: [Age, Implant, OneHot_View(6), OneHot_Machine(10)]
        # For simplicity in this implementation, we will pass indices and continuous vars
        # and let the model handle embedding/concatenation.
        # However, to match the "MLP processing" description, we'll return a raw vector
        # that the model can process.
        # Vector: [age_norm, implant, view_idx, machine_idx]
        # The model will need to handle the embedding of view/machine internally or we pass one-hot here.
        # Given the prompt description "normalized metadata vector... processed by a parallel MLP",
        # passing the raw features is most flexible.

        metadata = torch.tensor(
            [
                self.age_values[idx],
                self.implant_values[idx],
                self.view_indices[idx],
                self.machine_indices[idx],
            ],
            dtype=torch.float32,
        )

        # 5. Targets
        if self.mode in ["train", "val"]:
            targets = {
                "cancer": torch.tensor(self.cancer_labels[idx], dtype=torch.float32),
                "density": torch.tensor(self.density_labels[idx], dtype=torch.long),
                "birads": torch.tensor(
                    self.birads_labels[idx], dtype=torch.float32
                ),  # Regression target
            }
            return img_tensor, metadata, targets

        else:
            # Test mode
            return img_tensor, metadata, self.prediction_ids[idx]


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for the specified mode.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.15,
                    rotate_limit=20,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                A.OneOf(
                    [
                        A.GridDistortion(p=0.5),
                        A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=0.5),
                    ],
                    p=0.3,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Creates DataLoaders for Train, Validation, and Test sets.

    Args:
        load_cached_data (bool): Whether to use cached metadata.
        debug (bool): If True, subsets data for quick debugging.

    Returns:
        train_loader, val_loader, test_loader
    """
    # 1. Load and Process Metadata
    train_df = preprocess_metadata(Config.TRAIN_META_PATH, "train", load_cached_data)
    val_df = preprocess_metadata(Config.VAL_META_PATH, "val", load_cached_data)
    test_df = preprocess_metadata(Config.TEST_META_PATH, "test", load_cached_data)

    # Debug Mode: Subset data
    if debug or Config.DEBUG:
        train_df = train_df.head(100)
        val_df = val_df.head(50)
        test_df = test_df.head(50)

    # 2. Create Datasets
    train_dataset = MammographyDataset(
        train_df, transforms=get_transforms("train"), mode="train"
    )

    val_dataset = MammographyDataset(
        val_df, transforms=get_transforms("val"), mode="val"
    )

    test_dataset = MammographyDataset(
        test_df, transforms=get_transforms("test"), mode="test"
    )

    # 3. Create DataLoaders
    # Use standard RandomSampler (shuffle=True) for Train as per "Stabilized" strategy
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
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
