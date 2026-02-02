import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from library.config import Config
from library.utils import load_image, get_cached_data

# -------------------------------------------------------------------------
# Data Processing & Caching
# -------------------------------------------------------------------------


def compute_age_stats(df_train):
    """Computes mean and std of age from training dataframe."""
    age_mean = df_train["age"].mean()
    age_std = df_train["age"].std()
    return np.array([age_mean, age_std], dtype=np.float32)


def add_contralateral_paths(df):
    """
    Adds a 'contralateral_file_path' column to the dataframe.
    Matches samples based on patient_id and view, looking for opposite laterality.
    """
    # Create a lookup key
    df["lat_key"] = df["laterality"].map({"L": "R", "R": "L"})

    # Create a temporary lookup dataframe
    # We drop duplicates to ensure 1:1 mapping if possible, though dataset usually has unique (patient, view, lat)
    # If duplicates exist (e.g. multiple images per view), we take the first one.
    lookup = df[["patient_id", "view", "laterality", "file_path"]].drop_duplicates(
        subset=["patient_id", "view", "laterality"]
    )

    # Merge to find contralateral path
    merged = df.merge(
        lookup,
        left_on=["patient_id", "view", "lat_key"],
        right_on=["patient_id", "view", "laterality"],
        how="left",
        suffixes=("", "_contra"),
    )

    # The merge created 'file_path_contra'. We rename it.
    df["contralateral_file_path"] = merged["file_path_contra"]

    # Clean up
    df.drop(columns=["lat_key"], inplace=True)
    return df


def _process_metadata_internal():
    """
    Internal function to process metadata from scratch.
    Returns:
        dict: containing processed train, val, test dfs and age stats.
    """
    # Load raw metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # 1. Impute Missing Age
    # Calculate mean from train to fill NaNs in all sets
    train_age_mean = df_train["age"].mean()
    df_train["age"] = df_train["age"].fillna(train_age_mean)
    df_val["age"] = df_val["age"].fillna(train_age_mean)
    df_test["age"] = df_test["age"].fillna(train_age_mean)

    # 2. Compute Age Stats for Normalization (from Train only)
    age_stats = compute_age_stats(df_train)

    # 3. Add Contralateral Paths
    df_train = add_contralateral_paths(df_train)
    df_val = add_contralateral_paths(df_val)
    df_test = add_contralateral_paths(df_test)

    return {"train": df_train, "val": df_val, "test": df_test, "age_stats": age_stats}


def get_processed_data(load_cached=True):
    """
    Orchestrates caching for dataframes and stats.
    """
    cache_dir = Config.WORKING_DIR

    paths = {
        "train": os.path.join(cache_dir, "processed_train.parquet"),
        "val": os.path.join(cache_dir, "processed_val.parquet"),
        "test": os.path.join(cache_dir, "processed_test.parquet"),
        "age_stats": os.path.join(cache_dir, "age_stats.npy"),
    }

    # Check if all cache files exist
    all_cached = all(os.path.exists(p) for p in paths.values())

    if load_cached and all_cached:
        try:
            return {
                "train": pd.read_parquet(paths["train"]),
                "val": pd.read_parquet(paths["val"]),
                "test": pd.read_parquet(paths["test"]),
                "age_stats": np.load(paths["age_stats"]),
            }
        except Exception as e:
            print(f"Cache load failed: {e}. Recomputing.")

    # Compute
    data = _process_metadata_internal()

    # Save
    data["train"].to_parquet(paths["train"])
    data["val"].to_parquet(paths["val"])
    data["test"].to_parquet(paths["test"])
    np.save(paths["age_stats"], data["age_stats"])

    return data


# -------------------------------------------------------------------------
# Dataset Class
# -------------------------------------------------------------------------


class SiameseBreastCancerDataset(Dataset):
    def __init__(self, df, transform=None, age_stats=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            transform (albumentations.Compose): Augmentation pipeline.
            age_stats (np.array): [mean, std] for age normalization.
            is_test (bool): Whether this is the test set (no labels).
        """
        self.df = df
        self.transform = transform
        self.age_mean = age_stats[0] if age_stats is not None else 58.0
        self.age_std = age_stats[1] if age_stats is not None else 10.0
        self.is_test = is_test

        # Check if preprocessed directory exists
        self.preprocessed_dir = Config.PREPROCESSED_DIR
        self.use_preprocessed = os.path.exists(self.preprocessed_dir)

    def __len__(self):
        return len(self.df)

    def _construct_input(self, img, age, implant):
        """
        Constructs the 3-channel input: [Image, Age, Implant]
        img: (H, W) numpy array, 0-255
        age: scalar
        implant: scalar (0 or 1)
        Returns: (H, W, 3) numpy array, float32
        """
        h, w = img.shape

        # Normalize Image to 0-1
        img_norm = img.astype(np.float32) / 255.0

        # Normalize Age
        age_norm = (age - self.age_mean) / self.age_std

        # Create broadcasted channels
        age_channel = np.full((h, w), age_norm, dtype=np.float32)
        implant_channel = np.full((h, w), implant, dtype=np.float32)

        # Stack
        return np.stack([img_norm, age_channel, implant_channel], axis=-1)

    def _load_image_wrapper(self, rel_path):
        """Attempts to load from preprocessed npy first, then raw file."""
        if self.use_preprocessed and rel_path:
            # Construct npy filename: e.g. train_images/123/456.dcm -> train_images_123_456_dcm_512x512.npy
            npy_name = rel_path.replace("/", "_").replace(".", "_") + "_512x512.npy"
            npy_path = os.path.join(self.preprocessed_dir, npy_name)
            if os.path.exists(npy_path):
                try:
                    img = np.load(npy_path)
                    # Resize if needed (though filename suggests 512)
                    if img.shape[:2] != (Config.IMG_SIZE, Config.IMG_SIZE):
                        img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
                    return img
                except:
                    pass

        # Fallback to raw loading
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        return load_image(full_path, size=Config.IMG_SIZE)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Target Image
        target_img = self._load_image_wrapper(row["file_path"])

        # 2. Load Contralateral Image
        contra_path_rel = row.get("contralateral_file_path")
        contra_img = None

        if pd.notna(contra_path_rel):
            contra_img = self._load_image_wrapper(contra_path_rel)

        # Handle missing counterpart
        if contra_img is None:
            contra_img = np.zeros_like(target_img)

        # 3. Construct 3-Channel Inputs
        # Get scalar features
        age = row["age"]
        implant = row["implant"] if "implant" in row else 0

        target_input = self._construct_input(target_img, age, implant)
        contra_input = self._construct_input(contra_img, age, implant)

        # 4. Apply Transforms (Synchronized)
        if self.transform:
            # Albumentations 'additional_targets' ensures identical geometric transforms
            # We pass the 3-channel inputs. Albumentations treats them as images.
            augmented = self.transform(image=target_input, contra=contra_input)
            target_input = augmented["image"]
            contra_input = augmented["contra"]

        # Ensure channel-first for PyTorch (C, H, W)
        # If ToTensorV2 was used, it's already tensor. If not, convert.
        if not isinstance(target_input, torch.Tensor):
            target_input = torch.from_numpy(target_input.transpose(2, 0, 1))
            contra_input = torch.from_numpy(contra_input.transpose(2, 0, 1))

        # 5. Prepare Output
        sample = {
            "image": target_input,
            "contra_image": contra_input,
            "prediction_id": str(row["prediction_id"]) if self.is_test else "",
            "patient_id": row["patient_id"],
        }

        if not self.is_test:
            sample["label"] = torch.tensor(
                row["cancer"], dtype=torch.float32
            ).unsqueeze(0)

        return sample


# -------------------------------------------------------------------------
# Transforms
# -------------------------------------------------------------------------


def get_transforms(phase):
    """
    Returns Albumentations Compose object.
    Note: Inputs are already resized to Config.IMG_SIZE during loading.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                # ShiftScaleRotate with border_mode=0 (constant 0 padding)
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=0,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
                ToTensorV2(),
            ],
            additional_targets={"contra": "image"},
        )
    else:
        # For val/test, just convert to tensor (resizing done in loader)
        return A.Compose([ToTensorV2()], additional_targets={"contra": "image"})


# -------------------------------------------------------------------------
# Data Loader Factory
# -------------------------------------------------------------------------


def get_dataloaders(load_cached=True, debug=False):
    """
    Main entry point to get dataloaders.
    Args:
        load_cached (bool): Whether to use cached metadata.
        debug (bool): If True, subsample datasets for quick testing.
    """
    # 1. Get Processed Metadata
    data = get_processed_data(load_cached=load_cached)
    df_train = data["train"]
    df_val = data["val"]
    df_test = data["test"]
    age_stats = data["age_stats"]

    # Debug Subsampling
    if debug:
        df_train = df_train.sample(
            n=min(100, len(df_train)), random_state=Config.SEED
        ).reset_index(drop=True)
        df_val = df_val.sample(
            n=min(50, len(df_val)), random_state=Config.SEED
        ).reset_index(drop=True)
        # We don't subsample test usually to ensure submission validity, but for pure debug valid:
        # df_test = df_test.sample(n=min(50, len(df_test))).reset_index(drop=True)

    # 2. Create Datasets
    train_dataset = SiameseBreastCancerDataset(
        df_train, transform=get_transforms("train"), age_stats=age_stats, is_test=False
    )

    val_dataset = SiameseBreastCancerDataset(
        df_val, transform=get_transforms("val"), age_stats=age_stats, is_test=False
    )

    test_dataset = SiameseBreastCancerDataset(
        df_test, transform=get_transforms("test"), age_stats=age_stats, is_test=True
    )

    # 3. Create Loaders
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

    return train_loader, val_loader, test_loader
