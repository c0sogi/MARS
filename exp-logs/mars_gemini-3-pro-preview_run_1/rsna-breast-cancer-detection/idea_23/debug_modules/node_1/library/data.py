import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("data_module")


def get_transforms(phase: str):
    """
    Returns the Albumentations transform pipeline for the specified phase.
    Ensures synchronized geometric transformations for Siamese inputs.
    """
    height, width = Config.IMAGE_SIZE

    if phase == "train":
        return A.Compose(
            [
                A.Resize(height=height, width=width),
                # Geometric augmentations applied identically to both images
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5
                ),
                # Normalization is handled manually in the dataset to accommodate 3 custom channels
                ToTensorV2(),
            ],
            additional_targets={"contralateral": "image"},
        )
    else:
        return A.Compose(
            [A.Resize(height=height, width=width), ToTensorV2()],
            additional_targets={"contralateral": "image"},
        )


def process_data(load_cache=True):
    """
    Processes metadata to establish Siamese pairs and computes statistics.
    Implements caching to disk using Parquet and NPY files.
    """
    cache_train_path = os.path.join(Config.CACHE_DIR, "processed_train.parquet")
    cache_val_path = os.path.join(Config.CACHE_DIR, "processed_val.parquet")
    cache_test_path = os.path.join(Config.CACHE_DIR, "processed_test.parquet")
    cache_stats_path = os.path.join(Config.CACHE_DIR, "age_stats.npy")

    # 1. Try to load from cache
    if load_cache:
        if (
            os.path.exists(cache_train_path)
            and os.path.exists(cache_val_path)
            and os.path.exists(cache_test_path)
            and os.path.exists(cache_stats_path)
        ):
            logger.info("Loading processed data from cache...")
            try:
                df_train = pd.read_parquet(cache_train_path)
                df_val = pd.read_parquet(cache_val_path)
                df_test = pd.read_parquet(cache_test_path)
                stats = np.load(cache_stats_path, allow_pickle=True).item()
                return df_train, df_val, df_test, stats
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Reprocessing...")

    logger.info("Processing metadata from scratch...")

    # 2. Load raw metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # 3. Compute Age Statistics (from Train only)
    # Handle missing age values if any (impute with mean)
    if df_train["age"].isnull().any():
        df_train["age"] = df_train["age"].fillna(df_train["age"].mean())

    age_mean = df_train["age"].mean()
    age_std = df_train["age"].std()
    stats = {"age_mean": age_mean, "age_std": age_std}

    # 4. Generate Contralateral Paths
    # We need to find the image with: same patient_id, same view, opposite laterality
    def add_contralateral_path(df, lookup_df=None):
        if lookup_df is None:
            lookup_df = df

        # Create a lookup key: (patient_id, view, laterality) -> file_path
        # We use the lookup_df to find partners. For train/val, lookup is themselves.
        # For test, lookup is test itself.

        # Build lookup dictionary
        lookup = {}
        for idx, row in lookup_df.iterrows():
            key = (row["patient_id"], row["view"], row["laterality"])
            lookup[key] = row["file_path"]

        contralateral_paths = []
        for idx, row in df.iterrows():
            # Target key
            pid = row["patient_id"]
            view = row["view"]
            lat = row["laterality"]

            # Contralateral key
            opp_lat = "R" if lat == "L" else "L"
            contra_key = (pid, view, opp_lat)

            path = lookup.get(contra_key, None)
            contralateral_paths.append(path)

        df["contralateral_path"] = contralateral_paths
        return df

    # Apply pairing logic
    # Note: Train and Val are split by patient, so contralateral must be in the same split.
    df_train = add_contralateral_path(df_train)
    df_val = add_contralateral_path(df_val)
    df_test = add_contralateral_path(df_test)

    # 5. Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    df_train.to_parquet(cache_train_path)
    df_val.to_parquet(cache_val_path)
    df_test.to_parquet(cache_test_path)
    np.save(cache_stats_path, stats)

    logger.info(f"Data processing complete. Cache saved to {Config.CACHE_DIR}")

    return df_train, df_val, df_test, stats


class SiameseMammographyDataset(Dataset):
    def __init__(self, df, transforms, age_stats, is_test=False):
        self.df = df
        self.transforms = transforms
        self.age_mean = age_stats["age_mean"]
        self.age_std = age_stats["age_std"]
        self.is_test = is_test
        self.input_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def _load_image(self, rel_path):
        if rel_path is None or pd.isna(rel_path):
            return None

        full_path = os.path.join(self.input_dir, rel_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image file not found: {full_path}")

        # Load using cv2
        img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

        # Fallback for DICOM/Raw images (Cite debug_lesson_8)
        if img is None:
            try:
                with open(full_path, "rb") as f:
                    file_bytes = f.read()

                # Attempt 1: Decode as standard image stream (e.g. embedded JPEG)
                img = cv2.imdecode(
                    np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_UNCHANGED
                )

                # Attempt 2: Raw binary read (Cite debug_lesson_29)
                if img is None:
                    file_size = len(file_bytes)
                    # Check for 256x256 (Demo)
                    if file_size >= 65536 and file_size < 70000:
                        img = np.frombuffer(
                            file_bytes[-65536:], dtype=np.uint8
                        ).reshape(256, 256)
                    # Check for Config Size (768x768)
                    elif file_size >= (Config.IMAGE_SIZE[0] * Config.IMAGE_SIZE[1]):
                        expected = Config.IMAGE_SIZE[0] * Config.IMAGE_SIZE[1]
                        if file_size >= expected and file_size < expected + 8192:
                            img = np.frombuffer(
                                file_bytes[-expected:], dtype=np.uint8
                            ).reshape(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1])

            except Exception as e:
                logger.warning(f"Fallback loading failed for {rel_path}: {e}")

        if img is None:
            raise IOError(
                f"Failed to load image (corrupt or unsupported format): {full_path}"
            )

        # Normalize to 0-1 range
        if img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        else:
            img = img.astype(np.float32) / 255.0

        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Images
        target_img = self._load_image(row["file_path"])
        contra_img = self._load_image(row["contralateral_path"])

        # Handle missing contralateral image (physically missing)
        # "substitute a zero-tensor"
        has_contra = True
        if contra_img is None:
            has_contra = False
            contra_img = np.zeros_like(target_img)

        # 2. Apply Synchronized Augmentations
        # Albumentations expects HWC
        if len(target_img.shape) == 2:
            target_img = np.expand_dims(target_img, axis=-1)
        if len(contra_img.shape) == 2:
            contra_img = np.expand_dims(contra_img, axis=-1)

        # Ensure shapes match for transform
        if target_img.shape != contra_img.shape:
            # Resize contra to match target if dimensions differ (unlikely but possible)
            contra_img = cv2.resize(
                contra_img, (target_img.shape[1], target_img.shape[0])
            )
            if len(contra_img.shape) == 2:
                contra_img = np.expand_dims(contra_img, axis=-1)

        augmented = self.transforms(image=target_img, contralateral=contra_img)
        target_tensor_img = augmented["image"]  # (C, H, W) from ToTensorV2
        contra_tensor_img = augmented["contralateral"]  # (C, H, W)

        # 3. Construct 3-Channel Inputs [Image, Age, Implant]
        # Helper to build the full tensor
        def build_input(img_tensor, age, implant, is_zero_tensor=False):
            if is_zero_tensor:
                return torch.zeros(
                    (3, img_tensor.shape[1], img_tensor.shape[2]), dtype=torch.float32
                )

            # Channel 0: Image (Already tensor)
            # Channel 1: Age (Scalar -> Broadcast)
            age_norm = (age - self.age_mean) / (self.age_std + 1e-6)
            age_ch = torch.full_like(img_tensor, age_norm)

            # Channel 2: Implant (Scalar -> Broadcast)
            imp_val = 1.0 if implant == 1 else 0.0
            imp_ch = torch.full_like(img_tensor, imp_val)

            # Stack
            return torch.cat([img_tensor, age_ch, imp_ch], dim=0)

        # Get metadata
        age = row["age"] if not pd.isna(row["age"]) else self.age_mean
        implant = row["implant"] if "implant" in row else 0

        # Build inputs
        # Note: For contralateral, we use the SAME age and implant (patient level)
        input_target = build_input(
            target_tensor_img, age, implant, is_zero_tensor=False
        )
        input_contra = build_input(
            contra_tensor_img, age, implant, is_zero_tensor=not has_contra
        )

        # 4. Return
        if self.is_test:
            return input_target, input_contra, row["prediction_id"]
        else:
            label = torch.tensor(row["cancer"], dtype=torch.float32)
            return input_target, input_contra, label


def get_dataloaders(debug=False, load_cache=True):
    """
    Factory function to create dataloaders for train, val, and test.
    """
    # Process/Load Data
    df_train, df_val, df_test, stats = process_data(load_cache=load_cache)

    # Debug Mode: Subsample
    if debug:
        logger.info(f"Debug mode: Subsampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    # Create Datasets
    train_dataset = SiameseMammographyDataset(
        df_train, transforms=get_transforms("train"), age_stats=stats, is_test=False
    )

    val_dataset = SiameseMammographyDataset(
        df_val, transforms=get_transforms("val"), age_stats=stats, is_test=False
    )

    test_dataset = SiameseMammographyDataset(
        df_test, transforms=get_transforms("test"), age_stats=stats, is_test=True
    )

    # Create Loaders
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
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    logger.info(
        f"DataLoaders created. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader
