import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import get_logger

logger = get_logger("data_loader")


def get_contralateral_path(row, lookup_dict):
    """
    Finds the file path of the contralateral image (same patient, same view, opposite laterality).
    """
    patient_id = row["patient_id"]
    view = row["view"]
    laterality = row["laterality"]

    # Determine contralateral laterality
    contra_lat = "R" if laterality == "L" else "L"

    # Key to look up
    key = (patient_id, view, contra_lat)

    if key in lookup_dict:
        return lookup_dict[key]
    return None


def prepare_metadata(df, split_name, load_cached_data=True):
    """
    Prepares metadata by adding contralateral file paths.
    Implements caching using parquet to save processing time.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"processed_{split_name}.parquet")

    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached metadata for {split_name} from {cache_path}")
        return pd.read_parquet(cache_path)

    logger.info(f"Processing metadata for {split_name}...")

    # Create lookup dictionary for the current dataframe
    # We drop duplicates on key to ensure unique mapping (handling potential retakes by picking first)
    lookup_df = df.drop_duplicates(subset=["patient_id", "view", "laterality"])
    lookup_dict = lookup_df.set_index(["patient_id", "view", "laterality"])[
        "file_path"
    ].to_dict()

    # Apply lookup to find contralateral path for each row
    df["contra_file_path"] = df.apply(
        lambda row: get_contralateral_path(row, lookup_dict), axis=1
    )

    # Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info(f"Saved processed metadata to {cache_path}")

    return df


class MammogramPairDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train", age_stats=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            transforms (albumentations.Compose): Augmentation pipeline.
            mode (str): 'train', 'val', or 'test'.
            age_stats (tuple): (mean, std) for age normalization.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode

        # Age normalization stats
        if age_stats:
            self.age_mean, self.age_std = age_stats
        else:
            # Fallback if not provided
            self.age_mean = df["age"].mean()
            self.age_std = df["age"].std()
            if pd.isna(self.age_std) or self.age_std == 0:
                self.age_std = 1.0

        # Handle NaNs in age stats just in case
        if pd.isna(self.age_mean):
            self.age_mean = 60.0
        if pd.isna(self.age_std):
            self.age_std = 10.0

    def __len__(self):
        return len(self.df)

    def load_image(self, path):
        """
        Robust image loading trying cv2 then PIL.
        Falls back to raw binary reading if standard loaders fail.
        """
        img = None
        # Try OpenCV
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        except Exception:
            pass

        # Try PIL if OpenCV failed
        if img is None:
            try:
                pil_img = Image.open(path)
                img = np.array(pil_img)
            except Exception:
                pass

        # Fallback: Raw Binary (Cite debug_lesson_29)
        if img is None:
            try:
                # Cite debug_lesson_30: Derive dimensions from Config
                h, w = Config.IMAGE_SIZE
                expected_pixels = h * w
                file_size = os.path.getsize(path)

                # Check if file is large enough to contain the expected pixels
                if file_size >= expected_pixels:
                    with open(path, "rb") as f:
                        # Cite debug_lesson_29: Seek from end to skip variable header
                        f.seek(-expected_pixels, os.SEEK_END)
                        data = f.read(expected_pixels)
                    img = np.frombuffer(data, dtype=np.uint8).reshape(h, w)
            except Exception:
                pass

        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Target Image
        target_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Target image missing: {target_path}")

        target_img = self.load_image(target_path)
        if target_img is None:
            raise ValueError(f"Failed to load target image: {target_path}")

        # 2. Load Contralateral Image
        contra_path_rel = row["contra_file_path"]
        contra_img = None

        if contra_path_rel is not None:
            contra_path = os.path.join(Config.INPUT_DIR, contra_path_rel)
            if os.path.exists(contra_path):
                contra_img = self.load_image(contra_path)
                # If load fails but file exists, that's an error
                if contra_img is None:
                    raise ValueError(
                        f"Failed to load existing contralateral image: {contra_path}"
                    )
            else:
                # File path in metadata but file missing on disk -> Fail Loudly
                raise FileNotFoundError(
                    f"Contralateral image missing on disk: {contra_path}"
                )

        # Handle shapes (ensure 2D grayscale)
        if len(target_img.shape) > 2:
            target_img = target_img[:, :, 0]

        if contra_img is not None:
            if len(contra_img.shape) > 2:
                contra_img = contra_img[:, :, 0]
        else:
            # Create placeholder if no contralateral exists for this patient
            contra_img = np.zeros_like(target_img)

        # 3. Resize
        # Resize to Config.IMAGE_SIZE (768, 768)
        target_img = cv2.resize(target_img, Config.IMAGE_SIZE)
        contra_img = cv2.resize(contra_img, Config.IMAGE_SIZE)

        # 4. Normalize Pixel Intensity (Min-Max to 0-1)
        def normalize_img(img):
            img = img.astype(np.float32)
            if img.max() > img.min():
                img = (img - img.min()) / (img.max() - img.min())
            else:
                img = img - img.min()  # Zero out
            return img

        target_img = normalize_img(target_img)
        contra_img = normalize_img(contra_img)

        # 5. Synchronized Augmentation
        if self.transforms:
            # Albumentations expects images in HWC or HW.
            # We pass them as is (HW).
            # We use 'image' for target and 'contra' (additional target) for contralateral
            augmented = self.transforms(image=target_img, contra=contra_img)
            target_img = augmented["image"]
            contra_img = augmented["contra"]

        # 6. Construct Metadata Channels
        # Age
        age = row["age"]
        if pd.isna(age):
            age = self.age_mean
        age_norm = (age - self.age_mean) / (self.age_std + 1e-7)

        # Implant
        implant = row["implant"]
        if pd.isna(implant):
            implant = 0
        implant = float(implant)

        # Create channels (H, W)
        h, w = target_img.shape
        age_channel = np.full((h, w), age_norm, dtype=np.float32)
        implant_channel = np.full((h, w), implant, dtype=np.float32)

        # 7. Stack Channels
        # Input: Image (1) + Age (1) + Implant (1) = 3 channels
        # Target Input
        target_tensor = np.stack(
            [target_img, age_channel, implant_channel], axis=0
        )  # (3, H, W)

        # Contralateral Input
        contra_tensor = np.stack(
            [contra_img, age_channel, implant_channel], axis=0
        )  # (3, H, W)

        # Convert to Torch Tensor
        target_tensor = torch.from_numpy(target_tensor).float()
        contra_tensor = torch.from_numpy(contra_tensor).float()

        # 8. Return
        if self.mode in ["train", "val"]:
            label = row["cancer"]
            return (
                target_tensor,
                contra_tensor,
                torch.tensor(label, dtype=torch.float32),
            )
        else:
            # Test mode
            prediction_id = row["prediction_id"]
            return target_tensor, contra_tensor, prediction_id


def get_transforms(mode="train"):
    """
    Returns the augmentation pipeline.
    Uses 'contra' as an additional target to ensure synchronized geometric transforms.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5
                ),
                # No photometric distortions (brightness/contrast) as pixel intensity is physical density
            ],
            additional_targets={"contra": "image"},
        )
    else:
        return A.Compose(
            [
                # No ops for validation/test, just resizing was done in __getitem__
            ],
            additional_targets={"contra": "image"},
        )


def get_dataloaders(
    train_batch_size=Config.BATCH_SIZE,
    val_batch_size=Config.BATCH_SIZE,
    load_cached_data=True,
    debug=Config.DEBUG,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
):
    """
    Factory function to create DataLoaders.
    Handles metadata loading, caching, and dataset instantiation.
    """
    # 1. Load Raw Metadata
    # We assume these files exist as per the problem description
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # 2. Compute or Load Age Stats
    # Try to load cached stats first to ensure consistency in inference-only runs
    age_stats_path = os.path.join(Config.CACHE_DIR, "age_stats.npy")
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(age_stats_path):
        age_stats = tuple(np.load(age_stats_path))
        logger.info(f"Loaded age stats from {age_stats_path}: {age_stats}")
    else:
        # Compute from training data
        age_mean = train_df["age"].mean()
        age_std = train_df["age"].std()
        age_stats = (age_mean, age_std)
        np.save(age_stats_path, np.array(age_stats))
        logger.info(f"Computed and saved age stats: {age_stats}")

    # 3. Process Metadata (Pairing)
    train_df = prepare_metadata(train_df, "train", load_cached_data)
    val_df = prepare_metadata(val_df, "val", load_cached_data)
    test_df = prepare_metadata(test_df, "test", load_cached_data)

    # 4. Debug Subsampling
    if debug:
        logger.info(f"Debug mode enabled. Subsampling to {debug_sample_size} samples.")
        train_df = train_df.iloc[:debug_sample_size]
        val_df = val_df.iloc[:debug_sample_size]
        # We usually want to debug the pipeline but keep test logic intact,
        # but for speed we can subsample test too if needed.
        test_df = test_df.iloc[:debug_sample_size]

    # 5. Create Datasets
    train_dataset = MammogramPairDataset(
        train_df, transforms=get_transforms("train"), mode="train", age_stats=age_stats
    )
    val_dataset = MammogramPairDataset(
        val_df, transforms=get_transforms("val"), mode="val", age_stats=age_stats
    )
    test_dataset = MammogramPairDataset(
        test_df, transforms=get_transforms("test"), mode="test", age_stats=age_stats
    )

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last to ensure batch norm stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    logger.info(
        f"DataLoaders created. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )
    return train_loader, val_loader, test_loader
