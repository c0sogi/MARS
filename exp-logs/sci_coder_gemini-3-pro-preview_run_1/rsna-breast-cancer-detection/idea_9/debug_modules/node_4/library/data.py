import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import io
from PIL import Image

from library.config import Config
from library.utils import create_contralateral_lookup, set_seed

# Ensure reproducibility
set_seed(Config.SEED)


def get_age_statistics(df_train, cache_dir):
    """
    Computes or loads mean and std of age from training data.
    """
    cache_path = os.path.join(cache_dir, "age_stats.npy")

    if os.path.exists(cache_path):
        stats = np.load(cache_path, allow_pickle=True).item()
        return stats["mean"], stats["std"]

    # Compute
    # Handle missing ages if any (impute with mean of available)
    ages = df_train["age"].dropna().values
    mean_age = np.mean(ages)
    std_age = np.std(ages)

    # Save
    np.save(cache_path, {"mean": mean_age, "std": std_age})

    return mean_age, std_age


def load_image(path, size):
    """
    Loads an image, converts to grayscale, and resizes.
    Implements cascading fallbacks:
    1. Pre-processed .npy files (Primary)
    2. DICOM parsing (Fallback)
    """
    # Cite debug_lesson_4: Decouple Complex Data Ingestion via Pre-processing
    # Attempt to load pre-processed .npy file first
    try:
        # Construct the expected .npy filename from the relative path
        rel_path = os.path.relpath(path, Config.INPUT_DIR)
        flat_name = rel_path.replace("/", "_").replace(".", "_")
        npy_path = os.path.join(Config.PREPROCESSED_DIR, f"{flat_name}_512x512.npy")

        if os.path.exists(npy_path):
            img = np.load(npy_path)

            # Ensure correct size
            if img.shape[:2] != size:
                img = cv2.resize(
                    img, (size[1], size[0]), interpolation=cv2.INTER_LINEAR
                )

            # Ensure 8-bit uint8
            if img.dtype != np.uint8:
                img = img.astype(np.float32)
                img = (img - img.min()) / (img.max() - img.min()) * 255.0
                img = img.astype(np.uint8)

            return img
    except Exception:
        pass

    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    img = None

    # 1. Attempt using pydicom (if available)
    try:
        import pydicom

        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array

        # Normalize 16-bit/12-bit to 8-bit
        if img.max() > 0:
            img = img.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min()) * 255.0
        img = img.astype(np.uint8)
    except (ImportError, Exception):
        pass

    # 2. Attempt Byte Extraction (Cite debug_lesson_8)
    # Extracts embedded JPEG/JPEG2000 stream if pydicom fails/missing
    if img is None:
        try:
            with open(path, "rb") as f:
                content = f.read()

            # Signatures: JPEG 2000 (FF 4F FF 51), JPEG (FF D8)
            jp2_header = b"\xff\x4f\xff\x51"
            jpg_header = b"\xff\xd8"

            jp2_idx = content.find(jp2_header)
            jpg_idx = content.find(jpg_header)

            start_idx = -1
            if jp2_idx != -1 and jpg_idx != -1:
                start_idx = min(jp2_idx, jpg_idx)
            elif jp2_idx != -1:
                start_idx = jp2_idx
            elif jpg_idx != -1:
                start_idx = jpg_idx

            if start_idx != -1:
                stream_bytes = content[start_idx:]

                # Try cv2 first with IMREAD_UNCHANGED to handle 16-bit
                stream = np.frombuffer(stream_bytes, dtype=np.uint8)
                img = cv2.imdecode(stream, cv2.IMREAD_UNCHANGED)

                # Try PIL if cv2 failed (Cite debug_lesson_8: Cascading Fallbacks)
                if img is None:
                    try:
                        image = Image.open(io.BytesIO(stream_bytes))
                        img = np.array(image)
                    except:
                        pass

                # Normalize if loaded
                if img is not None:
                    # Handle 16-bit / 12-bit
                    if img.dtype != np.uint8:
                        img = img.astype(np.float32)
                        img = (img - img.min()) / (img.max() - img.min()) * 255.0
                        img = img.astype(np.uint8)

                    # Handle RGB
                    if len(img.shape) == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        except Exception:
            pass

    # 3. Attempt standard cv2.imread (fallback)
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Failed to load image (corrupt or unsupported): {path}")

    img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    return img


class SiameseMammographyDataset(Dataset):
    def __init__(self, df, age_mean, age_std, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.age_mean = age_mean
        self.age_std = age_std
        self.transform = transform
        self.is_test = is_test
        self.img_size = Config.IMG_SIZE

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Target Image
        target_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        try:
            target_img = load_image(target_path, self.img_size)
        except Exception as e:
            # As per requirements: Fail loudly
            raise e

        # 2. Load Contralateral Image
        contra_path_rel = row.get("contra_file_path")

        if contra_path_rel and pd.notna(contra_path_rel):
            full_contra_path = os.path.join(Config.INPUT_DIR, contra_path_rel)
            if os.path.exists(full_contra_path):
                try:
                    contra_img = load_image(full_contra_path, self.img_size)
                except:
                    # If corrupt, treat as missing (zero tensor) or fail?
                    # Prompt says: "Fail Loudly... if a file path exists in metadata but the file is unreadable."
                    raise ValueError(
                        f"Contralateral image defined but unreadable: {full_contra_path}"
                    )
            else:
                # If file doesn't exist, raise error as per Fail Loudly policy for expected files
                raise FileNotFoundError(
                    f"Contralateral file missing: {full_contra_path}"
                )
        else:
            # No pair exists
            contra_img = np.zeros(self.img_size, dtype=np.uint8)

        # 3. Augmentation (Synchronized)
        # Apply identical transforms to both images to maintain spatial correspondence
        if self.transform:
            augmented = self.transform(image=target_img, image_contra=contra_img)
            target_img = augmented["image"]
            contra_img = augmented["image_contra"]

        # 4. Normalization (Pixel) -> [0, 1]
        target_img = target_img.astype(np.float32) / 255.0
        contra_img = contra_img.astype(np.float32) / 255.0

        # 5. Metadata Channels
        # Age
        age = row["age"]
        if pd.isna(age):
            age = self.age_mean  # Simple imputation
        age_norm = (age - self.age_mean) / self.age_std
        age_map = np.full(self.img_size, age_norm, dtype=np.float32)

        # Implant
        implant = row["implant"]
        if pd.isna(implant):
            implant = 0
        implant_val = 1.0 if implant else 0.0
        implant_map = np.full(self.img_size, implant_val, dtype=np.float32)

        # 6. Stack Channels
        # Result: (H, W, 3) -> (Image, Age, Implant)
        target_tensor = np.stack([target_img, age_map, implant_map], axis=-1)
        contra_tensor = np.stack([contra_img, age_map, implant_map], axis=-1)

        # Convert to Torch Tensor (C, H, W)
        target_tensor = torch.from_numpy(target_tensor).permute(2, 0, 1).float()
        contra_tensor = torch.from_numpy(contra_tensor).permute(2, 0, 1).float()

        # 7. Label / Prediction ID
        if self.is_test:
            return target_tensor, contra_tensor, row["prediction_id"]
        else:
            # Cite debug_lesson_12: Explicitly unsqueeze scalar target to (1,)
            label = torch.tensor(row["cancer"], dtype=torch.float32).unsqueeze(0)
            return target_tensor, contra_tensor, label


def get_transforms(phase):
    """
    Returns albumentations transforms for train or val/test.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
                # No photometric augmentations as per strategy
            ],
            additional_targets={"image_contra": "image"},
        )
    else:
        # No test-time augmentation (TTA) specified
        return A.Compose([], additional_targets={"image_contra": "image"})


def process_metadata(df_path, cache_path, load_cached_data):
    """
    Loads metadata, adds contralateral paths, and caches the result.
    """
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            pass  # Fallback to recomputing

    df = pd.read_csv(df_path)

    # Create lookup
    lookup = create_contralateral_lookup(df)

    # Map lookup to dataframe
    df["contra_file_path"] = df["image_id"].map(lookup)

    # Save to cache
    df.to_parquet(cache_path, index=False)

    return df


def get_dataloaders(
    load_cached_data=True,
    debug=Config.DEBUG,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
):
    """
    Main function to prepare dataloaders.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Age Statistics
    # Load raw train to get stats (always from full train set for consistency)
    df_train_raw = pd.read_csv(Config.TRAIN_METADATA_PATH)
    age_mean, age_std = get_age_statistics(df_train_raw, Config.WORKING_DIR)

    # 2. Process Metadata (Train/Val/Test)
    train_cache = os.path.join(Config.WORKING_DIR, "processed_train.parquet")
    val_cache = os.path.join(Config.WORKING_DIR, "processed_val.parquet")
    test_cache = os.path.join(Config.WORKING_DIR, "processed_test.parquet")

    df_train = process_metadata(
        Config.TRAIN_METADATA_PATH, train_cache, load_cached_data
    )
    df_val = process_metadata(Config.VAL_METADATA_PATH, val_cache, load_cached_data)
    df_test = process_metadata(Config.TEST_METADATA_PATH, test_cache, load_cached_data)

    # Debug Subsampling
    if debug:
        df_train = df_train.sample(
            n=min(len(df_train), debug_sample_size), random_state=Config.SEED
        ).reset_index(drop=True)
        df_val = df_val.sample(
            n=min(len(df_val), debug_sample_size), random_state=Config.SEED
        ).reset_index(drop=True)
        df_test = df_test.sample(
            n=min(len(df_test), debug_sample_size), random_state=Config.SEED
        ).reset_index(drop=True)

    # 3. Datasets
    train_dataset = SiameseMammographyDataset(
        df_train, age_mean, age_std, transform=get_transforms("train"), is_test=False
    )
    val_dataset = SiameseMammographyDataset(
        df_val, age_mean, age_std, transform=get_transforms("val"), is_test=False
    )
    test_dataset = SiameseMammographyDataset(
        df_test, age_mean, age_std, transform=get_transforms("test"), is_test=True
    )

    # 4. Dataloaders
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
