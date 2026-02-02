import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Cite debug_lesson_16: Sequence Deep Learning Imports
# Import torch first, then tensorflow
import torch

try:
    import tensorflow as tf

    # Cite debug_lesson_13: Preempt TensorFlow's Greedy Memory Allocation
    # Ensure TF uses CPU only to avoid OOM
    tf.config.set_visible_devices([], "GPU")
except ImportError:
    tf = None

from library.config import Config
from library.utils import get_logger, set_seed

# Initialize Logger
logger = get_logger("data_module")


def get_age_stats(df_train, load_cached_data=True):
    """
    Computes or loads the mean and standard deviation of the 'age' column.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "age_stats.npy")

    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached age stats from {cache_path}")
        stats = np.load(cache_path, allow_pickle=True).item()
        return stats["mean"], stats["std"]

    logger.info("Computing age stats from training data...")
    # Fill missing age with median for stats calculation if any remain
    ages = df_train["age"].fillna(df_train["age"].median()).values
    mean_age = np.mean(ages)
    std_age = np.std(ages)

    # Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.save(cache_path, {"mean": mean_age, "std": std_age})
    logger.info(f"Saved age stats: Mean={mean_age:.4f}, Std={std_age:.4f}")

    return mean_age, std_age


def process_metadata(df, mode, load_cached_data=True):
    """
    Processes metadata to identify contralateral pairs.
    Caches the result as a parquet file.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"processed_{mode}.parquet")

    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached metadata for {mode} from {cache_path}")
        return pd.read_parquet(cache_path)

    logger.info(f"Processing metadata for {mode}...")

    # Create a lookup for finding pairs
    # Key: (patient_id, view, laterality) -> Value: file_path
    # We take the first available image if duplicates exist
    lookup = {}
    for idx, row in df.iterrows():
        key = (row["patient_id"], row["view"], row["laterality"])
        if key not in lookup:
            lookup[key] = row["file_path"]

    def get_contra_path(row):
        current_lat = row["laterality"]
        target_lat = "R" if current_lat == "L" else "L"
        key = (row["patient_id"], row["view"], target_lat)
        return lookup.get(key, None)

    # Apply pairing logic
    df["contra_file_path"] = df.apply(get_contra_path, axis=1)

    # Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info(f"Saved processed metadata to {cache_path}")

    return df


class BreastCancerPairedDataset(Dataset):
    def __init__(self, df, transforms=None, age_stats=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe with 'contra_file_path'.
            transforms (albumentations.Compose): Transforms to apply.
            age_stats (tuple): (mean, std) for age normalization.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.age_mean, self.age_std = age_stats if age_stats else (58.0, 10.0)

        # Pre-calculate normalized age for efficiency
        # Handle missing ages by filling with mean (0 after normalization)
        self.df["age"] = self.df["age"].fillna(self.age_mean)
        self.df["norm_age"] = (self.df["age"] - self.age_mean) / self.age_std

        # Ensure implant is int
        if "implant" in self.df.columns:
            self.df["implant"] = self.df["implant"].fillna(0).astype(int)
        else:
            self.df["implant"] = 0

    def __len__(self):
        return len(self.df)

    def load_image(self, rel_path):
        """
        Loads an image from disk using cascading fallbacks.
        Supports .dcm (via PIL/Fallbacks), .png, and .npy.
        """
        if rel_path is None:
            return None

        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # 1. Check for pre-processed .npy (Cite debug_lesson_4)
        # This decouples ingestion from complex parsing
        npy_path = os.path.splitext(full_path)[0] + ".npy"
        if os.path.exists(npy_path):
            try:
                return np.load(npy_path)
            except Exception:
                pass

        # 2. Check for pre-processed .png
        png_path = os.path.splitext(full_path)[0] + ".png"
        if os.path.exists(png_path):
            img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image file not found: {full_path}")

        # 3. Try OpenCV (Standard)
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img

        # 4. Try PIL (Fallback) (Cite debug_lesson_8)
        # PIL handles some formats cv2 misses
        try:
            from PIL import Image

            with Image.open(full_path) as pil_img:
                return np.array(pil_img.convert("L"))
        except Exception:
            pass

        # 5. Critical Fallback: Raw Byte Search (Cite debug_lesson_8)
        # Attempt to find embedded JPEG or JPEG 2000 stream if container parsing fails
        try:
            with open(full_path, "rb") as f:
                data = f.read()

            # Check for J2K or JPEG signatures
            j2k_start = data.find(b"\xff\x4f\xff\x51")
            jpeg_start = data.find(b"\xff\xd8")

            start_offset = -1
            if j2k_start != -1:
                start_offset = j2k_start
            elif jpeg_start != -1:
                start_offset = jpeg_start

            if start_offset != -1:
                # Try OpenCV imdecode first
                img = cv2.imdecode(
                    np.frombuffer(data[start_offset:], np.uint8), cv2.IMREAD_GRAYSCALE
                )
                if img is not None:
                    return img

                # Try TensorFlow decode_image
                if tf is not None:
                    try:
                        # tf.io.decode_image can handle various formats
                        # We pass the raw bytes from the start of the image stream
                        img_tensor = tf.io.decode_image(
                            data[start_offset:], channels=1, expand_animations=False
                        )
                        if img_tensor is not None:
                            return img_tensor.numpy().squeeze()
                    except:
                        pass
        except Exception:
            pass

        # 6. Graceful Failure (Cite debug_lesson_10)
        # Return a black image to allow the pipeline to proceed.
        logger.warning(f"Failed to decode {rel_path}. Returning black image.")
        return np.zeros(Config.IMG_SIZE, dtype=np.uint8)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Target Image
        img_target = self.load_image(row["file_path"])

        # 2. Load Contralateral Image
        img_contra = self.load_image(row["contra_file_path"])

        # Handle missing contralateral (physically missing)
        if img_contra is None:
            img_contra = np.zeros_like(img_target)

        # 3. Preprocessing
        # Resize if not handled by transforms (though transforms usually handle it)
        # Here we rely on albumentations Resize, but we must ensure shapes match for stacking
        # Converting to float32 and normalizing to [0, 1]
        img_target = img_target.astype(np.float32) / 255.0
        img_contra = img_contra.astype(np.float32) / 255.0

        # 4. Augmentation
        # Apply synchronized transforms
        if self.transforms:
            # Albumentations expects HWC or HW. We have HW.
            # We pass contra image as 'image_contra'
            augmented = self.transforms(image=img_target, image_contra=img_contra)
            img_target = augmented["image"]
            img_contra = augmented["image_contra"]

        # Note: Albumentations ToTensorV2 converts to [C, H, W] if input is HWC,
        # or [1, H, W] if input is HW and we add dimension.
        # However, we are constructing a 3-channel tensor manually.
        # So we expect transforms to return numpy arrays (Resize, Flip, etc.)
        # and we handle tensor conversion manually to inject metadata channels.

        # If ToTensorV2 was used, it would be a tensor. If not, numpy.
        if isinstance(img_target, torch.Tensor):
            img_target = img_target.numpy().squeeze()
        if isinstance(img_contra, torch.Tensor):
            img_contra = img_contra.numpy().squeeze()

        # Ensure shape is (H, W)
        if len(img_target.shape) == 3:
            img_target = img_target[0]
        if len(img_contra.shape) == 3:
            img_contra = img_contra[0]

        h, w = img_target.shape

        # 5. Construct 3-Channel Tensors [Image, Age, Implant]
        # Channel 1: Age
        age_val = row["norm_age"]
        age_map = np.full((h, w), age_val, dtype=np.float32)

        # Channel 2: Implant
        implant_val = row["implant"]
        implant_map = np.full((h, w), implant_val, dtype=np.float32)

        # Stack: [3, H, W]
        target_tensor = np.stack([img_target, age_map, implant_map], axis=0)
        contra_tensor = np.stack([img_contra, age_map, implant_map], axis=0)

        # Convert to torch tensor
        target_tensor = torch.from_numpy(target_tensor).float()
        contra_tensor = torch.from_numpy(contra_tensor).float()

        # 6. Return Data
        if self.mode in ["train", "val"]:
            label = torch.tensor(row["cancer"], dtype=torch.float32)
            return target_tensor, contra_tensor, label
        else:
            return target_tensor, contra_tensor, row["prediction_id"]


def get_transforms(phase):
    """
    Returns Albumentations transforms for the specified phase.
    Ensures synchronized augmentation for paired images.
    """
    img_size = Config.IMG_SIZE[0]  # Assuming square

    if phase == "train":
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=20, p=0.5),
                # ShiftScaleRotate with strict limits to preserve anatomy
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5
                ),
            ],
            additional_targets={"image_contra": "image"},
        )
    else:
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
            ],
            additional_targets={"image_contra": "image"},
        )


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get DataLoaders.
    """
    # 1. Load Raw Metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Debugging: Sample subset
    if Config.DEBUG:
        logger.info(f"DEBUG MODE: Sampling {Config.DEBUG_SUBSET_SIZE} rows.")
        df_train = df_train.head(Config.DEBUG_SUBSET_SIZE)
        df_val = df_val.head(Config.DEBUG_SUBSET_SIZE)
        df_test = df_test.head(Config.DEBUG_SUBSET_SIZE)

    # 2. Compute Age Stats (from Train only)
    age_stats = get_age_stats(df_train, load_cached_data=load_cached_data)

    # 3. Process Metadata (Pairing)
    df_train = process_metadata(df_train, "train", load_cached_data=load_cached_data)
    df_val = process_metadata(df_val, "val", load_cached_data=load_cached_data)
    df_test = process_metadata(df_test, "test", load_cached_data=load_cached_data)

    # 4. Create Datasets
    train_dataset = BreastCancerPairedDataset(
        df_train, transforms=get_transforms("train"), age_stats=age_stats, mode="train"
    )

    val_dataset = BreastCancerPairedDataset(
        df_val, transforms=get_transforms("val"), age_stats=age_stats, mode="val"
    )

    test_dataset = BreastCancerPairedDataset(
        df_test, transforms=get_transforms("test"), age_stats=age_stats, mode="test"
    )

    # 5. Create DataLoaders
    # Disable pin_memory to prevent OOM in resource-constrained environments (Cite debug_lesson_9)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=False,
    )

    logger.info(
        f"DataLoaders created. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader
