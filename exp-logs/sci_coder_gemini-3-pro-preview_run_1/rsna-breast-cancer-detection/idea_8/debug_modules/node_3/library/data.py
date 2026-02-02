import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import cv2

# Import from provided library files
from library.config import Config
from library.utils import load_image


class SiameseMammographyDataset(Dataset):
    """
    PyTorch Dataset for Spatial Symmetry-Difference Siamese Network.
    Yields pairs of (Target, Contralateral) images with spatial metadata channels.
    """

    def __init__(self, df, age_stats, augment=False, img_size=768):
        self.df = df
        self.age_mean = age_stats["mean"]
        self.age_std = age_stats["std"]
        self.augment = augment
        self.img_size = img_size

        # Pre-convert columns to lists for faster access
        self.file_paths = df["file_path"].tolist()
        self.contra_paths = df["contra_file_path"].tolist()
        self.ages = df["age"].tolist()
        self.implants = df["implant"].tolist()

        # Labels are only present for train/val
        self.labels = df["cancer"].tolist() if "cancer" in df.columns else None
        self.prediction_ids = (
            df["prediction_id"].tolist() if "prediction_id" in df.columns else None
        )

    def __len__(self):
        return len(self.df)

    def _process_single_image(self, img_path, age, implant):
        """
        Loads image, resizes, and prepares components.
        Returns:
            img_tensor: Tensor of shape (1, H, W)
            age_norm: float
            implant_val: float
        """
        # 1. Load Image
        if (
            img_path
            and isinstance(img_path, str)
            and os.path.exists(os.path.join(Config.INPUT_DIR, img_path))
        ):
            full_path = os.path.join(Config.INPUT_DIR, img_path)
            try:
                img = load_image(full_path)
                # If grayscale, keep as is. If RGB, convert to Gray.
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except Exception:
                # Fallback for corrupt images (though utils.load_image fails loudly usually)
                img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        else:
            # Missing file (e.g. no contralateral view)
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        # 2. Resize
        img = cv2.resize(
            img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR
        )

        # 3. Normalize Image (0-1)
        img = img.astype(np.float32) / 255.0

        # 4. Normalize Age
        # Handle NaN age by filling with mean (0 after normalization)
        if np.isnan(age):
            age_norm = 0.0
        else:
            age_norm = (age - self.age_mean) / (self.age_std + 1e-7)

        # 5. Process Implant
        implant_val = 1.0 if implant == 1 else 0.0

        # 6. Prepare Image Tensor
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)

        return img_tensor, age_norm, implant_val

    def __getitem__(self, idx):
        # Retrieve metadata
        target_path = self.file_paths[idx]
        contra_path = self.contra_paths[idx]
        age = self.ages[idx]
        implant = self.implants[idx]

        # Process Inputs (Image only + scalars)
        target_img, age_norm, implant_val = self._process_single_image(
            target_path, age, implant
        )
        contra_img, _, _ = self._process_single_image(contra_path, age, implant)

        # Synchronized Augmentation (Image only)
        if self.augment:
            # Apply identical transforms to both images

            # Random Horizontal Flip
            if random.random() > 0.5:
                target_img = TF.hflip(target_img)
                contra_img = TF.hflip(contra_img)

            # Random Vertical Flip
            if random.random() > 0.5:
                target_img = TF.vflip(target_img)
                contra_img = TF.vflip(contra_img)

            # Random Rotation
            angle = random.uniform(-20, 20)
            target_img = TF.rotate(target_img, angle)
            contra_img = TF.rotate(contra_img, angle)

        # Construct Final 3-Channel Tensors (Post-Augmentation)
        # Channel 0: Image
        # Channel 1: Age Map (Spatially Constant)
        # Channel 2: Implant Map (Spatially Constant)

        target_tensor = torch.zeros(
            (3, self.img_size, self.img_size), dtype=torch.float32
        )
        target_tensor[0] = target_img[0]
        target_tensor[1] = age_norm
        target_tensor[2] = implant_val

        contra_tensor = torch.zeros(
            (3, self.img_size, self.img_size), dtype=torch.float32
        )
        contra_tensor[0] = contra_img[0]
        contra_tensor[1] = age_norm
        contra_tensor[2] = implant_val

        # Return format
        # X: (target, contra)
        # y: label (float for BCE)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return (target_tensor, contra_tensor), label
        else:
            # Test mode: return prediction_id as well for submission mapping
            pred_id = self.prediction_ids[idx]
            return (target_tensor, contra_tensor), pred_id


def get_age_stats(df_train, load_cached=True):
    """
    Computes or loads age statistics from the training set.
    """
    stats_path = os.path.join(Config.WORKING_DIR, "age_stats.npy")

    if load_cached and os.path.exists(stats_path):
        stats = np.load(stats_path, allow_pickle=True).item()
        return stats

    # Compute
    ages = df_train["age"].dropna()
    stats = {"mean": ages.mean(), "std": ages.std()}

    # Save
    np.save(stats_path, stats)
    return stats


def prepare_metadata(mode, load_cached_data=True):
    """
    Loads metadata and pairs images with their contralateral counterparts.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to load from parquet cache.

    Returns:
        pd.DataFrame: Processed dataframe with 'contra_file_path'.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"processed_{mode}.parquet")

    if load_cached_data and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    # Load original metadata
    if mode == "train":
        path = Config.TRAIN_METADATA
    elif mode == "val":
        path = Config.VAL_METADATA
    else:
        path = Config.TEST_METADATA

    df = pd.read_csv(path)

    # Logic to find contralateral image:
    # Same patient_id, Same view, Opposite laterality
    # We create a lookup dictionary for fast access

    # Create a unique key for each image: "patient_view_laterality"
    # But to find the pair, we need to query: "patient_view_OPPOSITE_laterality"

    # 1. Build Lookup
    # Key: (patient_id, view, laterality) -> file_path
    lookup = {}
    for idx, row in df.iterrows():
        key = (row["patient_id"], row["view"], row["laterality"])
        lookup[key] = row["file_path"]

    # 2. Find Pairs
    contra_paths = []
    for idx, row in df.iterrows():
        pid = row["patient_id"]
        view = row["view"]
        lat = row["laterality"]

        target_lat = "R" if lat == "L" else "L"
        query_key = (pid, view, target_lat)

        # Get path if exists, else None
        contra_paths.append(lookup.get(query_key, None))

    df["contra_file_path"] = contra_paths

    # Cache result
    df.to_parquet(cache_path, index=False)

    return df


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get PyTorch DataLoaders.

    Args:
        load_cached_data (bool): Whether to use cached processed metadata.

    Returns:
        train_loader, val_loader, test_loader
    """
    # 1. Prepare Metadata
    df_train = prepare_metadata("train", load_cached_data)
    df_val = prepare_metadata("val", load_cached_data)
    df_test = prepare_metadata("test", load_cached_data)

    # Debug mode support
    if Config.DEBUG:
        df_train = df_train.head(Config.DEBUG_SIZE)
        df_val = df_val.head(Config.DEBUG_SIZE)
        df_test = df_test.head(Config.DEBUG_SIZE)

    # 2. Get Age Stats (computed on Train only)
    age_stats = get_age_stats(df_train, load_cached=load_cached_data)

    # 3. Create Datasets
    train_dataset = SiameseMammographyDataset(
        df_train, age_stats, augment=True, img_size=Config.IMG_SIZE
    )

    val_dataset = SiameseMammographyDataset(
        df_val, age_stats, augment=False, img_size=Config.IMG_SIZE
    )

    test_dataset = SiameseMammographyDataset(
        df_test, age_stats, augment=False, img_size=Config.IMG_SIZE
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=False,
        drop_last=True,  # Drop last to avoid batch norm issues with size 1
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

    return train_loader, val_loader, test_loader
