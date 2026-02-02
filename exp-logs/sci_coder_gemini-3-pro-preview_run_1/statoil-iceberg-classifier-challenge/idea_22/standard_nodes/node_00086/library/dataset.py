import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import setup_logger

# Initialize logger
logger = setup_logger()


class IcebergDataset(Dataset):
    def __init__(self, images, angles, labels=None, ids=None, transform=None):
        """
        Args:
            images (np.ndarray): Shape (N, 75, 75, 3) - Normalized and stacked bands.
            angles (np.ndarray): Shape (N,) - Normalized incidence angles.
            labels (np.ndarray, optional): Shape (N,) - Target labels (0 or 1).
            ids (np.ndarray, optional): Shape (N,) - Image IDs.
            transform (albumentations.Compose): Transforms to apply.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve data
        image = self.images[idx]
        angle = self.angles[idx]

        # Apply transforms (Augmentation + Resize + ToTensor)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback if no transform provided (shouldn't happen in pipeline)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        # Prepare return values
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        sample = {
            "image": image,
            "inc_angle": angle_tensor,
        }

        if self.ids is not None:
            sample["id"] = self.ids[idx]

        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            sample["label"] = label_tensor

        return sample


def get_transforms(phase="train"):
    """
    Returns the Albumentations transform pipeline.
    """
    transforms = []

    # 1. Upsampling to 224x224 using Bicubic Interpolation
    transforms.append(
        A.Resize(
            height=Config.IMAGE_SIZE,
            width=Config.IMAGE_SIZE,
            interpolation=cv2.INTER_CUBIC,
        )
    )

    # 2. Augmentations (Train only)
    if phase == "train":
        transforms.extend(
            [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)]
        )

    # 3. Convert to Tensor
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


def _process_json_data(json_path, is_train=True):
    """
    Internal helper to process raw JSON into numpy arrays.
    Handles 'na' imputation and independent band normalization.
    """
    logger.info(f"Processing raw data from {json_path}...")

    with open(json_path, "r") as f:
        data = json.load(f)

    ids = []
    band_1 = []
    band_2 = []
    angles = []
    labels = []

    for item in data:
        ids.append(item["id"])
        band_1.append(item["band_1"])
        band_2.append(item["band_2"])
        angles.append(item["inc_angle"])
        if is_train:
            labels.append(item["is_iceberg"])

    # Convert to numpy
    ids = np.array(ids)
    # Bands are lists of floats, reshape to (N, 75, 75)
    b1 = np.array(band_1).reshape(-1, 75, 75)
    b2 = np.array(band_2).reshape(-1, 75, 75)
    labels = np.array(labels) if is_train else None

    # --- Angle Handling ---
    # Replace 'na' with NaN first to compute stats
    angles_fixed = []
    for a in angles:
        if a == "na":
            angles_fixed.append(np.nan)
        else:
            angles_fixed.append(float(a))
    angles_np = np.array(angles_fixed)

    # --- Statistics Calculation & Normalization ---
    stats_path = os.path.join(Config.WORKING_DIR, "stats.json")

    if is_train:
        # Compute stats from training data
        valid_angles = angles_np[~np.isnan(angles_np)]
        angle_mean = float(np.mean(valid_angles))
        angle_min = float(np.min(valid_angles))
        angle_max = float(np.max(valid_angles))

        b1_min = float(np.min(b1))
        b1_max = float(np.max(b1))
        b2_min = float(np.min(b2))
        b2_max = float(np.max(b2))

        stats = {
            "angle_mean": angle_mean,
            "angle_min": angle_min,
            "angle_max": angle_max,
            "b1_min": b1_min,
            "b1_max": b1_max,
            "b2_min": b2_min,
            "b2_max": b2_max,
        }

        with open(stats_path, "w") as f:
            json.dump(stats, f)
        logger.info(f"Saved statistics to {stats_path}")

    else:
        # Load stats for test data
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"Stats file not found at {stats_path}. Process training data first."
            )
        with open(stats_path, "r") as f:
            stats = json.load(f)

        angle_mean = stats["angle_mean"]
        angle_min = stats["angle_min"]
        angle_max = stats["angle_max"]
        b1_min = stats["b1_min"]
        b1_max = stats["b1_max"]
        b2_min = stats["b2_min"]
        b2_max = stats["b2_max"]

    # Impute missing angles
    angles_np[np.isnan(angles_np)] = angle_mean

    # Normalize Angles (Min-Max to 0-1)
    # Avoid division by zero if max == min
    angle_denom = angle_max - angle_min if angle_max > angle_min else 1.0
    angles_norm = (angles_np - angle_min) / angle_denom

    # Normalize Bands (Min-Max)
    b1_norm = (b1 - b1_min) / (b1_max - b1_min)
    b2_norm = (b2 - b2_min) / (b2_max - b2_min)

    # Create Band 3 (Average of Normalized Bands)
    b3_norm = (b1_norm + b2_norm) / 2.0

    # Stack to (N, 75, 75, 3)
    # Stack along the last axis
    images = np.stack([b1_norm, b2_norm, b3_norm], axis=-1).astype(np.float32)
    angles_norm = angles_norm.astype(np.float32)

    return images, angles_norm, labels, ids


def load_cached_data(is_train=True, load_cache=True):
    """
    Loads data from cache or processes from scratch.
    """
    prefix = "train" if is_train else "test"
    cache_dir = Config.WORKING_DIR

    files = {
        "images": os.path.join(cache_dir, f"{prefix}_images.npy"),
        "angles": os.path.join(cache_dir, f"{prefix}_angles.npy"),
        "ids": os.path.join(cache_dir, f"{prefix}_ids.npy"),
    }
    if is_train:
        files["labels"] = os.path.join(cache_dir, f"{prefix}_labels.npy")

    # Check cache
    if load_cache and all(os.path.exists(p) for p in files.values()):
        logger.info(f"Loading {prefix} data from cache...")
        images = np.load(files["images"])
        angles = np.load(files["angles"])
        ids = np.load(files["ids"])
        labels = np.load(files["labels"]) if is_train else None
        return images, angles, labels, ids

    # Process
    json_path = Config.TRAIN_JSON if is_train else Config.TEST_JSON
    images, angles, labels, ids = _process_json_data(json_path, is_train)

    # Save cache
    logger.info(f"Caching {prefix} data...")
    np.save(files["images"], images)
    np.save(files["angles"], angles)
    np.save(files["ids"], ids)
    if is_train:
        np.save(files["labels"], labels)

    return images, angles, labels, ids


def get_dataloaders(fold=None, phase="calibration", load_cache=True):
    """
    Factory function to create dataloaders.

    Args:
        fold (int, optional): Fold index for CV (0-4). If None and phase='calibration', uses fixed split.
        phase (str): 'calibration' (Train/Val), 'production' (Full Train), or 'test'.
        load_cache (bool): Whether to use cached numpy arrays.

    Returns:
        If phase == 'test': test_loader
        If phase == 'production': full_train_loader
        If phase == 'calibration': train_loader, val_loader
    """

    if phase == "test":
        # Load test data
        images, angles, _, ids = load_cached_data(is_train=False, load_cache=load_cache)

        dataset = IcebergDataset(
            images=images,
            angles=angles,
            labels=None,
            ids=ids,
            transform=get_transforms(phase="test"),
        )

        return DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

    # --- Training Data Handling ---
    # Load all training data
    all_images, all_angles, all_labels, all_ids = load_cached_data(
        is_train=True, load_cache=load_cache
    )

    if phase == "production":
        # Use 100% of data for training
        dataset = IcebergDataset(
            images=all_images,
            angles=all_angles,
            labels=all_labels,
            ids=all_ids,
            transform=get_transforms(phase="train"),
        )
        return DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

    if phase == "calibration":
        # Determine split indices
        train_indices = []
        val_indices = []

        if fold is not None:
            # Dynamic Stratified K-Fold
            logger.info(f"Using Stratified K-Fold (Fold {fold}/{Config.N_FOLDS})")
            skf = StratifiedKFold(
                n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
            )
            # We need to iterate to find the specific fold
            for i, (t_idx, v_idx) in enumerate(skf.split(all_images, all_labels)):
                if i == fold:
                    train_indices = t_idx
                    val_indices = v_idx
                    break
        else:
            # Fixed Split from Metadata
            logger.info("Using Fixed Split from Metadata")
            train_meta = pd.read_csv(Config.TRAIN_META_PATH)
            val_meta = pd.read_csv(Config.VAL_META_PATH)

            # Map IDs to indices in the loaded arrays
            # We create a lookup map: id -> index
            id_to_idx = {id_: i for i, id_ in enumerate(all_ids)}

            train_indices = [
                id_to_idx[id_] for id_ in train_meta["id"].values if id_ in id_to_idx
            ]
            val_indices = [
                id_to_idx[id_] for id_ in val_meta["id"].values if id_ in id_to_idx
            ]

        # Create subsets
        train_dataset = IcebergDataset(
            images=all_images[train_indices],
            angles=all_angles[train_indices],
            labels=all_labels[train_indices],
            ids=all_ids[train_indices],
            transform=get_transforms(phase="train"),
        )

        val_dataset = IcebergDataset(
            images=all_images[val_indices],
            angles=all_angles[val_indices],
            labels=all_labels[val_indices],
            ids=all_ids[val_indices],
            transform=get_transforms(phase="val"),
        )

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

        return train_loader, val_loader

    raise ValueError(f"Unknown phase: {phase}")
