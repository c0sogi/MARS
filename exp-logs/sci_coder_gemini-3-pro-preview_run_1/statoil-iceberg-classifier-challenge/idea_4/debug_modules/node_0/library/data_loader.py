import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from library.config import Config
from library.utils import seed_everything


class IcebergDataset(Dataset):
    """
    Custom Dataset for Iceberg vs Ship classification.
    """

    def __init__(self, images, angles, labels=None, ids=None, transform=None):
        self.images = images
        self.angles = angles
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image (75, 75, 3)
        image = self.images[idx]
        angle = self.angles[idx]

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Convert angle to tensor
        angle = torch.tensor(angle, dtype=torch.float32)

        if self.labels is not None:
            # Training/Validation mode
            label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)
            return image, angle, label
        else:
            # Test mode (return ID for submission)
            img_id = self.ids[idx]
            return image, angle, img_id


def get_transforms(phase="train"):
    """
    Returns Albumentations transforms for train, validation, or test phases.
    """
    if phase == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=0.5
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def process_json_data(json_path, is_train=True):
    """
    Reads JSON, extracts bands, stacks them, and handles metadata.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    ids = []
    band_1 = []
    band_2 = []
    inc_angles = []
    labels = []

    for item in data:
        ids.append(item["id"])
        # Reshape flattened bands to 75x75
        b1 = np.array(item["band_1"]).reshape(75, 75)
        b2 = np.array(item["band_2"]).reshape(75, 75)
        band_1.append(b1)
        band_2.append(b2)

        angle = item["inc_angle"]
        if angle == "na":
            inc_angles.append(np.nan)
        else:
            inc_angles.append(float(angle))

        if is_train:
            labels.append(item["is_iceberg"])

    # Stack bands to create (N, 75, 75) arrays
    b1_stack = np.stack(band_1)
    b2_stack = np.stack(band_2)
    # Create 3rd band: Mean of Band 1 and Band 2
    b3_stack = (b1_stack + b2_stack) / 2.0

    # Stack channels to create (N, 75, 75, 3)
    images = np.stack([b1_stack, b2_stack, b3_stack], axis=-1)

    return {
        "ids": np.array(ids),
        "images": images,
        "angles": np.array(inc_angles),
        "labels": np.array(labels) if is_train else None,
    }


def load_and_process_data(load_cached_data=True):
    """
    Main data processing function with caching mechanism.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    files = {
        "train_images": os.path.join(cache_dir, "train_images.npy"),
        "train_angles": os.path.join(cache_dir, "train_angles.npy"),
        "train_labels": os.path.join(cache_dir, "train_labels.npy"),
        "train_ids": os.path.join(cache_dir, "train_ids.npy"),
        "test_images": os.path.join(cache_dir, "test_images.npy"),
        "test_angles": os.path.join(cache_dir, "test_angles.npy"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
    }

    # Check if all cache files exist
    if load_cached_data and all(os.path.exists(f) for f in files.values()):
        print("Loading cached data from", cache_dir)
        data = {k: np.load(v, allow_pickle=True) for k, v in files.items()}
        return data

    print("Processing data from scratch...")

    # Process Train and Test raw data
    train_raw = process_json_data(Config.TRAIN_JSON, is_train=True)
    test_raw = process_json_data(Config.TEST_JSON, is_train=False)

    # --- Incidence Angle Imputation ---
    train_angles = train_raw["angles"]
    # Calculate mean from valid training angles only
    valid_mask = ~np.isnan(train_angles)
    angle_mean = np.mean(train_angles[valid_mask])

    # Fill NaNs in train and test
    train_angles[np.isnan(train_angles)] = angle_mean
    test_angles = test_raw["angles"]
    test_angles[np.isnan(test_angles)] = angle_mean

    # --- Incidence Angle Normalization ---
    scaler = StandardScaler()
    # Fit on train, transform both
    train_angles = scaler.fit_transform(train_angles.reshape(-1, 1)).flatten()
    test_angles = scaler.transform(test_angles.reshape(-1, 1)).flatten()

    # --- Image Scaling (Min-Max to [0, 1]) ---
    train_imgs = train_raw["images"]
    test_imgs = test_raw["images"]

    # Compute global min/max from training data
    global_min = train_imgs.min()
    global_max = train_imgs.max()

    # Avoid division by zero
    if global_max == global_min:
        global_max += 1e-6

    # Scale to [0, 1]
    train_imgs = (train_imgs - global_min) / (global_max - global_min)
    test_imgs = (test_imgs - global_min) / (global_max - global_min)

    # Clip to ensure bounds (especially for test data)
    train_imgs = np.clip(train_imgs, 0, 1).astype(np.float32)
    test_imgs = np.clip(test_imgs, 0, 1).astype(np.float32)

    data = {
        "train_images": train_imgs,
        "train_angles": train_angles,
        "train_labels": train_raw["labels"].astype(np.float32),
        "train_ids": train_raw["ids"],
        "test_images": test_imgs,
        "test_angles": test_angles,
        "test_ids": test_raw["ids"],
    }

    # Save to cache
    print("Saving processed data to cache...")
    for k, v in data.items():
        np.save(files[k], v)

    return data


def get_fold_loaders(fold_idx, load_cached_data=True):
    """
    Returns train and val loaders for a specific fold using StratifiedKFold.
    """
    # Load all labeled data
    data = load_and_process_data(load_cached_data=load_cached_data)

    X = data["train_images"]
    y = data["train_labels"]
    angles = data["train_angles"]
    ids = data["train_ids"]

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Find indices for the requested fold
    train_index, val_index = None, None
    for i, (t_idx, v_idx) in enumerate(skf.split(X, y)):
        if i == fold_idx:
            train_index = t_idx
            val_index = v_idx
            break

    if train_index is None:
        raise ValueError(
            f"Fold index {fold_idx} out of range for {Config.NUM_FOLDS} folds."
        )

    # Subset data
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    ang_train, ang_val = angles[train_index], angles[val_index]
    ids_train, ids_val = ids[train_index], ids[val_index]

    # Create Datasets
    train_dataset = IcebergDataset(
        X_train, ang_train, y_train, ids_train, transform=get_transforms("train")
    )
    val_dataset = IcebergDataset(
        X_val, ang_val, y_val, ids_val, transform=get_transforms("val")
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Returns the test data loader.
    """
    data = load_and_process_data(load_cached_data=load_cached_data)

    X_test = data["test_images"]
    ang_test = data["test_angles"]
    ids_test = data["test_ids"]

    test_dataset = IcebergDataset(
        X_test, ang_test, labels=None, ids=ids_test, transform=get_transforms("test")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
