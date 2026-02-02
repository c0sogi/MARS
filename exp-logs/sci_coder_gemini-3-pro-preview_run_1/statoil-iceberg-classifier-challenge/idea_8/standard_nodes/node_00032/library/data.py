import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def get_transforms(phase: str):
    """
    Returns Albumentations transforms for the specified phase.

    Args:
        phase (str): 'train' or 'valid'/'test'.

    Returns:
        A.Compose: The composition of transforms.
    """
    if phase == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=20, p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE), ToTensorV2()]
        )


def process_and_cache_data(load_cached_data: bool = True):
    """
    Loads raw JSON data, performs global preprocessing (3-channel composite,
    global min-max scaling, angle imputation/normalization), and caches the
    result as .npy files.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        dict: A dictionary containing processed numpy arrays for images, angles,
              labels, and ids for both train and test sets.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    paths = {
        "train_images": os.path.join(cache_dir, "train_images.npy"),
        "train_angles": os.path.join(cache_dir, "train_angles.npy"),
        "train_labels": os.path.join(cache_dir, "train_labels.npy"),
        "test_images": os.path.join(cache_dir, "test_images.npy"),
        "test_angles": os.path.join(cache_dir, "test_angles.npy"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
    }

    # Attempt to load from cache
    if load_cached_data:
        if all(os.path.exists(p) for p in paths.values()):
            print("Loading cached data from", cache_dir)
            return {k: np.load(v, allow_pickle=True) for k, v in paths.items()}

    print("Processing data from scratch...")

    # Load raw JSON data
    with open(Config.TRAIN_JSON, "r") as f:
        train_data = json.load(f)
    with open(Config.TEST_JSON, "r") as f:
        test_data = json.load(f)

    # --- Helper Functions ---
    def extract_images(data):
        # Extract Band 1 and Band 2, reshape to 75x75
        imgs = []
        for item in data:
            b1 = np.array(item["band_1"]).reshape(75, 75)
            b2 = np.array(item["band_2"]).reshape(75, 75)
            imgs.append(np.stack([b1, b2], axis=-1))
        return np.array(imgs)  # Shape: (N, 75, 75, 2)

    def extract_angles(data):
        angles = []
        for item in data:
            a = item["inc_angle"]
            if a == "na":
                angles.append(np.nan)
            else:
                angles.append(float(a))
        return np.array(angles)

    # --- Extract Data ---
    X_train = extract_images(train_data)
    X_test = extract_images(test_data)

    ang_train = extract_angles(train_data)
    ang_test = extract_angles(test_data)

    y_train = np.array([item["is_iceberg"] for item in train_data])
    ids_test = np.array([item["id"] for item in test_data])

    # --- Global Min-Max Scaling ---
    # Compute statistics on the entire training set per channel
    # Band 1 (HH)
    min_b1 = X_train[:, :, :, 0].min()
    max_b1 = X_train[:, :, :, 0].max()
    # Band 2 (HV)
    min_b2 = X_train[:, :, :, 1].min()
    max_b2 = X_train[:, :, :, 1].max()

    # Apply scaling to Train and Test
    X_train[:, :, :, 0] = (X_train[:, :, :, 0] - min_b1) / (max_b1 - min_b1)
    X_train[:, :, :, 1] = (X_train[:, :, :, 1] - min_b2) / (max_b2 - min_b2)

    X_test[:, :, :, 0] = (X_test[:, :, :, 0] - min_b1) / (max_b1 - min_b1)
    X_test[:, :, :, 1] = (X_test[:, :, :, 1] - min_b2) / (max_b2 - min_b2)

    # --- Construct 3rd Channel (Mean) ---
    # We use the mean of the normalized bands
    def add_mean_channel(X):
        b3 = (X[:, :, :, 0] + X[:, :, :, 1]) / 2.0
        b3 = np.expand_dims(b3, axis=-1)
        return np.concatenate([X, b3], axis=-1)

    X_train = add_mean_channel(X_train)
    X_test = add_mean_channel(X_test)

    # --- Process Angles ---
    # Impute 'na' in training data with mean
    train_mean_angle = np.nanmean(ang_train)
    ang_train[np.isnan(ang_train)] = train_mean_angle
    # Impute test data if necessary (though usually not 'na' in test, good safety)
    ang_test[np.isnan(ang_test)] = train_mean_angle

    # Normalize angles (Standard Scaling using Train stats)
    angle_mean = np.mean(ang_train)
    angle_std = np.std(ang_train)

    ang_train = (ang_train - angle_mean) / angle_std
    ang_test = (ang_test - angle_mean) / angle_std

    # --- Save to Cache ---
    # Use float32 for model compatibility
    X_train = X_train.astype(np.float32)
    ang_train = ang_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    ang_test = ang_test.astype(np.float32)

    np.save(paths["train_images"], X_train)
    np.save(paths["train_angles"], ang_train)
    np.save(paths["train_labels"], y_train)
    np.save(paths["test_images"], X_test)
    np.save(paths["test_angles"], ang_test)
    np.save(paths["test_ids"], ids_test)

    return {
        "train_images": X_train,
        "train_angles": ang_train,
        "train_labels": y_train,
        "test_images": X_test,
        "test_angles": ang_test,
        "test_ids": ids_test,
    }


class IcebergDataset(Dataset):
    """
    Dataset class for Ship vs Iceberg classification.
    """

    def __init__(self, images, angles, labels=None, transform=None):
        self.images = images
        self.angles = angles
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve data
        image = self.images[idx]  # (75, 75, 3)
        angle = self.angles[idx]  # scalar

        # Apply transforms (Augmentations + Resize)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback: Convert to tensor (C, H, W)
            image = torch.tensor(image).permute(2, 0, 1)

        # Ensure angle is a tensor
        angle = torch.tensor(angle, dtype=torch.float32)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, angle, label
        else:
            return image, angle


def mixup_data(images, angles, labels, alpha=0.4):
    """
    Performs Mixup on the batch.
    Mixes both the images and the incidence angles.

    Args:
        images (torch.Tensor): Batch of images.
        angles (torch.Tensor): Batch of incidence angles.
        labels (torch.Tensor): Batch of labels.
        alpha (float): Mixup alpha parameter.

    Returns:
        mixed_images, mixed_angles, labels_a, labels_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = images.size(0)
    index = torch.randperm(batch_size).to(images.device)

    mixed_images = lam * images + (1 - lam) * images[index, :]
    mixed_angles = lam * angles + (1 - lam) * angles[index]
    y_a, y_b = labels, labels[index]

    return mixed_images, mixed_angles, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes the mixup loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
