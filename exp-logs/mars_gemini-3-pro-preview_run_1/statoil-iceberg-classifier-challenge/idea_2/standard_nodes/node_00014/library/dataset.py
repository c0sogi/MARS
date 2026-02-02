import os
import json
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import (
    TRAIN_JSON,
    TEST_JSON,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    WORKING_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)


class IcebergDataset(Dataset):
    """
    Custom Dataset for Iceberg vs Ship classification.
    """

    def __init__(self, images, angles, labels=None, transform=None, ids=None):
        """
        Args:
            images (np.ndarray): Shape (N, 3, H, W), float32.
            angles (np.ndarray): Shape (N,), float32.
            labels (np.ndarray, optional): Shape (N,), float32.
            transform (callable, optional): Transform to be applied on a sample.
            ids (list/array, optional): IDs of the images (useful for test set).
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.transform = transform
        self.ids = ids

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image is already (3, H, W) from processing
        img_np = self.images[idx]
        angle_val = self.angles[idx]

        # Convert to tensor
        img_tensor = torch.from_numpy(img_np)
        angle_tensor = torch.tensor([angle_val], dtype=torch.float32)

        # Apply transforms (Augmentation)
        # Note: torchvision transforms expect (C, H, W) tensor
        if self.transform:
            img_tensor = self.transform(img_tensor)

        if self.labels is not None:
            label_val = self.labels[idx]
            label_tensor = torch.tensor([label_val], dtype=torch.float32)
            return img_tensor, angle_tensor, label_tensor
        else:
            # For test set, we might want to return ID, but DataLoader usually handles indices.
            # We will return just image and angle for inference.
            return img_tensor, angle_tensor


def _process_and_cache(load_cached_data=True):
    """
    Loads raw data, processes it (resize, scale, impute), and caches it.
    Returns dictionaries containing train, val, test data.
    """
    # Cache filenames
    cache_files = {
        "train_img": os.path.join(WORKING_DIR, "train_images.npy"),
        "train_ang": os.path.join(WORKING_DIR, "train_angles.npy"),
        "train_lbl": os.path.join(WORKING_DIR, "train_labels.npy"),
        "val_img": os.path.join(WORKING_DIR, "val_images.npy"),
        "val_ang": os.path.join(WORKING_DIR, "val_angles.npy"),
        "val_lbl": os.path.join(WORKING_DIR, "val_labels.npy"),
        "test_img": os.path.join(WORKING_DIR, "test_images.npy"),
        "test_ang": os.path.join(WORKING_DIR, "test_angles.npy"),
        "test_ids": os.path.join(WORKING_DIR, "test_ids.npy"),
    }

    # Check if cache exists
    all_exist = all(os.path.exists(f) for f in cache_files.values())

    if load_cached_data and all_exist:
        print("Loading processed data from cache...")
        data = {}
        data["train_images"] = np.load(cache_files["train_img"])
        data["train_angles"] = np.load(cache_files["train_ang"])
        data["train_labels"] = np.load(cache_files["train_lbl"])
        data["val_images"] = np.load(cache_files["val_img"])
        data["val_angles"] = np.load(cache_files["val_ang"])
        data["val_labels"] = np.load(cache_files["val_lbl"])
        data["test_images"] = np.load(cache_files["test_img"])
        data["test_angles"] = np.load(cache_files["test_ang"])
        data["test_ids"] = np.load(cache_files["test_ids"])
        return data

    print("Processing data from scratch...")

    # 1. Load Metadata
    df_train_meta = pd.read_csv(TRAIN_META_PATH)
    df_val_meta = pd.read_csv(VAL_META_PATH)
    df_test_meta = pd.read_csv(TEST_META_PATH)

    # 2. Load Raw JSONs
    print("Loading raw JSON files...")
    with open(TRAIN_JSON, "r") as f:
        raw_train_data = json.load(f)  # List of dicts
    with open(TEST_JSON, "r") as f:
        raw_test_data = json.load(f)  # List of dicts

    # Helper to extract data based on metadata indices
    def extract_data(df, raw_source, has_label=True):
        images = []
        angles = []
        labels = []
        ids = []

        # Create a map for O(1) access if raw_source is large, but here indices are provided
        # raw_source is a list, df['sample_index'] are indices into this list.

        indices = df["sample_index"].values
        ids_vec = df["id"].values

        for i, idx in enumerate(indices):
            item = raw_source[idx]

            # Bands: Flattened list of 5625 floats -> 75x75
            b1 = np.array(item["band_1"], dtype=np.float32).reshape(75, 75)
            b2 = np.array(item["band_2"], dtype=np.float32).reshape(75, 75)

            # Stack: (75, 75, 2)
            img = np.dstack((b1, b2))
            images.append(img)

            # Angle
            ang = item["inc_angle"]
            # Handle 'na'
            if ang == "na":
                angles.append(np.nan)
            else:
                angles.append(float(ang))

            ids.append(ids_vec[i])

            if has_label:
                labels.append(item["is_iceberg"])

        return (
            np.array(images),
            np.array(angles, dtype=np.float32),
            np.array(labels, dtype=np.float32),
            np.array(ids),
        )

    print("Extracting arrays...")
    X_train_raw, ang_train, y_train, _ = extract_data(
        df_train_meta, raw_train_data, has_label=True
    )
    X_val_raw, ang_val, y_val, _ = extract_data(
        df_val_meta, raw_train_data, has_label=True
    )
    X_test_raw, ang_test, _, ids_test = extract_data(
        df_test_meta, raw_test_data, has_label=False
    )

    # 3. Image Processing
    print("Resizing and constructing 3-channel images...")

    def process_images(X_raw):
        # X_raw: (N, 75, 75, 2)
        N = X_raw.shape[0]
        X_processed = np.zeros((N, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)

        for i in range(N):
            b1 = X_raw[i, :, :, 0]
            b2 = X_raw[i, :, :, 1]

            # Create 3rd channel: Mean of b1 and b2
            b3 = (b1 + b2) / 2.0

            # Stack to 3 channels
            img_3ch = np.dstack((b1, b2, b3))  # (75, 75, 3)

            # Resize
            img_resized = cv2.resize(
                img_3ch, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC
            )
            X_processed[i] = img_resized

        return X_processed

    X_train = process_images(X_train_raw)
    X_val = process_images(X_val_raw)
    X_test = process_images(X_test_raw)

    # 4. Scaling (Min-Max based on Train)
    print("Applying Min-Max scaling...")
    # Compute stats per channel on TRAIN only
    # Channels: 0=Band1, 1=Band2, 2=Mean

    for c in range(3):
        train_c = X_train[:, :, :, c]
        min_val = np.min(train_c)
        max_val = np.max(train_c)

        # Apply to all splits
        X_train[:, :, :, c] = (X_train[:, :, :, c] - min_val) / (max_val - min_val)
        X_val[:, :, :, c] = (X_val[:, :, :, c] - min_val) / (max_val - min_val)
        X_test[:, :, :, c] = (X_test[:, :, :, c] - min_val) / (max_val - min_val)

    # Transpose to (N, 3, H, W) for PyTorch
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_val = np.transpose(X_val, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))

    # 5. Angle Imputation and Scaling
    print("Processing incidence angles...")
    # Impute NaNs with Train Mean
    train_ang_mean = np.nanmean(ang_train)

    # Fill NaNs
    ang_train = np.nan_to_num(ang_train, nan=train_ang_mean)
    ang_val = np.nan_to_num(ang_val, nan=train_ang_mean)
    ang_test = np.nan_to_num(ang_test, nan=train_ang_mean)

    # Standard Scaling (Z-score) based on Train
    ang_std = np.std(ang_train)
    if ang_std == 0:
        ang_std = 1.0  # Avoid div by zero

    ang_train = (ang_train - train_ang_mean) / ang_std
    ang_val = (ang_val - train_ang_mean) / ang_std
    ang_test = (ang_test - train_ang_mean) / ang_std

    # 6. Save to Cache
    print("Saving processed data to cache...")
    os.makedirs(WORKING_DIR, exist_ok=True)
    np.save(cache_files["train_img"], X_train)
    np.save(cache_files["train_ang"], ang_train)
    np.save(cache_files["train_lbl"], y_train)
    np.save(cache_files["val_img"], X_val)
    np.save(cache_files["val_ang"], ang_val)
    np.save(cache_files["val_lbl"], y_val)
    np.save(cache_files["test_img"], X_test)
    np.save(cache_files["test_ang"], ang_test)
    np.save(cache_files["test_ids"], ids_test)

    return {
        "train_images": X_train,
        "train_angles": ang_train,
        "train_labels": y_train,
        "val_images": X_val,
        "val_angles": ang_val,
        "val_labels": y_val,
        "test_images": X_test,
        "test_angles": ang_test,
        "test_ids": ids_test,
    }


def get_dataloaders(
    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, load_cached_data=True
):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    data = _process_and_cache(load_cached_data)

    # Define Transforms for Training
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            # Cite solution_lesson_node_00006: Add scaling and translation
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ]
    )

    # No transforms for Val/Test (except resizing/scaling which is done in preprocessing)

    # Create Datasets
    train_dataset = IcebergDataset(
        data["train_images"],
        data["train_angles"],
        data["train_labels"],
        transform=train_transform,
    )

    val_dataset = IcebergDataset(
        data["val_images"], data["val_angles"], data["val_labels"], transform=None
    )

    test_dataset = IcebergDataset(
        data["test_images"],
        data["test_angles"],
        labels=None,
        transform=None,
        ids=data["test_ids"],
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader, data["test_ids"]
