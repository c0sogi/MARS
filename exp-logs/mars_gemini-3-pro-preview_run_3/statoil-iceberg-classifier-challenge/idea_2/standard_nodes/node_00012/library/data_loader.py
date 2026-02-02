import os
import json
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from library.config import (
    TRAIN_JSON,
    TEST_JSON,
    WORKING_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    N_FOLDS,
    SEED,
)

# Define internal cache paths for the full dataset used in CV
CACHE_X_CV = os.path.join(WORKING_DIR, "X_train_cv.npy")
CACHE_Y_CV = os.path.join(WORKING_DIR, "y_train_cv.npy")
CACHE_ANGLE_CV = os.path.join(WORKING_DIR, "angle_train_cv.npy")
CACHE_SCALING = os.path.join(WORKING_DIR, "scaling_params.npy")

CACHE_X_TEST_CV = os.path.join(WORKING_DIR, "X_test_cv.npy")
CACHE_ID_TEST_CV = os.path.join(WORKING_DIR, "id_test_cv.npy")
CACHE_ANGLE_TEST_CV = os.path.join(WORKING_DIR, "angle_test_cv.npy")


class IcebergDataset(Dataset):
    def __init__(self, X, angles, y=None, transform=None):
        """
        Args:
            X (np.ndarray): Images of shape (N, 224, 224, 3), float32 in [0, 1].
            angles (np.ndarray): Incidence angles of shape (N,).
            y (np.ndarray, optional): Labels of shape (N,).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.X = X
        self.angles = angles
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        angle = self.angles[idx]

        if self.transform:
            img = self.transform(img)

        # Ensure tensor if transform didn't convert it (e.g. if transform is None)
        if not torch.is_tensor(img):
            # Convert HWC to CHW
            img = torch.from_numpy(img).permute(2, 0, 1)

        img = img.float()
        angle = torch.tensor(angle, dtype=torch.float32)

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return img, angle, label
        else:
            return img, angle


def process_sample(band_1, band_2):
    """
    Reshapes, stacks, and resizes a single image sample.
    """
    # Reshape flattened bands to 75x75
    b1 = np.array(band_1, dtype=np.float32).reshape(75, 75)
    b2 = np.array(band_2, dtype=np.float32).reshape(75, 75)

    # Create 3rd channel (average of HH and HV)
    b3 = (b1 + b2) / 2.0

    # Stack to (75, 75, 3)
    img = np.dstack((b1, b2, b3))

    # No resizing needed for Custom CNN (Cite solution_lesson_node_00009)
    # img is already 75x75

    return img


def load_and_process_data(load_cached_data=True):
    """
    Loads raw JSON data, processes images, handles imputation, and caches results.
    Returns full training and test sets.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)

    # --- Load Training Data ---
    if (
        load_cached_data
        and os.path.exists(CACHE_X_CV)
        and os.path.exists(CACHE_Y_CV)
        and os.path.exists(CACHE_SCALING)
    ):
        print("Loading training data from cache...")
        X_train = np.load(CACHE_X_CV)
        y_train = np.load(CACHE_Y_CV)
        angles_train = np.load(CACHE_ANGLE_CV)
        scaling_params = np.load(CACHE_SCALING)
        min_val, max_val = scaling_params[0], scaling_params[1]
    else:
        print("Processing training data from raw JSON...")
        with open(TRAIN_JSON, "r") as f:
            train_data = json.load(f)
        df_train = pd.DataFrame(train_data)

        # Impute missing angles with median
        df_train["inc_angle"] = pd.to_numeric(df_train["inc_angle"], errors="coerce")
        median_angle = df_train["inc_angle"].median()
        df_train["inc_angle"] = df_train["inc_angle"].fillna(median_angle)

        # Process images
        X_train_list = []
        for _, row in df_train.iterrows():
            X_train_list.append(process_sample(row["band_1"], row["band_2"]))

        X_train = np.array(X_train_list, dtype=np.float32)
        y_train = df_train["is_iceberg"].values.astype(np.float32)
        angles_train = df_train["inc_angle"].values.astype(np.float32)

        # Compute scaling parameters (Global Min-Max)
        min_val = X_train.min()
        max_val = X_train.max()

        # Scale to [0, 1]
        X_train = (X_train - min_val) / (max_val - min_val)

        # Cache
        np.save(CACHE_X_CV, X_train)
        np.save(CACHE_Y_CV, y_train)
        np.save(CACHE_ANGLE_CV, angles_train)
        np.save(CACHE_SCALING, np.array([min_val, max_val]))

    # --- Load Test Data ---
    if load_cached_data and os.path.exists(CACHE_X_TEST_CV):
        print("Loading test data from cache...")
        X_test = np.load(CACHE_X_TEST_CV)
        ids_test = np.load(CACHE_ID_TEST_CV, allow_pickle=True)
        angles_test = np.load(CACHE_ANGLE_TEST_CV)
    else:
        print("Processing test data from raw JSON...")
        with open(TEST_JSON, "r") as f:
            test_data = json.load(f)
        df_test = pd.DataFrame(test_data)

        df_test["inc_angle"] = pd.to_numeric(df_test["inc_angle"], errors="coerce")
        # Fill NA with median if any (assuming test set distribution is similar or clean)
        df_test["inc_angle"] = df_test["inc_angle"].fillna(
            df_test["inc_angle"].median()
        )

        X_test_list = []
        for _, row in df_test.iterrows():
            X_test_list.append(process_sample(row["band_1"], row["band_2"]))

        X_test = np.array(X_test_list, dtype=np.float32)
        ids_test = df_test["id"].values
        angles_test = df_test["inc_angle"].values.astype(np.float32)

        # Scale using training parameters
        X_test = (X_test - min_val) / (max_val - min_val)

        # Cache
        np.save(CACHE_X_TEST_CV, X_test)
        np.save(CACHE_ID_TEST_CV, ids_test)
        np.save(CACHE_ANGLE_TEST_CV, angles_test)

    return X_train, y_train, angles_train, X_test, ids_test, angles_test


def get_fold_loaders(fold_idx, X, y, angles, batch_size=BATCH_SIZE):
    """
    Returns train and validation DataLoaders for a specific fold in Stratified K-Fold.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Get indices for the specific fold
    train_idx, val_idx = list(skf.split(X, y))[fold_idx]

    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    ang_train_fold, ang_val_fold = angles[train_idx], angles[val_idx]

    # Define Transforms
    # Images are already [0, 1]. Normalization uses ImageNet stats.
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Converts numpy [0, 1] to tensor [0, 1] (CHW)
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = IcebergDataset(
        X_train_fold, ang_train_fold, y_train_fold, transform=train_transform
    )
    val_ds = IcebergDataset(
        X_val_fold, ang_val_fold, y_val_fold, transform=val_transform
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader


def get_test_loader(X_test, angles_test, batch_size=BATCH_SIZE):
    """
    Returns a DataLoader for the test set.
    """
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_ds = IcebergDataset(X_test, angles_test, transform=test_transform)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return test_loader
