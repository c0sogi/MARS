import os
import cv2
import torch
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TABULAR_PREPROCESSOR_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    IMAGE_PATH_COL,
    TARGET_COL,
    SEED,
)
from library.utils import seed_everything

# Ensure deterministic behavior
seed_everything(SEED)

# Cache paths for processed tabular data
TRAIN_TAB_CACHE = os.path.join(WORKING_DIR, "train_tabular.npy")
VAL_TAB_CACHE = os.path.join(WORKING_DIR, "val_tabular.npy")
TEST_TAB_CACHE = os.path.join(WORKING_DIR, "test_tabular.npy")


def get_transforms(phase: str):
    """
    Returns torchvision transforms for the specified phase.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if phase == "train":
        return transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )


class MelanomaDataset(Dataset):
    def __init__(self, df, tabular_data, transform=None, is_test=False):
        self.df = df
        self.tabular_data = tabular_data.astype(np.float32)
        self.transform = transform
        self.is_test = is_test

        # Pre-compute paths to avoid overhead in __getitem__
        # Metadata file_path is relative to INPUT_DIR (e.g., "jpeg/train/ISIC_xxx.jpg")
        self.image_paths = [
            os.path.join(INPUT_DIR, p) for p in df[IMAGE_PATH_COL].values
        ]

        if not self.is_test:
            self.targets = df[TARGET_COL].values.astype(np.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load Image
        img_path = self.image_paths[idx]
        # Use PIL for compatibility with torchvision transforms
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback for missing images (though verification script says 0 missing)
            # Create a black image
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        # Get Tabular Data
        tab_features = torch.tensor(self.tabular_data[idx], dtype=torch.float32)

        # Get Target
        if self.is_test:
            # Return dummy target for test set
            target = torch.tensor(0.0, dtype=torch.float32)
        else:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)

        return image, tab_features, target


def preprocess_metadata(load_cached_data=True):
    """
    Loads metadata, processes tabular features, and caches the result.
    Uses .npy for data cache and joblib for the preprocessor object.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Check if cache exists
    if (
        load_cached_data
        and os.path.exists(TRAIN_TAB_CACHE)
        and os.path.exists(VAL_TAB_CACHE)
        and os.path.exists(TEST_TAB_CACHE)
    ):

        print("Loading cached tabular data...")
        train_tab = np.load(TRAIN_TAB_CACHE)
        val_tab = np.load(VAL_TAB_CACHE)
        test_tab = np.load(TEST_TAB_CACHE)

        # Load DataFrames just for the other columns (paths, targets)
        train_df = pd.read_csv(TRAIN_METADATA_PATH)
        val_df = pd.read_csv(VAL_METADATA_PATH)
        test_df = pd.read_csv(TEST_METADATA_PATH)

        return train_df, val_df, test_df, train_tab, val_tab, test_tab

    print("Processing metadata from scratch...")
    train_df = pd.read_csv(TRAIN_METADATA_PATH)
    val_df = pd.read_csv(VAL_METADATA_PATH)
    test_df = pd.read_csv(TEST_METADATA_PATH)

    # Define Preprocessing Pipeline
    # Numerical: Impute Mean -> Scale
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical: Impute 'missing' -> OneHot
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, NUMERICAL_COLS),
            ("cat", cat_transformer, CATEGORICAL_COLS),
        ]
    )

    # Fit on Train, Transform All
    train_tab = preprocessor.fit_transform(train_df)
    val_tab = preprocessor.transform(val_df)
    test_tab = preprocessor.transform(test_df)

    # Cache Data (NPY)
    np.save(TRAIN_TAB_CACHE, train_tab)
    np.save(VAL_TAB_CACHE, val_tab)
    np.save(TEST_TAB_CACHE, test_tab)

    # Save Preprocessor (Joblib)
    joblib.dump(preprocessor, TABULAR_PREPROCESSOR_PATH)
    print(f"Tabular processing complete. Features shape: {train_tab.shape}")

    return train_df, val_df, test_df, train_tab, val_tab, test_tab


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.
    Handles class imbalance in training set using WeightedRandomSampler.
    """
    # 1. Preprocess Data
    train_df, val_df, test_df, train_tab, val_tab, test_tab = preprocess_metadata(
        load_cached_data
    )

    # 2. Create Datasets
    train_dataset = MelanomaDataset(
        train_df, train_tab, transform=get_transforms("train"), is_test=False
    )
    val_dataset = MelanomaDataset(
        val_df, val_tab, transform=get_transforms("val"), is_test=False
    )
    test_dataset = MelanomaDataset(
        test_df, test_tab, transform=get_transforms("test"), is_test=True
    )

    # 3. Handle Class Imbalance (WeightedRandomSampler)
    # Extract targets to compute weights
    targets = train_df[TARGET_COL].values
    class_counts = np.bincount(targets)

    # Avoid division by zero if a class is missing (unlikely in this dataset)
    class_weights = 1.0 / (class_counts + 1e-6)

    # Assign weight to each sample based on its class
    sample_weights = class_weights[targets]
    sample_weights = torch.from_numpy(sample_weights).double()

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,  # Mutually exclusive with shuffle
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
