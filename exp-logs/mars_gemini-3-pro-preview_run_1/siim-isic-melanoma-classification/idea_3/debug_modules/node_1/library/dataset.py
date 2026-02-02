import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from library.config import (
    INPUT_DIR,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    TABULAR_COLS,
    USE_RANDOM_ERASING,
    RANDOM_ERASE_PROB,
    WORKING_DIR,
    SEED,
    DEBUG,
    DEBUG_SAMPLE_SIZE,
)


def get_transforms(mode="train", image_size=IMAGE_SIZE):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        mode (str): 'train' for augmentation, 'val'/'test' for deterministic resizing.
        image_size (int): Target spatial dimension.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=20, p=0.5),
                A.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0, p=0.5
                ),
                # CoarseDropout acts as Random Erasing
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(image_size * 0.15),
                    max_width=int(image_size * 0.15),
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=RANDOM_ERASE_PROB if USE_RANDOM_ERASING else 0.0,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def get_tabular_pipeline():
    """
    Defines the preprocessing pipeline for tabular data.
    """
    # Numerical: age_approx
    num_features = ["age_approx"]
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical: sex, anatom_site_general_challenge
    cat_features = ["sex", "anatom_site_general_challenge"]
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
        ],
        verbose_feature_names_out=False,
    )
    return preprocessor


def load_or_process_tabular(train_df, val_df, test_df, load_cached_data=True):
    """
    Handles caching logic for tabular data preprocessing.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)

    train_path = os.path.join(WORKING_DIR, "train_tabular.npy")
    val_path = os.path.join(WORKING_DIR, "val_tabular.npy")
    test_path = os.path.join(WORKING_DIR, "test_tabular.npy")

    # Check if cache exists
    cache_exists = (
        os.path.exists(train_path)
        and os.path.exists(val_path)
        and os.path.exists(test_path)
    )

    if load_cached_data and cache_exists:
        print("Loading cached tabular features...")
        X_train = np.load(train_path)
        X_val = np.load(val_path)
        X_test = np.load(test_path)
    else:
        print("Processing tabular features from scratch...")
        pipeline = get_tabular_pipeline()

        # Fit on training data only to prevent leakage
        X_train = pipeline.fit_transform(train_df)
        X_val = pipeline.transform(val_df)
        X_test = pipeline.transform(test_df)

        # Save to cache
        np.save(train_path, X_train)
        np.save(val_path, X_val)
        np.save(test_path, X_test)

    return X_train, X_val, X_test


class ISICDataset(Dataset):
    def __init__(self, df, tabular_data, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            tabular_data (np.array): Preprocessed tabular features matching df rows.
            transforms (albumentations.Compose): Image transformations.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.tabular_data = tabular_data
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # 1. Load Image
        # file_path in metadata is relative to INPUT_DIR (e.g., "jpeg/train/ISIC_xxxx.jpg")
        img_path = os.path.join(INPUT_DIR, row["file_path"])

        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing images (though verification script says 0 missing)
            # Create a black image
            image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # 3. Get Tabular Features
        tab_feat = torch.tensor(self.tabular_data[index], dtype=torch.float32)

        # 4. Get Target
        if self.mode == "test":
            target = torch.tensor(0.0, dtype=torch.float32)  # Dummy target for test
        else:
            target = torch.tensor(row["target"], dtype=torch.float32)

        return image, tab_feat, target


def get_dataloaders(
    load_cached_data=True,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    num_workers=NUM_WORKERS,
):
    """
    Prepares DataLoaders for train, val, and test sets.
    """
    print("Initializing DataLoaders...")

    # 1. Load Metadata
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # Debug Mode: Subsample
    if DEBUG:
        print(f"DEBUG mode: Sampling {DEBUG_SAMPLE_SIZE} rows.")
        train_df = train_df.head(DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(DEBUG_SAMPLE_SIZE)

    # 2. Process Tabular Data
    # Ensure we pass copies or specific columns to avoid modifying original df structure if needed,
    # but pipeline handles column selection internally.
    X_train, X_val, X_test = load_or_process_tabular(
        train_df, val_df, test_df, load_cached_data=load_cached_data
    )

    # 3. Create Datasets
    train_ds = ISICDataset(
        train_df, X_train, transforms=get_transforms("train", image_size), mode="train"
    )
    val_ds = ISICDataset(
        val_df, X_val, transforms=get_transforms("val", image_size), mode="val"
    )
    test_ds = ISICDataset(
        test_df, X_test, transforms=get_transforms("test", image_size), mode="test"
    )

    # 4. Handle Class Imbalance (WeightedRandomSampler)
    # Calculate weights for training set
    targets = train_df["target"].values
    class_counts = np.bincount(targets)
    # Avoid division by zero if debug set has only one class
    if len(class_counts) < 2:
        class_weights = np.ones(2)
    else:
        class_weights = 1.0 / class_counts

    sample_weights = np.array([class_weights[t] for t in targets])
    sample_weights = torch.from_numpy(sample_weights).double()

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # 5. Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,  # Sampler is mutually exclusive with shuffle
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(
        f"DataLoaders ready. Train: {len(train_loader)} batches, Val: {len(val_loader)} batches."
    )
    return train_loader, val_loader, test_loader
