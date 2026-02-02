import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from library.utils import seed_everything

# Constants
INPUT_ROOT = "./input"
CACHE_DIR = "./working/idea_6"


class MelanomaDataset(Dataset):
    def __init__(self, csv, mode, meta_features=None, transform=None):
        """
        Args:
            csv (pd.DataFrame): Dataframe containing image paths and targets.
            mode (str): 'train', 'val', or 'test'.
            meta_features (np.ndarray): Preprocessed metadata features.
            transform (albumentations.Compose): Augmentation pipeline.
        """
        self.csv = csv.reset_index(drop=True)
        self.mode = mode
        self.meta_features = meta_features
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        # Image Path Construction
        img_path = os.path.join(INPUT_ROOT, row["file_path"])

        # Load Image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for safety, though data verification ensures paths exist
            # Create a black image of expected size
            image = np.zeros((384, 384, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transform:
            res = self.transform(image=image)
            image = res["image"]
        else:
            # Minimal fallback transform
            T = A.Compose([A.Resize(384, 384), A.Normalize(), ToTensorV2()])
            res = T(image=image)
            image = res["image"]

        # Prepare Output Dictionary
        data = {}
        data["image"] = image
        data["image_name"] = row["image_name"]

        # Attach Metadata Features
        if self.meta_features is not None:
            data["meta"] = torch.tensor(self.meta_features[index], dtype=torch.float32)

        # Attach Targets (Train/Val only)
        if self.mode != "test":
            data["target"] = torch.tensor(row["target"], dtype=torch.float32)

            # Auxiliary Target: Diagnosis
            # We expect 'diagnosis_idx' to be present in the dataframe if processing was done
            if "diagnosis_idx" in self.csv.columns:
                data["diagnosis"] = torch.tensor(row["diagnosis_idx"], dtype=torch.long)

        return data


def get_transforms(image_size=384):
    """
    Returns training and validation/test transformations.
    """
    train_transforms = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )

    val_transforms = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )

    return train_transforms, val_transforms


def process_metadata(df_train, df_val, df_test, load_cached_data=True):
    """
    Processes metadata: One-Hot Encoding, Scaling, Label Encoding.
    Implements caching mechanism using .npy files.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define cache file paths
    cache_files = {
        "meta_train": os.path.join(CACHE_DIR, "meta_train.npy"),
        "meta_val": os.path.join(CACHE_DIR, "meta_val.npy"),
        "meta_test": os.path.join(CACHE_DIR, "meta_test.npy"),
        "diag_train": os.path.join(CACHE_DIR, "diag_train.npy"),
        "diag_val": os.path.join(CACHE_DIR, "diag_val.npy"),
        "num_diag": os.path.join(CACHE_DIR, "num_diag_classes.npy"),
    }

    # Check if cache exists
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print("Loading cached metadata features...")
        meta_train = np.load(cache_files["meta_train"])
        meta_val = np.load(cache_files["meta_val"])
        meta_test = np.load(cache_files["meta_test"])
        diag_train = np.load(cache_files["diag_train"])
        diag_val = np.load(cache_files["diag_val"])
        num_diag_classes = int(np.load(cache_files["num_diag"]))

        return meta_train, meta_val, meta_test, diag_train, diag_val, num_diag_classes

    print("Processing metadata from scratch...")

    # Configuration
    cat_cols = ["sex", "anatom_site_general_challenge"]
    num_cols = ["age_approx"]

    # Copy dataframes to avoid side effects
    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()

    # 1. Imputation
    # Categorical: Fill with 'unknown'
    for col in cat_cols:
        df_train[col] = df_train[col].fillna("unknown")
        df_val[col] = df_val[col].fillna("unknown")
        df_test[col] = df_test[col].fillna("unknown")

    # Numerical: Mean imputation
    imputer = SimpleImputer(strategy="mean")
    df_train[num_cols] = imputer.fit_transform(df_train[num_cols])
    df_val[num_cols] = imputer.transform(df_val[num_cols])
    df_test[num_cols] = imputer.transform(df_test[num_cols])

    # 2. One-Hot Encoding
    # Handle unknown categories (e.g., in test set) by ignoring them (all zeros)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    # Fit on training data
    X_cat_train = df_train[cat_cols].values
    ohe.fit(X_cat_train)

    X_cat_train_enc = ohe.transform(X_cat_train)
    X_cat_val_enc = ohe.transform(df_val[cat_cols].values)
    X_cat_test_enc = ohe.transform(df_test[cat_cols].values)

    # 3. Standardization
    scaler = StandardScaler()

    # Fit on training data
    X_num_train = df_train[num_cols].values
    scaler.fit(X_num_train)

    X_num_train_sc = scaler.transform(X_num_train)
    X_num_val_sc = scaler.transform(df_val[num_cols].values)
    X_num_test_sc = scaler.transform(df_test[num_cols].values)

    # Concatenate processed features
    meta_train = np.hstack([X_cat_train_enc, X_num_train_sc])
    meta_val = np.hstack([X_cat_val_enc, X_num_val_sc])
    meta_test = np.hstack([X_cat_test_enc, X_num_test_sc])

    # 4. Process Auxiliary Target (Diagnosis)
    # Only available in train/val
    df_train["diagnosis"] = df_train["diagnosis"].fillna("unknown")
    df_val["diagnosis"] = df_val["diagnosis"].fillna("unknown")

    le = LabelEncoder()
    # Fit on combined train+val to ensure all training classes are mapped
    all_diag = (
        pd.concat([df_train["diagnosis"], df_val["diagnosis"]]).astype(str).unique()
    )
    le.fit(all_diag)

    diag_train = le.transform(df_train["diagnosis"].astype(str))
    diag_val = le.transform(df_val["diagnosis"].astype(str))
    num_diag_classes = len(le.classes_)

    # Save to cache
    np.save(cache_files["meta_train"], meta_train)
    np.save(cache_files["meta_val"], meta_val)
    np.save(cache_files["meta_test"], meta_test)
    np.save(cache_files["diag_train"], diag_train)
    np.save(cache_files["diag_val"], diag_val)
    np.save(cache_files["num_diag"], np.array(num_diag_classes))

    return meta_train, meta_val, meta_test, diag_train, diag_val, num_diag_classes


def get_dataloaders(
    batch_size=32, image_size=384, num_workers=4, load_cached_data=True
):
    """
    Main function to prepare DataLoaders.
    """
    seed_everything(42)

    # Load Metadata CSVs
    try:
        df_train = pd.read_csv("./metadata/train.csv")
        df_val = pd.read_csv("./metadata/val.csv")
        df_test = pd.read_csv("./metadata/test.csv")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Metadata files not found in ./metadata. Error: {e}")

    # Process Metadata
    meta_train, meta_val, meta_test, diag_train, diag_val, num_diag_classes = (
        process_metadata(df_train, df_val, df_test, load_cached_data=load_cached_data)
    )

    # Inject diagnosis indices into dataframes for Dataset consumption
    df_train["diagnosis_idx"] = diag_train
    df_val["diagnosis_idx"] = diag_val

    # Get Transforms
    train_transforms, val_transforms = get_transforms(image_size)

    # Instantiate Datasets
    train_dataset = MelanomaDataset(
        csv=df_train, mode="train", meta_features=meta_train, transform=train_transforms
    )

    val_dataset = MelanomaDataset(
        csv=df_val, mode="val", meta_features=meta_val, transform=val_transforms
    )

    test_dataset = MelanomaDataset(
        csv=df_test, mode="test", meta_features=meta_test, transform=val_transforms
    )

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, num_diag_classes
