import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from library.config import Config


def get_transforms(image_size: int, mode: str = "train") -> A.Compose:
    """
    Returns the Albumentations transformation pipeline.

    Args:
        image_size (int): The spatial resolution (height/width) for resizing.
        mode (str): 'train' for augmentations, 'val' or 'test' for deterministic resizing.

    Returns:
        A.Compose: The composition of transforms.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def process_metadata(config: Config, load_cached_data: bool = True):
    """
    Loads and processes metadata. Handles One-Hot Encoding, Standardization, and Label Encoding.
    Implements caching to ./working/idea_5/ using .npy files.

    Args:
        config (Config): Configuration object containing paths.
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        tuple: (train_data, val_data, test_data, num_diag_classes)
               Each *_data is a tuple: (df, meta_features, target, diagnosis)
               For test_data, target and diagnosis are None.
    """
    # Define cache paths
    cache_dir = config.working_dir
    os.makedirs(cache_dir, exist_ok=True)

    files = {
        "meta_train": os.path.join(cache_dir, "meta_train.npy"),
        "target_train": os.path.join(cache_dir, "target_train.npy"),
        "diag_train": os.path.join(cache_dir, "diag_train.npy"),
        "meta_val": os.path.join(cache_dir, "meta_val.npy"),
        "target_val": os.path.join(cache_dir, "target_val.npy"),
        "diag_val": os.path.join(cache_dir, "diag_val.npy"),
        "meta_test": os.path.join(cache_dir, "meta_test.npy"),
        "classes": os.path.join(cache_dir, "num_diag_classes.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in files.values())

    # Load DataFrames (always needed for file paths)
    df_train = pd.read_csv(config.train_csv)
    df_val = pd.read_csv(config.val_csv)
    df_test = pd.read_csv(config.test_csv)

    if config.debug:
        df_train = df_train.head(config.debug_sample_size)
        df_val = df_val.head(config.debug_sample_size)
        df_test = df_test.head(config.debug_sample_size)

    if load_cached_data and cache_exists:
        print("Loading processed metadata from cache...")
        meta_train = np.load(files["meta_train"])
        target_train = np.load(files["target_train"])
        diag_train = np.load(files["diag_train"])

        meta_val = np.load(files["meta_val"])
        target_val = np.load(files["target_val"])
        diag_val = np.load(files["diag_val"])

        meta_test = np.load(files["meta_test"])
        num_diag_classes = int(np.load(files["classes"]))

        # Validate dimensions against current configuration
        if (
            len(meta_train) == len(df_train)
            and len(meta_val) == len(df_val)
            and len(meta_test) == len(df_test)
        ):
            return (
                (df_train, meta_train, target_train, diag_train),
                (df_val, meta_val, target_val, diag_val),
                (df_test, meta_test, None, None),
                num_diag_classes,
            )
        else:
            print(
                f"Cache dimension mismatch (Train: {len(meta_train)} vs {len(df_train)}). Re-processing..."
            )

    print("Processing metadata from scratch...")

    # --- Feature Engineering ---

    # 1. Prepare Concatenated DataFrame for Fitting
    # We need to fit encoders on the full available data to handle all categories
    # Diagnosis is only in train/val

    # Fill NaNs
    # Age: Mean imputation
    imputer_age = SimpleImputer(strategy="mean")
    # Sex/Site: Constant imputation
    imputer_cat = SimpleImputer(strategy="constant", fill_value="unknown")

    # Columns
    num_col = ["age_approx"]
    cat_cols = ["sex", "anatom_site_general_challenge"]

    # Combine for fitting
    combined_df = pd.concat([df_train, df_val, df_test], axis=0, ignore_index=True)

    # Fit Imputers
    combined_df[num_col] = imputer_age.fit_transform(combined_df[num_col])
    combined_df[cat_cols] = imputer_cat.fit_transform(combined_df[cat_cols])

    # Fit Scaler
    scaler = StandardScaler()
    scaler.fit(combined_df[num_col])

    # Fit OneHotEncoder
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe.fit(combined_df[cat_cols])

    # Fit LabelEncoder for Diagnosis (Only Train/Val)
    # Handle NaNs in diagnosis by treating them as a specific class or filling
    df_train["diagnosis"] = df_train["diagnosis"].fillna("unknown")
    df_val["diagnosis"] = df_val["diagnosis"].fillna("unknown")

    le_diag = LabelEncoder()
    # Fit on all unique diagnoses in train and val
    all_diags = pd.concat([df_train["diagnosis"], df_val["diagnosis"]]).unique()
    le_diag.fit(all_diags)
    num_diag_classes = len(le_diag.classes_)

    # --- Transform and Save ---

    def process_subset(df, is_test=False):
        # Impute
        df_c = df.copy()
        df_c[num_col] = imputer_age.transform(df_c[num_col])
        df_c[cat_cols] = imputer_cat.transform(df_c[cat_cols])

        # Scale Numerical
        num_feats = scaler.transform(df_c[num_col])

        # Encode Categorical
        cat_feats = ohe.transform(df_c[cat_cols])

        # Concatenate
        meta_feats = np.concatenate([num_feats, cat_feats], axis=1).astype(np.float32)

        if is_test:
            return meta_feats, None, None
        else:
            targets = df_c["target"].values.astype(np.float32)
            diags = le_diag.transform(df_c["diagnosis"]).astype(np.int64)
            return meta_feats, targets, diags

    meta_train, target_train, diag_train = process_subset(df_train, is_test=False)
    meta_val, target_val, diag_val = process_subset(df_val, is_test=False)
    meta_test, _, _ = process_subset(df_test, is_test=True)

    # Save to Cache
    np.save(files["meta_train"], meta_train)
    np.save(files["target_train"], target_train)
    np.save(files["diag_train"], diag_train)

    np.save(files["meta_val"], meta_val)
    np.save(files["target_val"], target_val)
    np.save(files["diag_val"], diag_val)

    np.save(files["meta_test"], meta_test)
    np.save(files["classes"], np.array(num_diag_classes))

    return (
        (df_train, meta_train, target_train, diag_train),
        (df_val, meta_val, target_val, diag_val),
        (df_test, meta_test, None, None),
        num_diag_classes,
    )


class SkinLesionDataset(Dataset):
    """
    PyTorch Dataset for Skin Lesion Classification.
    Returns:
        - image: Transformed image tensor
        - meta: Processed metadata vector
        - target: Malignancy label (float)
        - diagnosis: Diagnosis label (long)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        meta_features: np.ndarray,
        targets: np.ndarray = None,
        diagnoses: np.ndarray = None,
        transforms: A.Compose = None,
        input_root: str = "./input",
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'file_path'.
            meta_features (np.ndarray): Matrix of processed metadata features.
            targets (np.ndarray, optional): Array of malignancy labels.
            diagnoses (np.ndarray, optional): Array of diagnosis labels.
            transforms (A.Compose): Albumentations transforms.
            input_root (str): Root directory for image files.
        """
        self.df = df
        self.meta_features = meta_features
        self.targets = targets
        self.diagnoses = diagnoses
        self.transforms = transforms
        self.input_root = input_root

        # Pre-check file existence to avoid runtime errors (optional but good practice)
        # We assume metadata generation verified paths, but cv2 needs valid files.

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Image
        file_path = os.path.join(self.input_root, row["file_path"])

        # Read image
        image = cv2.imread(file_path)
        if image is None:
            # Fallback for missing/corrupt images: return a black image
            # This prevents the dataloader from crashing
            image = np.zeros((384, 384, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion if no transforms provided
            image = ToTensorV2()(image=image)["image"]

        # 3. Get Metadata
        meta = self.meta_features[idx]

        # 4. Get Targets
        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float)
        else:
            target = torch.tensor(-1.0, dtype=torch.float)  # Dummy for test

        if self.diagnoses is not None:
            diagnosis = torch.tensor(self.diagnoses[idx], dtype=torch.long)
        else:
            diagnosis = torch.tensor(0, dtype=torch.long)  # Dummy for test

        return {
            "image": image,
            "meta": torch.tensor(meta, dtype=torch.float),
            "target": target,
            "diagnosis": diagnosis,
            "image_name": row["image_name"],
        }
