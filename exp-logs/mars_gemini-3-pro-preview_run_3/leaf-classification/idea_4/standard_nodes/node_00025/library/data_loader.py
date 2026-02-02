import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library import config
from library import utils


class LeafDataset(Dataset):
    def __init__(
        self,
        file_paths,
        tabular_features,
        labels=None,
        ids=None,
        transform=None,
        multi_view=True,
    ):
        """
        PyTorch Dataset for Leaf Images and Tabular Data.

        Args:
            file_paths (list): List of relative file paths to images.
            tabular_features (np.ndarray): Matrix of tabular features (N, 192).
            labels (np.ndarray, optional): Array of class labels (N,).
            ids (np.ndarray, optional): Array of image IDs (N,).
            transform (A.Compose, optional): Albumentations transformation pipeline.
            multi_view (bool): If True, returns 4 rotated views of the image.
        """
        self.file_paths = file_paths
        self.tabular_features = tabular_features
        self.labels = labels
        self.ids = ids
        self.transform = transform
        self.multi_view = multi_view
        self.input_dir = config.INPUT_DIR

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Construct full image path
        rel_path = self.file_paths[idx]
        img_path = os.path.join(self.input_dir, rel_path)

        # Load image
        # Images are binary (black leaf on white background)
        # We convert to RGB to match standard backbone input requirements
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for safety, though data checks should prevent this
            image = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Prepare views
        views = []

        if self.multi_view:
            # Generate 4 rotated views: 0, 90, 180, 270 degrees
            rotations = [
                None,
                cv2.ROTATE_90_CLOCKWISE,
                cv2.ROTATE_180,
                cv2.ROTATE_90_COUNTERCLOCKWISE,
            ]

            for rot_code in rotations:
                img_view = image.copy()
                if rot_code is not None:
                    img_view = cv2.rotate(img_view, rot_code)

                if self.transform:
                    augmented = self.transform(image=img_view)
                    img_view = augmented["image"]
                else:
                    # Default to simple tensor conversion if no transform provided
                    img_view = (
                        torch.from_numpy(img_view.transpose(2, 0, 1)).float() / 255.0
                    )

                views.append(img_view)

            # Stack views into a single tensor: (4, C, H, W)
            image_tensor = torch.stack(views)

        else:
            # Single view mode
            if self.transform:
                augmented = self.transform(image=image)
                image_tensor = augmented["image"]
            else:
                image_tensor = (
                    torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                )

        # Process Tabular Data
        tab_feat = torch.tensor(self.tabular_features[idx], dtype=torch.float32)

        # Process Label (return -1 if not available, e.g., test set)
        label = (
            torch.tensor(self.labels[idx], dtype=torch.long)
            if self.labels is not None
            else torch.tensor(-1, dtype=torch.long)
        )

        # Process ID
        img_id = (
            torch.tensor(self.ids[idx], dtype=torch.long)
            if self.ids is not None
            else torch.tensor(-1, dtype=torch.long)
        )

        return image_tensor, tab_feat, label, img_id


def get_transforms(img_size=224):
    """
    Returns the Albumentations transformation pipeline.
    """
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def load_tabular_data(split, load_cached_data=True):
    """
    Loads tabular data, labels, and IDs. Handles caching to disk.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X, y, ids, classes)
            X (np.ndarray): Tabular features (N, 192).
            y (np.ndarray or None): Encoded labels. None for test split.
            ids (np.ndarray): Image IDs.
            classes (np.ndarray): Array of class names.
    """
    cache_dir = config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    X_path = os.path.join(cache_dir, f"X_{split}.npy")
    y_path = os.path.join(cache_dir, f"y_{split}.npy")
    ids_path = os.path.join(cache_dir, f"ids_{split}.npy")
    classes_path = os.path.join(cache_dir, "classes.npy")

    # 1. Try loading from cache
    if load_cached_data:
        # Check if essential files exist
        if os.path.exists(X_path) and os.path.exists(ids_path):
            # Check y existence (required for train/val)
            if split == "test" or os.path.exists(y_path):
                # Check classes existence
                if os.path.exists(classes_path):
                    X = np.load(X_path)
                    ids = np.load(ids_path)
                    classes = np.load(classes_path, allow_pickle=True)
                    y = np.load(y_path) if split != "test" else None
                    return X, y, ids, classes

    # 2. Process from scratch if cache miss or load_cached_data is False

    # Determine which metadata file to read
    if split == "train":
        meta_path = config.TRAIN_META_PATH
    elif split == "val":
        meta_path = config.VAL_META_PATH
    elif split == "test":
        meta_path = config.TEST_META_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    df = pd.read_csv(meta_path)

    # Extract IDs
    ids = df["id"].values

    # Extract Tabular Features
    # We identify columns by prefix to ensure we get the correct 192 features
    margin_cols = [c for c in df.columns if c.startswith(config.MARGIN_COLS_PREFIX)]
    shape_cols = [c for c in df.columns if c.startswith(config.SHAPE_COLS_PREFIX)]
    texture_cols = [c for c in df.columns if c.startswith(config.TEXTURE_COLS_PREFIX)]

    # Concatenate feature groups
    feature_cols = margin_cols + shape_cols + texture_cols
    X = df[feature_cols].values.astype(np.float32)

    # Extract Labels and Classes
    y = None

    # Always derive class mapping from the training set to ensure consistency
    # regardless of which split is being processed.
    train_df = pd.read_csv(config.TRAIN_META_PATH)
    classes = np.sort(train_df["species"].unique())

    if split != "test":
        if "species" not in df.columns:
            raise ValueError(f"Species column missing in {split} metadata.")

        species = df["species"].values
        # Encode labels using the fixed class list
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        y = np.array([class_to_idx[s] for s in species], dtype=np.int64)

    # 3. Save to cache
    np.save(X_path, X)
    np.save(ids_path, ids)
    np.save(classes_path, classes)  # Overwrites, ensuring latest mapping
    if y is not None:
        np.save(y_path, y)

    return X, y, ids, classes


def get_dataloaders(batch_size=32, load_cached_data=True, num_workers=4):
    """
    Constructs DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size for the dataloaders.
        load_cached_data (bool): Whether to use cached processed data.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        tuple: (train_loader, val_loader, test_loader, classes)
    """
    # Ensure reproducibility
    utils.seed_everything(config.SEED)

    # Load processed tabular data and labels
    X_train, y_train, ids_train, classes = load_tabular_data("train", load_cached_data)
    X_val, y_val, ids_val, _ = load_tabular_data("val", load_cached_data)
    X_test, _, ids_test, _ = load_tabular_data("test", load_cached_data)

    # Load metadata dataframes to get image file paths
    # The order of rows in metadata CSVs matches the order of X/y arrays loaded above
    df_train = pd.read_csv(config.TRAIN_META_PATH)
    df_val = pd.read_csv(config.VAL_META_PATH)
    df_test = pd.read_csv(config.TEST_META_PATH)

    # Sanity check to ensure alignment
    if not np.array_equal(df_train["id"].values, ids_train):
        raise RuntimeError(
            "ID mismatch between training metadata and cached tabular data."
        )

    # Prepare transforms
    transforms = get_transforms(config.IMG_SIZE)

    # Instantiate Datasets
    # We enable multi_view for all sets to support the averaging ensemble strategy

    train_dataset = LeafDataset(
        file_paths=df_train["file_path"].tolist(),
        tabular_features=X_train,
        labels=y_train,
        ids=ids_train,
        transform=transforms,
        multi_view=True,
    )

    val_dataset = LeafDataset(
        file_paths=df_val["file_path"].tolist(),
        tabular_features=X_val,
        labels=y_val,
        ids=ids_val,
        transform=transforms,
        multi_view=True,
    )

    test_dataset = LeafDataset(
        file_paths=df_test["file_path"].tolist(),
        tabular_features=X_test,
        labels=None,
        ids=ids_test,
        transform=transforms,
        multi_view=True,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
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

    return train_loader, val_loader, test_loader, classes
