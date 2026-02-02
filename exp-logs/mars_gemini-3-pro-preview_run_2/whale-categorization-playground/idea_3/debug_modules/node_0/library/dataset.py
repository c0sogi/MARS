import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class WhaleDataset(Dataset):
    """
    Dataset class for Whale Identification.
    """

    def __init__(self, images, targets=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images with shape (N, H, W, 3).
            targets (np.ndarray, optional): Array of labels/targets.
            transform (albumentations.Compose, optional): Augmentations.
        """
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.targets is not None:
            label = self.targets[idx]
            return {"image": image, "label": torch.tensor(label, dtype=torch.long)}
        else:
            return {"image": image}


def get_transforms(phase):
    """
    Returns Albumentations transforms for the specified phase.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test / Gallery
        return A.Compose(
            [
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )


def load_and_process_images(df, cache_path, load_cached_data=True):
    """
    Loads images from disk or cache.

    Args:
        df (pd.DataFrame): DataFrame containing 'file_path'.
        cache_path (str): Path to the .npy cache file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: Array of images (N, H, W, 3) in uint8.
    """
    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading images from cache: {cache_path}")
        try:
            images = np.load(cache_path)
            if len(images) == len(df):
                return images
            else:
                print(
                    f"Cache size mismatch ({len(images)} vs {len(df)}). Reloading from raw files."
                )
        except Exception as e:
            print(f"Error loading cache: {e}. Reloading from raw files.")

    print(f"Processing {len(df)} images from raw files...")
    images = []

    # Pre-allocate array if possible to save memory fragmentation, but list append is safer for variable success
    # given the small dataset size (relative to RAM), list append is fine.

    for idx, row in df.iterrows():
        file_path = os.path.join(Config.input_dir, row["file_path"])

        # Read image
        img = cv2.imread(file_path)
        if img is None:
            # Fallback for missing images: create black image
            print(f"Warning: Image not found at {file_path}. Using black placeholder.")
            img = np.zeros((Config.img_size, Config.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[:2] != (Config.img_size, Config.img_size):
                img = cv2.resize(
                    img,
                    (Config.img_size, Config.img_size),
                    interpolation=cv2.INTER_CUBIC,
                )

        images.append(img)

    images = np.array(images, dtype=np.uint8)

    # Save to cache
    print(f"Saving images to cache: {cache_path}")
    np.save(cache_path, images)

    return images


def get_label_encoder(train_df):
    """
    Creates a mapping from string Ids to integers.
    'new_whale' is mapped to -1.
    All other known whales are mapped to 0...N-1.
    """
    unique_ids = sorted(train_df["Id"].unique())

    # Remove new_whale from the sorted list to assign it a special ID later
    if "new_whale" in unique_ids:
        unique_ids.remove("new_whale")

    # Create mapping
    id2label = {label: i for i, label in enumerate(unique_ids)}
    id2label["new_whale"] = -1

    return id2label


def get_loaders(debug=False, load_cached_data=True):
    """
    Prepares DataLoaders for Training, Validation, and Testing.

    Returns:
        train_loader: DataLoader for training (filtered, augmented).
        gallery_loader: DataLoader for gallery (all train images, no aug).
        val_loader: DataLoader for validation query.
        test_loader: DataLoader for test query.
        id2label: Dictionary mapping ID strings to integers.
    """
    # 1. Load Metadata
    df_train = pd.read_csv(Config.meta_train_path)
    df_val = pd.read_csv(Config.meta_val_path)
    df_test = pd.read_csv(Config.meta_test_path)

    if debug:
        print("Debug mode: Subsampling datasets.")
        df_train = df_train.iloc[: Config.debug_sample_size]
        df_val = df_val.iloc[: Config.debug_sample_size]
        df_test = df_test.iloc[: Config.debug_sample_size]

    # 2. Prepare Images (Load or Cache)
    # Note: We use different cache filenames for debug mode implicitly if we wanted,
    # but here we just load full cache and slice if debug is False,
    # OR if debug is True we might overwrite cache with small data.
    # To be safe: If debug is True, we do NOT use the main cache file to avoid corruption.

    if debug:
        # Use temporary paths or just don't load from main cache
        train_cache = os.path.join(
            Config.working_dir, f"debug_train_images_{Config.img_size}.npy"
        )
        val_cache = os.path.join(
            Config.working_dir, f"debug_val_images_{Config.img_size}.npy"
        )
        test_cache = os.path.join(
            Config.working_dir, f"debug_test_images_{Config.img_size}.npy"
        )
    else:
        train_cache = Config.cache_train_images
        val_cache = Config.cache_val_images
        test_cache = Config.cache_test_images

    X_train_all = load_and_process_images(df_train, train_cache, load_cached_data)
    X_val = load_and_process_images(df_val, val_cache, load_cached_data)
    X_test = load_and_process_images(df_test, test_cache, load_cached_data)

    # 3. Encode Labels
    # We always build encoder from the full training metadata to ensure consistency
    # regardless of debug slicing (though in debug, some IDs might be missing from the batch).
    full_train_meta = pd.read_csv(Config.meta_train_path)
    id2label = get_label_encoder(full_train_meta)

    y_train_all = df_train["Id"].map(id2label).values.astype(np.int32)
    y_val = df_val["Id"].map(id2label).values.astype(np.int32)
    # Test has no labels

    # 4. Prepare Datasets

    # A. Training Set (Clean): Exclude 'new_whale' (-1)
    # This dataset is used for training the ArcFace model
    clean_mask = y_train_all != -1
    X_train_clean = X_train_all[clean_mask]
    y_train_clean = y_train_all[clean_mask]

    train_dataset = WhaleDataset(
        images=X_train_clean, targets=y_train_clean, transform=get_transforms("train")
    )

    # B. Gallery Set: All training images (including new_whale)
    # Used as the reference database during inference/validation
    gallery_dataset = WhaleDataset(
        images=X_train_all,
        targets=y_train_all,
        transform=get_transforms("val"),  # No augmentation
    )

    # C. Validation Set
    val_dataset = WhaleDataset(
        images=X_val, targets=y_val, transform=get_transforms("val")
    )

    # D. Test Set
    test_dataset = WhaleDataset(
        images=X_test, targets=None, transform=get_transforms("val")
    )

    # 5. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return train_loader, gallery_loader, val_loader, test_loader, id2label
