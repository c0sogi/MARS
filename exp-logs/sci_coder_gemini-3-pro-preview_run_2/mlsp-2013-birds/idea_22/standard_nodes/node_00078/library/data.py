import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config

# -----------------------------------------------------------------------------
# Custom Augmentations
# -----------------------------------------------------------------------------


class TimeRoll(A.ImageOnlyTransform):
    """
    Cyclic Time-Rolling Augmentation.
    Shifts the spectrogram along the time axis (width) wrapping around.
    This preserves the temporal patterns while introducing invariance to start time.
    """

    def __init__(self, always_apply=False, p=0.5):
        super(TimeRoll, self).__init__(always_apply, p)

    def apply(self, img, **params):
        # img shape is (Height, Width, Channels)
        # We roll along axis 1 (Width/Time)
        shift = np.random.randint(0, img.shape[1])
        return np.roll(img, shift, axis=1)

    def get_transform_init_args_names(self):
        return ()


# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------


class BirdDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_cache: dict,
        transforms: A.Compose = None,
        soft_targets: dict = None,
        is_test: bool = False,
    ):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing 'rec_id', 'file_path_spec', and labels.
            image_cache (dict): Dictionary mapping rec_id to loaded numpy image arrays.
            transforms (A.Compose): Albumentations transforms to apply.
            soft_targets (dict, optional): Dictionary mapping rec_id to soft probability vectors (for distillation).
            is_test (bool): If True, returns dummy targets.
        """
        self.df = df
        self.image_cache = image_cache
        self.transforms = transforms
        self.soft_targets = soft_targets
        self.is_test = is_test

        # Identify label columns (species_0 to species_18)
        self.label_cols = [c for c in df.columns if c.startswith("species_")]
        self.num_classes = len(self.label_cols)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rec_id = int(row["rec_id"])

        # 1. Load Image
        # Retrieve from cache (guaranteed to be present by get_loaders logic)
        image = self.image_cache[rec_id]

        # 2. Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback to ToTensor if no transforms provided
            image = ToTensorV2()(image=image)["image"]

        # 3. Get Hard Targets
        if self.is_test:
            # Dummy targets for test set
            hard_target = torch.zeros(self.num_classes, dtype=torch.float32)
        else:
            hard_target = torch.tensor(
                row[self.label_cols].values.astype(np.float32), dtype=torch.float32
            )

        # 4. Get Soft Targets (for Distillation)
        if self.soft_targets is not None and rec_id in self.soft_targets:
            soft_target = torch.tensor(self.soft_targets[rec_id], dtype=torch.float32)
        else:
            # Return zeros if no soft targets available (e.g. Stage 1 or Test)
            soft_target = torch.zeros(self.num_classes, dtype=torch.float32)

        return image, hard_target, soft_target, torch.tensor(rec_id, dtype=torch.long)


# -----------------------------------------------------------------------------
# Data Processing & Caching
# -----------------------------------------------------------------------------


def load_and_cache_images(df_list, load_cached_data=True):
    """
    Loads all images referenced in the dataframes into memory.
    Implements caching using .npy files in the working directory.

    Args:
        df_list (list): List of dataframes (train, val, test) to collect rec_ids from.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Mapping of rec_id -> numpy image array (RGB).
    """
    cache_path = os.path.join(Config.WORKING_DIR, "images_cache.npy")
    ids_path = os.path.join(Config.WORKING_DIR, "ids_cache.npy")

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_path) and os.path.exists(ids_path):
        try:
            images_arr = np.load(cache_path)
            ids_arr = np.load(ids_path)
            # Reconstruct dictionary
            image_cache = {rec_id: img for rec_id, img in zip(ids_arr, images_arr)}
            print(f"Loaded {len(image_cache)} images from cache.")
            return image_cache
        except Exception as e:
            print(f"Cache load failed: {e}. Reloading from disk.")

    # 2. Process from Scratch
    print("Loading images from disk...")
    image_cache = {}

    # Combine all unique paths from all dataframes
    all_df = pd.concat(df_list, ignore_index=True).drop_duplicates(subset=["rec_id"])

    processed_images = []
    processed_ids = []

    for idx, row in all_df.iterrows():
        rec_id = int(row["rec_id"])

        # Construct path to Filtered Spectrograms
        # Metadata path: supplemental_data/spectrograms/filename.bmp
        # Target path: input/supplemental_data/filtered_spectrograms/filename.bmp
        rel_path = row["file_path_spec"]
        # Replace folder name and prepend input dir
        rel_path = rel_path.replace("spectrograms", "filtered_spectrograms")
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            # Fallback to original spectrograms if filtered not found (should not happen based on task desc)
            full_path = os.path.join(Config.INPUT_DIR, row["file_path_spec"])

        # Load Image
        img = cv2.imread(full_path, cv2.IMREAD_COLOR)  # Load as BGR
        if img is None:
            # Create black image if file is corrupt/missing
            img = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image_cache[rec_id] = img
        processed_images.append(img)
        processed_ids.append(rec_id)

    # 3. Save to Cache
    try:
        np.save(cache_path, np.array(processed_images))
        np.save(ids_path, np.array(processed_ids))
        print(f"Cached {len(image_cache)} images to {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")

    return image_cache


# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------


def get_transforms(data="train"):
    """
    Returns Albumentations transforms for training or validation.

    Args:
        data (str): 'train' or 'valid'.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_HEIGHT, Config.IMG_WIDTH),
                # Cyclic Time-Rolling (Horizontal Shift)
                TimeRoll(p=0.5),
                # SpecAugment-like Masking
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(Config.IMG_HEIGHT * 0.1),
                    max_width=int(Config.IMG_WIDTH * 0.1),
                    min_holes=2,
                    fill_value=0,
                    p=0.5,
                ),
                # Normalization (ImageNet stats for RGB)
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    elif data == "valid":
        return A.Compose(
            [
                A.Resize(Config.IMG_HEIGHT, Config.IMG_WIDTH),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


# -----------------------------------------------------------------------------
# Data Loaders
# -----------------------------------------------------------------------------


def get_loaders(
    fold: int = 0,
    load_cached_data: bool = True,
    soft_targets: dict = None,
    debug: bool = Config.DEBUG,
):
    """
    Prepares DataLoaders for Training, Validation, and Testing.

    Args:
        fold (int): Fold index for validation (not strictly used here as splits are fixed in metadata,
                    but kept for interface consistency).
        load_cached_data (bool): Whether to use cached images.
        soft_targets (dict): Dictionary of soft targets for distillation.
        debug (bool): If True, subsamples datasets for rapid debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    if debug:
        df_train = df_train.head(Config.DEBUG_SUBSET_SIZE)
        df_val = df_val.head(Config.DEBUG_SUBSET_SIZE)
        df_test = df_test.head(Config.DEBUG_SUBSET_SIZE)

    # 2. Load Images into Memory (Cache)
    image_cache = load_and_cache_images(
        [df_train, df_val, df_test], load_cached_data=load_cached_data
    )

    # 3. Create Datasets
    train_dataset = BirdDataset(
        df=df_train,
        image_cache=image_cache,
        transforms=get_transforms(data="train"),
        soft_targets=soft_targets,
        is_test=False,
    )

    val_dataset = BirdDataset(
        df=df_val,
        image_cache=image_cache,
        transforms=get_transforms(data="valid"),
        soft_targets=soft_targets,  # Soft targets usually not used in val, but harmless
        is_test=False,
    )

    test_dataset = BirdDataset(
        df=df_test,
        image_cache=image_cache,
        transforms=get_transforms(data="valid"),
        soft_targets=None,
        is_test=True,
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
