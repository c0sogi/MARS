import os
import json
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import set_seed


def get_taxonomy_mapping(load_cached_data=True):
    """
    Creates or loads a mapping from species (category_id) to family (family_id).

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (species_to_family_dict, num_families)
            species_to_family_dict: dict mapping category_id (int) -> family_id (int)
            num_families: int, total number of unique families
    """
    cache_path = Config.FAMILY_MAPPING_CACHE

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df_map = pd.read_parquet(cache_path)
            # Convert to dictionary
            mapping = pd.Series(
                df_map.family_id.values, index=df_map.category_id
            ).to_dict()
            num_families = df_map.family_id.nunique()
            print(f"Loaded taxonomy mapping from cache. {num_families} families found.")
            return mapping, num_families
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # Compute from scratch
    print("Computing taxonomy mapping from metadata...")
    with open(Config.TRAIN_METADATA_JSON, "r") as f:
        meta = json.load(f)

    categories = meta["categories"]
    # Create DataFrame: category_id, family
    cat_df = pd.DataFrame(categories)

    # We need 'id' (species) and 'family'
    # Ensure columns exist
    if "id" not in cat_df.columns or "family" not in cat_df.columns:
        raise ValueError("Metadata categories missing 'id' or 'family' fields.")

    # Encode families to integers
    unique_families = sorted(cat_df["family"].unique())
    family_to_idx = {fam: i for i, fam in enumerate(unique_families)}

    cat_df["family_id"] = cat_df["family"].map(family_to_idx)

    # Create mapping dict
    mapping = pd.Series(cat_df.family_id.values, index=cat_df.id).to_dict()
    num_families = len(unique_families)

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    save_df = cat_df[["id", "family_id"]].rename(columns={"id": "category_id"})
    save_df.to_parquet(cache_path, index=False)

    print(
        f"Computed taxonomy mapping. {num_families} families found. Saved to {cache_path}"
    )

    return mapping, num_families


def undersample_data(df):
    """
    Performs hard undersampling on the dataframe.
    Caps the number of samples per category_id to Config.MAX_SAMPLES_PER_CLASS.
    """
    print(
        f"Undersampling: Capping samples at {Config.MAX_SAMPLES_PER_CLASS} per class..."
    )
    original_len = len(df)

    # Shuffle first to ensure random selection of the subset
    df = df.sample(frac=1, random_state=Config.SEED).reset_index(drop=True)

    # Filter using groupby and cumcount
    # This is generally faster than groupby().head() for large dataframes
    df["cum_count"] = df.groupby("category_id").cumcount()
    df_filtered = df[df["cum_count"] < Config.MAX_SAMPLES_PER_CLASS].drop(
        columns=["cum_count"]
    )

    print(f"Data reduced from {original_len} to {len(df_filtered)} samples.")
    return df_filtered


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for train or val/test.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
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
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class HerbariumDataset(Dataset):
    def __init__(self, df, transform=None, taxonomy_map=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing file paths and labels.
            transform (albumentations.Compose): Transforms to apply.
            taxonomy_map (dict): Mapping from category_id to family_id. Required for train/val.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.taxonomy_map = taxonomy_map
        self.mode = mode

        # Pre-compute full paths to avoid string concatenation in __getitem__
        self.file_paths = [
            os.path.join(Config.INPUT_DIR, p) for p in self.df["file_path"].values
        ]

        if self.mode != "test":
            self.species_labels = self.df["category_id"].values

            # Map species to family
            if self.taxonomy_map is None:
                raise ValueError("taxonomy_map must be provided for train/val modes")

            # Handle potential missing keys if dataset has species not in map (should not happen with valid split)
            self.family_labels = np.array(
                [self.taxonomy_map.get(s, -1) for s in self.species_labels]
            )
        else:
            self.image_ids = self.df["image_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        # Load image
        image = cv2.imread(path)
        if image is None:
            # Fallback for missing images (though metadata check passed)
            # Create a black image to prevent crash
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to tensor conversion if no transform provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        if self.mode == "test":
            image_id = self.image_ids[idx]
            return image, image_id
        else:
            species_label = self.species_labels[idx]
            family_label = self.family_labels[idx]
            return (
                image,
                torch.tensor(species_label, dtype=torch.long),
                torch.tensor(family_label, dtype=torch.long),
            )


def get_dataloaders(debug=False):
    """
    Prepares DataLoaders for train, val, and test.

    Args:
        debug (bool): If True, truncates datasets for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader, num_families)
    """
    set_seed(Config.SEED)

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    if debug or Config.DEBUG:
        print(f"Debug mode: Truncating datasets to {Config.DEBUG_SAMPLES} samples.")
        train_df = train_df.iloc[: Config.DEBUG_SAMPLES]
        val_df = val_df.iloc[: Config.DEBUG_SAMPLES]
        test_df = test_df.iloc[: Config.DEBUG_SAMPLES]

    # 2. Get Taxonomy Mapping
    taxonomy_map, num_families = get_taxonomy_mapping(load_cached_data=True)

    # 3. Undersample Training Data
    train_df = undersample_data(train_df)

    # 4. Create Datasets
    train_dataset = HerbariumDataset(
        train_df,
        transform=get_transforms("train"),
        taxonomy_map=taxonomy_map,
        mode="train",
    )

    val_dataset = HerbariumDataset(
        val_df, transform=get_transforms("val"), taxonomy_map=taxonomy_map, mode="val"
    )

    test_dataset = HerbariumDataset(
        test_df, transform=get_transforms("test"), taxonomy_map=None, mode="test"
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader, num_families
