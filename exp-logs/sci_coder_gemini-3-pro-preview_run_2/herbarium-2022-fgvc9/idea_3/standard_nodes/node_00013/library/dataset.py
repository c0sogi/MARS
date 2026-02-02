import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import LabelEncoder

# Import from provided library files
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("dataset")


def get_transforms(data_split="train"):
    """
    Returns the Albumentations transform pipeline for the specified data split.

    Args:
        data_split (str): "train", "val", or "test".

    Returns:
        A.Compose: The transform pipeline.
    """
    img_size = Config.IMAGE_SIZE

    # ImageNet normalization statistics
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if data_split == "train":
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.75
                ),
                # CoarseDropout acts as a regularization similar to Cutout
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(img_size * 0.1),
                    max_width=int(img_size * 0.1),
                    min_holes=1,
                    min_height=int(img_size * 0.05),
                    min_width=int(img_size * 0.05),
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test: Resize and Normalize only
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def process_taxonomy(load_cached_data=True):
    """
    Processes the taxonomy metadata to create mappings for species, genus, and family.
    Handles caching to parquet to save time/ensure determinism.

    Returns:
        pd.DataFrame: DataFrame containing 'category_id', 'species_idx', 'genus_idx', 'family_idx'.
        dict: Metadata about number of classes for each level.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "taxonomy_mapping.parquet")

    # Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading taxonomy mapping from cache: {cache_path}")
        try:
            mapping_df = pd.read_parquet(cache_path)
            # Reconstruct counts
            meta_counts = {
                "num_species": mapping_df["species_idx"].max() + 1,
                "num_genera": mapping_df["genus_idx"].max() + 1,
                "num_families": mapping_df["family_idx"].max() + 1,
            }
            return mapping_df, meta_counts
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Recomputing...")

    # Compute from scratch
    logger.info("Processing taxonomy metadata from json...")

    if not os.path.exists(Config.TRAIN_META_JSON):
        raise FileNotFoundError(f"Metadata file not found: {Config.TRAIN_META_JSON}")

    with open(Config.TRAIN_META_JSON, "r") as f:
        data = json.load(f)

    categories = data.get("categories", [])
    df_tax = pd.DataFrame(categories)

    # Ensure category_id is int
    df_tax["category_id"] = df_tax["category_id"].astype(int)

    # Encode Genus and Family
    # We sort to ensure deterministic encoding
    genus_encoder = LabelEncoder()
    family_encoder = LabelEncoder()

    df_tax = df_tax.sort_values("category_id")

    df_tax["genus_idx"] = genus_encoder.fit_transform(df_tax["genus"])
    df_tax["family_idx"] = family_encoder.fit_transform(df_tax["family"])

    # Map category_id (which might have gaps) to contiguous species_idx
    # The competition data has 15501 taxa.
    # We create a mapping: raw_id -> contiguous_idx
    unique_ids = sorted(df_tax["category_id"].unique())
    id_map = {uid: i for i, uid in enumerate(unique_ids)}
    df_tax["species_idx"] = df_tax["category_id"].map(id_map)

    # Select relevant columns
    mapping_df = df_tax[
        ["category_id", "species_idx", "genus_idx", "family_idx"]
    ].copy()

    # Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    mapping_df.to_parquet(cache_path, index=False)
    logger.info(f"Saved taxonomy mapping to {cache_path}")

    meta_counts = {
        "num_species": len(unique_ids),
        "num_genera": len(genus_encoder.classes_),
        "num_families": len(family_encoder.classes_),
    }

    return mapping_df, meta_counts


class PlantDataset(Dataset):
    def __init__(self, df, transforms=None, taxonomy_map=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): Dataframe containing image paths and labels (or image_ids).
            transforms (A.Compose): Albumentations transforms.
            taxonomy_map (pd.DataFrame): Mapping from category_id to hierarchical indices.
            is_test (bool): If True, returns image_id instead of labels.
        """
        self.df = df
        self.transforms = transforms
        self.taxonomy_map = taxonomy_map
        self.is_test = is_test

        # Create a fast lookup dictionary for taxonomy if not testing
        self.tax_lookup = {}
        if not self.is_test and self.taxonomy_map is not None:
            # key: category_id (label), value: (species_idx, genus_idx, family_idx)
            for _, row in self.taxonomy_map.iterrows():
                self.tax_lookup[row["category_id"]] = (
                    row["species_idx"],
                    row["genus_idx"],
                    row["family_idx"],
                )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image Path Construction
        # The dataframe contains relative paths from input/
        img_path = os.path.join(Config.INPUT_DIR, row["image_path"])

        # Load Image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing images (should ideally not happen given validation)
            # Create a black image
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        if self.is_test:
            # Return image and ID for submission
            return image, str(row["image_id"])
        else:
            # Get Raw Label
            raw_label = int(row["label"])

            # Lookup Hierarchical Labels
            if raw_label in self.tax_lookup:
                species_idx, genus_idx, family_idx = self.tax_lookup[raw_label]
            else:
                # Fallback if label not in taxonomy (should not happen)
                species_idx, genus_idx, family_idx = 0, 0, 0

            return (
                image,
                torch.tensor(species_idx),
                torch.tensor(genus_idx),
                torch.tensor(family_idx),
            )


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.
    Implements Square-Root Sampling for the training set.

    Args:
        debug (bool): If True, subsets data for quick debugging.
        load_cached_data (bool): Whether to use cached taxonomy/metadata.

    Returns:
        tuple: (train_loader, val_loader, test_loader, meta_counts)
    """
    # 1. Load Taxonomy Mapping
    tax_map, meta_counts = process_taxonomy(load_cached_data=load_cached_data)

    # 2. Load Metadata CSVs
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    if debug:
        logger.info(f"Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        train_df = train_df.sample(
            n=min(len(train_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(len(test_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    # 3. Create Datasets
    train_dataset = PlantDataset(
        train_df,
        transforms=get_transforms("train"),
        taxonomy_map=tax_map,
        is_test=False,
    )

    val_dataset = PlantDataset(
        val_df, transforms=get_transforms("val"), taxonomy_map=tax_map, is_test=False
    )

    test_dataset = PlantDataset(
        test_df, transforms=get_transforms("test"), taxonomy_map=None, is_test=True
    )

    # 4. Implement Sampling Strategy (Square Root Sampling)
    sampler = None
    if Config.SAMPLING_STRATEGY == "sqrt" and not debug:
        logger.info("Computing weights for Square-Root Sampling...")

        # Count instances per class (raw label)
        class_counts = train_df["label"].value_counts().to_dict()

        # Calculate weight for each sample
        # Weight(class) = 1 / sqrt(count(class))
        # This balances between uniform (1/count) and natural (1)
        weights = []
        for label in train_df["label"]:
            count = class_counts.get(label, 1)
            weight = 1.0 / np.sqrt(count)
            weights.append(weight)

        weights = torch.DoubleTensor(weights)
        sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )
        logger.info("Sampler initialized.")

    # 5. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=(sampler is None),  # Shuffle only if no sampler
        sampler=sampler,
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

    logger.info(
        f"DataLoaders created. Train: {len(train_loader)} batches, Val: {len(val_loader)} batches."
    )
    logger.info(
        f"Classes: {meta_counts['num_species']} Species, {meta_counts['num_genera']} Genera, {meta_counts['num_families']} Families."
    )

    return train_loader, val_loader, test_loader, meta_counts
