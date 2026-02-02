import os
import json
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import get_logger

logger = get_logger(__name__)


def get_transforms(phase: str):
    """
    Returns the Albumentations transform pipeline for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if phase == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Simulate RandAugment (N=2) with various transformations
                A.SomeOf(
                    [
                        A.ShiftScaleRotate(
                            scale_limit=0.2, rotate_limit=30, shift_limit=0.1, p=1.0
                        ),
                        A.RandomBrightnessContrast(
                            brightness_limit=0.2, contrast_limit=0.2, p=1.0
                        ),
                        A.HueSaturationValue(
                            hue_shift_limit=20,
                            sat_shift_limit=30,
                            val_shift_limit=20,
                            p=1.0,
                        ),
                        A.RGBShift(
                            r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0
                        ),
                        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
                        A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1.0),
                    ],
                    n=Config.RA_N,
                    p=1.0,
                ),
                # CoarseDropout
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(Config.IMG_SIZE * 0.1),
                    max_width=int(Config.IMG_SIZE * 0.1),
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def process_taxonomy(load_cached_data=True):
    """
    Parses train_metadata.json to create a mapping from category_id (species)
    to genus_id and family_id. Caches the result to parquet.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: DataFrame indexed by category_id with columns [genus_id, family_id].
        dict: Metadata about counts (num_families, num_genera).
    """
    os.makedirs(Config.WORK_DIR, exist_ok=True)
    cache_path = Config.TAXONOMY_MAP_PATH

    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading taxonomy mapping from cache: {cache_path}")
        try:
            df_map = pd.read_parquet(cache_path)
            # Reconstruct counts from max ID + 1
            meta_counts = {
                "num_families": df_map["family_id"].max() + 1,
                "num_genera": df_map["genus_id"].max() + 1,
            }
            return df_map, meta_counts
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Recomputing...")

    logger.info("Processing taxonomy from train_metadata.json...")

    if not os.path.exists(Config.TRAIN_META_JSON):
        raise FileNotFoundError(f"Metadata file not found: {Config.TRAIN_META_JSON}")

    with open(Config.TRAIN_META_JSON, "r") as f:
        data = json.load(f)

    categories = data.get("categories", [])
    if not categories:
        raise ValueError("No categories found in metadata.")

    # Create DataFrame
    df = pd.DataFrame(categories)
    # Ensure category_id is int
    df["category_id"] = df["category_id"].astype(int)

    # Encode Family and Genus to Integers
    # Sort to ensure deterministic encoding
    unique_families = sorted(df["family"].unique())
    unique_genera = sorted(df["genus"].unique())

    fam_map = {name: i for i, name in enumerate(unique_families)}
    gen_map = {name: i for i, name in enumerate(unique_genera)}

    df["family_id"] = df["family"].map(fam_map)
    df["genus_id"] = df["genus"].map(gen_map)

    # Select only needed columns and set index
    df_map = (
        df[["category_id", "family_id", "genus_id"]]
        .set_index("category_id")
        .sort_index()
    )

    # Save to cache
    logger.info(f"Saving taxonomy mapping to {cache_path}")
    df_map.to_parquet(cache_path)

    meta_counts = {
        "num_families": len(unique_families),
        "num_genera": len(unique_genera),
    }

    return df_map, meta_counts


class PlantTaxonomyDataset(Dataset):
    """
    Dataset for Plant Classification.
    Returns:
        Train/Val: image, (species_label, genus_label, family_label)
        Test: image, image_id
    """

    def __init__(self, df, taxonomy_map=None, transforms=None, phase="train"):
        self.df = df
        self.taxonomy_map = taxonomy_map
        self.transforms = transforms
        self.phase = phase
        self.input_dir = Config.INPUT_DIR

        # Pre-fetch taxonomy info for faster access if available
        self.labels = None
        if self.phase in ["train", "val"] and self.taxonomy_map is not None:
            # Join labels with taxonomy
            # df has 'label' (species_id), map has index 'category_id'
            # We create a numpy array for fast lookup: [N_samples, 3] -> (species, genus, family)

            # Ensure df 'label' matches map index
            temp_df = self.df.copy()
            # Map genus and family
            # We use a dictionary lookup or merge. Merge is safer.
            temp_df = temp_df.merge(
                self.taxonomy_map, left_on="label", right_index=True, how="left"
            )

            # Fill NaNs if any (should not happen if metadata is complete)
            if temp_df[["family_id", "genus_id"]].isnull().any().any():
                logger.warning(
                    "Some labels in dataset not found in taxonomy map. Filling with 0."
                )
                temp_df.fillna(0, inplace=True)

            self.species_ids = temp_df["label"].values.astype(np.int64)
            self.genus_ids = temp_df["genus_id"].values.astype(np.int64)
            self.family_ids = temp_df["family_id"].values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image Loading
        img_path = os.path.join(self.input_dir, row["image_path"])
        image = cv2.imread(img_path)

        if image is None:
            # Fallback for missing images (should be handled by metadata check, but for safety)
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Transform
        if self.transforms:
            res = self.transforms(image=image)
            image = res["image"]

        # Return
        if self.phase == "test":
            # row['image_id'] is required for submission
            return image, str(row["image_id"])
        else:
            # Train/Val
            species = self.species_ids[idx]
            genus = self.genus_ids[idx]
            family = self.family_ids[idx]
            return image, (species, genus, family)


def get_dataloaders(debug=False):
    """
    Creates DataLoaders for train, val, and test sets.
    Implements Square-Root Sampling for the training set.
    """
    # 1. Load Metadata CSVs
    if not os.path.exists(Config.TRAIN_CSV):
        raise FileNotFoundError(f"Train CSV not found at {Config.TRAIN_CSV}")

    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    if debug:
        logger.info("Debug mode: subsampling datasets.")
        train_df = train_df.sample(
            n=min(2000, len(train_df)), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(500, len(val_df)), random_state=Config.SEED
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(500, len(test_df)), random_state=Config.SEED
        ).reset_index(drop=True)

    # 2. Process Taxonomy
    taxonomy_map, meta_counts = process_taxonomy(load_cached_data=True)
    logger.info(
        f"Taxonomy: {meta_counts['num_families']} families, {meta_counts['num_genera']} genera."
    )

    # 3. Calculate Sampling Weights for Training
    # Square-Root Sampling: Weight ~ 1 / sqrt(Count)
    # This results in sampling probability proportional to sqrt(Count) relative to natural distribution
    logger.info("Calculating sampling weights...")
    class_counts = train_df["label"].value_counts().sort_index()
    # Map counts to the dataframe
    # Create a weight map: label -> weight
    # weight = 1 / sqrt(count)
    weights_map = (1.0 / np.sqrt(class_counts)).to_dict()

    # Assign weight to each sample
    sample_weights = train_df["label"].map(weights_map).values
    sample_weights = torch.from_numpy(sample_weights).double()

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # 4. Create Datasets
    train_dataset = PlantTaxonomyDataset(
        train_df,
        taxonomy_map=taxonomy_map,
        transforms=get_transforms("train"),
        phase="train",
    )

    val_dataset = PlantTaxonomyDataset(
        val_df, taxonomy_map=taxonomy_map, transforms=get_transforms("val"), phase="val"
    )

    test_dataset = PlantTaxonomyDataset(
        test_df, taxonomy_map=None, transforms=get_transforms("test"), phase="test"
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
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
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, meta_counts
