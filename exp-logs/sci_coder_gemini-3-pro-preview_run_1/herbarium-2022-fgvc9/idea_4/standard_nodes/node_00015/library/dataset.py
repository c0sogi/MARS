import os
import json
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.utils import Config, set_seed, ensure_dirs


def load_and_process_taxonomy(load_cached_data=True):
    """
    Loads taxonomy information from train_metadata.json and creates mappings
    for species, genus, and family.

    Handles remapping of non-contiguous category_ids to contiguous 0..N-1 indices
    for model training.

    Args:
        load_cached_data (bool): If True, attempts to load mappings from disk.

    Returns:
        dict: A dictionary containing all necessary mappings and counts.
    """
    ensure_dirs()
    cache_path = os.path.join(Config.WORKING_DIR, "taxonomy_mappings.json")

    if load_cached_data and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                mappings = json.load(f)
            # Convert keys that are strings back to integers where necessary
            # JSON keys are always strings, but our IDs are ints.
            # We need to be careful when using these maps.
            # Let's standardize: keys in JSON are strings, we will convert to int when looking up if needed.
            return mappings
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # Load raw metadata
    with open(Config.TRAIN_META_JSON, "r") as f:
        meta = json.load(f)

    categories = meta.get("categories", [])

    # 1. Species Mapping (category_id -> contiguous index)
    # Sort by category_id to ensure deterministic mapping
    sorted_cats = sorted(categories, key=lambda x: x["category_id"])

    species_to_idx = {}  # original category_id -> 0..15500
    idx_to_species = {}  # 0..15500 -> original category_id

    # 2. Genus and Family Mapping (name -> index)
    genus_names = sorted(list(set(c["genus"] for c in sorted_cats)))
    family_names = sorted(list(set(c["family"] for c in sorted_cats)))

    genus_to_idx = {name: i for i, name in enumerate(genus_names)}
    family_to_idx = {name: i for i, name in enumerate(family_names)}

    # 3. Hierarchical Mapping (species_idx -> genus_idx, species_idx -> family_idx)
    species_idx_to_genus_idx = {}
    species_idx_to_family_idx = {}

    for cat in sorted_cats:
        orig_id = cat["category_id"]
        genus = cat["genus"]
        family = cat["family"]

        # Assign contiguous ID
        if orig_id not in species_to_idx:
            new_id = len(species_to_idx)
            species_to_idx[orig_id] = new_id
            idx_to_species[new_id] = orig_id

            species_idx_to_genus_idx[new_id] = genus_to_idx[genus]
            species_idx_to_family_idx[new_id] = family_to_idx[family]

    mappings = {
        "species_to_idx": species_to_idx,  # Key: original category_id (int), Value: model label (int)
        "idx_to_species": idx_to_species,  # Key: model label (int), Value: original category_id (int)
        "genus_to_idx": genus_to_idx,
        "family_to_idx": family_to_idx,
        "species_idx_to_genus_idx": species_idx_to_genus_idx,  # Key: model species label, Value: genus label
        "species_idx_to_family_idx": species_idx_to_family_idx,  # Key: model species label, Value: family label
        "num_species": len(species_to_idx),
        "num_genera": len(genus_names),
        "num_families": len(family_names),
    }

    # Save to cache
    # Note: JSON keys must be strings. When saving, integer keys become strings.
    # When loading, we must handle this.
    with open(cache_path, "w") as f:
        json.dump(mappings, f)

    return mappings


class PlantDataset(Dataset):
    def __init__(
        self,
        df,
        transform=None,
        taxonomy_maps=None,
        is_test=False,
        input_dir=Config.INPUT_DIR,
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing file paths and labels.
            transform (albumentations.Compose): Augmentation pipeline.
            taxonomy_maps (dict): Mappings loaded from load_and_process_taxonomy.
            is_test (bool): If True, return only image and ID.
            input_dir (str): Root directory for images.
        """
        self.df = df
        self.transform = transform
        self.taxonomy_maps = taxonomy_maps
        self.is_test = is_test
        self.input_dir = input_dir

        # Pre-process mappings for fast lookup
        if not self.is_test and self.taxonomy_maps:
            self.species_to_idx = {
                int(k): v for k, v in self.taxonomy_maps["species_to_idx"].items()
            }
            self.species_idx_to_genus = {
                int(k): v
                for k, v in self.taxonomy_maps["species_idx_to_genus_idx"].items()
            }
            self.species_idx_to_family = {
                int(k): v
                for k, v in self.taxonomy_maps["species_idx_to_family_idx"].items()
            }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # The metadata file_path is relative to input_dir
        file_path = os.path.join(self.input_dir, row["file_path"])

        # Load Image
        image = cv2.imread(file_path)
        if image is None:
            # Fallback for missing images (should not happen based on validation)
            # Create a black image
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.is_test:
            return image, str(row["image_id"])

        # Training/Validation Targets
        orig_cat_id = int(row["category_id"])

        # Map to contiguous model label
        species_label = self.species_to_idx[orig_cat_id]
        genus_label = self.species_idx_to_genus[species_label]
        family_label = self.species_idx_to_family[species_label]

        return {
            "image": image,
            "species": torch.tensor(species_label, dtype=torch.long),
            "genus": torch.tensor(genus_label, dtype=torch.long),
            "family": torch.tensor(family_label, dtype=torch.long),
        }


def get_transforms(split="train"):
    """
    Returns Albumentations transforms for the specified split.

    Args:
        split (str): 'train', 'val', or 'test'.
    """
    if split == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(
                    size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                    scale=(0.8, 1.0),
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test / TTA
        return A.Compose(
            [
                A.Resize(size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached taxonomy mappings.

    Returns:
        tuple: (train_loader, val_loader, test_loader, taxonomy_maps)
    """
    # 1. Load Taxonomy Mappings
    maps = load_and_process_taxonomy(load_cached_data=load_cached_data)

    # 2. Load Metadata DataFrames
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # 3. Debug Mode: Subsample
    if Config.DEBUG:
        print(
            f"DEBUG MODE: Subsampling datasets to {Config.DEBUG_SAMPLE_SIZE} samples."
        )
        train_df = train_df.sample(
            n=min(len(train_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        # We keep test set full usually, or subsample if strictly debugging pipeline
        # For submission generation, we need full test set. Assuming DEBUG is for training loop check.

    # 4. Create Datasets
    train_dataset = PlantDataset(
        train_df, transform=get_transforms("train"), taxonomy_maps=maps, is_test=False
    )

    val_dataset = PlantDataset(
        val_df, transform=get_transforms("val"), taxonomy_maps=maps, is_test=False
    )

    test_dataset = PlantDataset(
        test_df, transform=get_transforms("test"), taxonomy_maps=maps, is_test=True
    )

    # 5. Create DataLoaders
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
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, maps
