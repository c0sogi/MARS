import os
import json
import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.utils import get_taxonomy_mappings

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
CACHE_DIR = "./working/idea_2/"


def get_species_mapping(train_csv_path="./metadata/train.csv", load_cached_data=True):
    """
    Creates a mapping from category_id to a contiguous 0..N-1 index.
    Ensures consistency with utils.compute_loss_weights by sorting category_ids.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "species_mapping.json")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)
            # Convert keys back to int (JSON stores keys as strings)
            cat_to_label = {int(k): v for k, v in cached["cat_to_label"].items()}
            num_species = cached["num_species"]
            return cat_to_label, num_species
        except Exception:
            pass  # Fallback to computation

    # 2. Compute mapping
    df = pd.read_csv(train_csv_path)
    unique_cats = sorted(df["category_id"].unique())

    cat_to_label = {int(cat_id): idx for idx, cat_id in enumerate(unique_cats)}
    num_species = len(unique_cats)

    # 3. Save to cache
    to_save = {"cat_to_label": cat_to_label, "num_species": num_species}
    with open(cache_path, "w") as f:
        json.dump(to_save, f)

    return cat_to_label, num_species


def get_transforms(mode="train", image_size=256):
    """
    Returns Albumentations transforms for the specified mode.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.RandomResizedCrop(
                    height=image_size, width=image_size, scale=(0.8, 1.0), p=1.0
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


class HierarchicalPlantDataset(Dataset):
    def __init__(
        self,
        df,
        transform=None,
        cat_to_fam=None,
        cat_to_gen=None,
        cat_to_label=None,
        mode="train",
    ):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            transform (albumentations.Compose): Transforms to apply.
            cat_to_fam (dict): Mapping from category_id to family_id.
            cat_to_gen (dict): Mapping from category_id to genus_id.
            cat_to_label (dict): Mapping from category_id to species_label_idx.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transform = transform
        self.cat_to_fam = cat_to_fam
        self.cat_to_gen = cat_to_gen
        self.cat_to_label = cat_to_label
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])

        # Construct full path
        file_path = os.path.join(INPUT_DIR, row["file_path"])

        # Load image
        image = cv2.imread(file_path)
        if image is None:
            # In a strict pipeline, we might raise an error.
            # For robustness, we could return a zero tensor, but here we assume data integrity based on metadata check.
            raise FileNotFoundError(f"Image not found at {file_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        targets = {}

        if self.mode in ["train", "val"]:
            # Retrieve raw category_id
            cat_id = int(row["category_id"])

            # Map to contiguous indices
            species_label = self.cat_to_label[cat_id]
            family_label = self.cat_to_fam[cat_id]
            genus_label = self.cat_to_gen[cat_id]

            targets = {
                "species": torch.tensor(species_label, dtype=torch.long),
                "genus": torch.tensor(genus_label, dtype=torch.long),
                "family": torch.tensor(family_label, dtype=torch.long),
            }

        # Return image, targets dict, and image_id (useful for submission)
        return image, targets, image_id


def get_dataloaders(
    batch_size=32, num_workers=4, image_size=256, load_cached_data=True
):
    """
    Initializes datasets and dataloaders for Train, Val, and Test splits.

    Returns:
        train_loader, val_loader, test_loader, metadata_dict
    """
    # 1. Load Taxonomy Mappings (Family/Genus)
    cat_to_fam, cat_to_gen, num_fam, num_gen = get_taxonomy_mappings(
        metadata_json_path=os.path.join(INPUT_DIR, "train_metadata.json"),
        load_cached_data=load_cached_data,
    )

    # 2. Load/Compute Species Mapping (0..N-1)
    cat_to_label, num_species = get_species_mapping(
        train_csv_path=os.path.join(METADATA_DIR, "train.csv"),
        load_cached_data=load_cached_data,
    )

    # 3. Load Metadata DataFrames
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # 4. Define Transforms
    train_tf = get_transforms(mode="train", image_size=image_size)
    val_tf = get_transforms(mode="val", image_size=image_size)

    # 5. Instantiate Datasets
    train_ds = HierarchicalPlantDataset(
        train_df,
        transform=train_tf,
        cat_to_fam=cat_to_fam,
        cat_to_gen=cat_to_gen,
        cat_to_label=cat_to_label,
        mode="train",
    )

    val_ds = HierarchicalPlantDataset(
        val_df,
        transform=val_tf,
        cat_to_fam=cat_to_fam,
        cat_to_gen=cat_to_gen,
        cat_to_label=cat_to_label,
        mode="val",
    )

    test_ds = HierarchicalPlantDataset(
        test_df, transform=val_tf, mode="test"  # Mappings not needed for test
    )

    # 6. Instantiate DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
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
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    metadata = {
        "num_families": num_fam,
        "num_genera": num_gen,
        "num_species": num_species,
        "cat_to_label": cat_to_label,  # Useful for reverse mapping later
    }

    return train_loader, val_loader, test_loader, metadata
