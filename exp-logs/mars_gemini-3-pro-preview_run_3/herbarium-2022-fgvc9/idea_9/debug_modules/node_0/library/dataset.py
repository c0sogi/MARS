import os
import json
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.utils import seed_everything


def get_hierarchy_mappings(
    metadata_json_path, load_cached_data=True, cache_dir="./working/idea_9"
):
    """
    Parses train_metadata.json to create mappings from category_id (species)
    to genus_id and family_id. Handles caching to parquet.

    Args:
        metadata_json_path (str): Path to train_metadata.json.
        load_cached_data (bool): Whether to try loading from cache.
        cache_dir (str): Directory to store the cached parquet file.

    Returns:
        tuple: (df_mappings, num_families, num_genera, num_species)
            df_mappings (pd.DataFrame): Columns [category_id, genus_id, family_id]
            num_families (int): Total number of unique families (max_id + 1)
            num_genera (int): Total number of unique genera (max_id + 1)
            num_species (int): Total number of unique species (max_id + 1)
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "hierarchy_mappings.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading hierarchy mappings from {cache_path}")
        df_mappings = pd.read_parquet(cache_path)

        # Infer counts from the dataframe
        num_families = df_mappings["family_id"].max() + 1
        num_genera = df_mappings["genus_id"].max() + 1
        num_species = df_mappings["category_id"].max() + 1

        return df_mappings, num_families, num_genera, num_species

    print(f"Computing hierarchy mappings from {metadata_json_path}")
    with open(metadata_json_path, "r") as f:
        data = json.load(f)

    categories = data["categories"]
    df = pd.DataFrame(categories)
    # Expected columns in df: 'id', 'name', 'family', 'genus'

    # 1. Encode Families
    unique_families = sorted(df["family"].unique())
    family_map = {name: i for i, name in enumerate(unique_families)}
    df["family_id"] = df["family"].map(family_map)

    # 2. Encode Genera
    unique_genera = sorted(df["genus"].unique())
    genus_map = {name: i for i, name in enumerate(unique_genera)}
    df["genus_id"] = df["genus"].map(genus_map)

    # 3. Prepare Mapping DataFrame
    # 'id' in categories corresponds to 'category_id' in train.csv
    df_mappings = df[["id", "family_id", "genus_id"]].rename(
        columns={"id": "category_id"}
    )

    # Save to cache
    df_mappings.to_parquet(cache_path)

    num_families = len(unique_families)
    num_genera = len(unique_genera)
    # We use max() + 1 for species count to handle potential gaps in IDs,
    # ensuring the embedding/classification layer covers the largest ID.
    num_species = df["id"].max() + 1

    return df_mappings, num_families, num_genera, num_species


def get_transforms(data="train", image_size=224):
    """
    Returns Albumentations transforms for training or validation/testing.

    Args:
        data (str): 'train', 'valid', or 'test'.
        image_size (int): Target image resolution (e.g., 224, 288).
    """
    if data == "train":
        return A.Compose(
            [
                # Strong augmentation for regularization
                A.RandomResizedCrop(image_size, image_size, scale=(0.6, 1.0)),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.3
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    elif data == "valid" or data == "test":
        return A.Compose(
            [
                # Deterministic resizing for evaluation
                A.Resize(image_size, image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data mode: {data}")


class PlantDataset(Dataset):
    def __init__(self, df, root_dir, hierarchy_df=None, transform=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            root_dir (str): Root directory for images (e.g., './input').
            hierarchy_df (pd.DataFrame, optional): Mapping of category_id to genus/family.
            transform (albumentations.Compose): Transforms to apply.
            is_test (bool): If True, returns (image, image_id). If False, returns (image, species, genus, family).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test

        # If training/validation, merge hierarchy info
        if not is_test and hierarchy_df is not None:
            # Ensure we don't have duplicate columns before merge
            cols_to_use = hierarchy_df.columns.difference(df.columns).tolist()
            if "category_id" not in cols_to_use:
                cols_to_use.append("category_id")

            self.data = df.merge(
                hierarchy_df[cols_to_use], on="category_id", how="left"
            )
        else:
            self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Construct full file path
        file_path = os.path.join(self.root_dir, row["file_path"])

        # Load image
        image = cv2.imread(file_path)
        if image is None:
            # In a real scenario, we might handle this differently.
            # Given the dataset analysis showed no missing files, we raise an error.
            raise FileNotFoundError(f"Image not found at {file_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.is_test:
            # Return image and image_id for submission generation
            return image, row["image_id"]
        else:
            # Return image and hierarchical labels
            species_label = int(row["category_id"])
            genus_label = int(row["genus_id"])
            family_label = int(row["family_id"])

            return (
                image,
                torch.tensor(species_label, dtype=torch.long),
                torch.tensor(genus_label, dtype=torch.long),
                torch.tensor(family_label, dtype=torch.long),
            )
