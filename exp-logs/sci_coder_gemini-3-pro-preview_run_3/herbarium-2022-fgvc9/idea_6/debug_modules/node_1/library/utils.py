import os
import json
import random
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_hierarchy_mappings(load_cached_data=True):
    """
    Parses train_metadata.json to create mappings from species (category_id)
    to genus and family. Caches the result in a parquet file.

    Args:
        load_cached_data (bool): If True, attempts to load from cached parquet file.

    Returns:
        tuple: (species_to_genus, species_to_family, num_genera, num_families)
            - species_to_genus (dict): Map category_id -> genus_id
            - species_to_family (dict): Map category_id -> family_id
            - num_genera (int): Total number of unique genera
            - num_families (int): Total number of unique families
    """
    mapping_path = Config.HIERARCHY_MAPPING_PATH

    # Ensure working directory exists
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)

    df_mappings = None

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(mapping_path):
        try:
            df_mappings = pd.read_parquet(mapping_path)
        except Exception as e:
            print(f"Failed to load cached hierarchy mappings: {e}. Recomputing...")
            df_mappings = None

    # 2. Compute if not loaded
    if df_mappings is None:
        json_path = Config.HIERARCHY_JSON_PATH
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Hierarchy JSON not found at {json_path}")

        with open(json_path, "r") as f:
            data = json.load(f)

        # Extract categories: list of dicts with 'id', 'family', 'genus', etc.
        if "categories" not in data:
            raise ValueError("Invalid JSON format: 'categories' key missing.")

        cats = pd.DataFrame(data["categories"])

        # Ensure we have the required columns
        required_cols = {"category_id", "family", "genus"}
        if not required_cols.issubset(cats.columns):
            raise ValueError(
                f"JSON categories missing required columns. Found: {cats.columns}"
            )

        # Create integer IDs for family and genus
        # Sort by name to ensure deterministic ID assignment
        families = sorted(cats["family"].unique())
        genera = sorted(cats["genus"].unique())

        family_map = {name: i for i, name in enumerate(families)}
        genus_map = {name: i for i, name in enumerate(genera)}

        cats["family_id"] = cats["family"].map(family_map)
        cats["genus_id"] = cats["genus"].map(genus_map)

        # Select relevant columns for the mapping dataframe
        df_mappings = cats[["category_id", "genus_id", "family_id"]].copy()

        # Save to cache
        df_mappings.to_parquet(mapping_path, index=False)

    # 3. Convert to output format
    species_to_genus = dict(zip(df_mappings["category_id"], df_mappings["genus_id"]))
    species_to_family = dict(zip(df_mappings["category_id"], df_mappings["family_id"]))

    num_genera = df_mappings["genus_id"].max() + 1
    num_families = df_mappings["family_id"].max() + 1

    return species_to_genus, species_to_family, int(num_genera), int(num_families)


def get_transforms(data, image_size):
    """
    Returns Albumentations transforms for training or validation/inference.

    Args:
        data (str): 'train' or 'valid' (also used for test/inference).
        image_size (int): Target resolution (e.g., 224 or 320).

    Returns:
        A.Compose: Composed albumentations transforms.
    """
    # ImageNet normalization statistics
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if data == "train":
        return A.Compose(
            [
                # Strong Data Augmentation as per strategy
                A.RandomResizedCrop(
                    height=image_size, width=image_size, scale=(0.08, 1.0), p=1.0
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    elif data == "valid" or data == "test":
        return A.Compose(
            [
                # Deterministic resizing for validation/inference
                # Using simple Resize to preserve all content, as cropping might remove key features
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(
            f"Unknown data mode: {data}. Expected 'train', 'valid', or 'test'."
        )
