import os
import json
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_f1(y_true, y_pred):
    """
    Calculates the Macro F1 score.

    Args:
        y_true: Array-like of ground truth labels.
        y_pred: Array-like of predicted labels.

    Returns:
        float: The Macro F1 score.
    """
    return f1_score(y_true, y_pred, average="macro")


def process_taxonomy(
    metadata_path, output_dir="./working/idea_8", load_cached_data=True
):
    """
    Parses metadata to create a mapping between category_id (species), genus, and family.
    Encodes these hierarchical levels into contiguous integers for training.

    Args:
        metadata_path (str): Path to the training metadata.json file.
        output_dir (str): Directory to save/load the cached mapping.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        pd.DataFrame: A DataFrame containing:
            - category_id: Original species ID.
            - family: Family name.
            - genus: Genus name.
            - species_label: Encoded contiguous integer for species (0 to N-1).
            - genus_label: Encoded contiguous integer for genus.
            - family_label: Encoded contiguous integer for family.
    """
    os.makedirs(output_dir, exist_ok=True)
    cache_path = os.path.join(output_dir, "taxonomy_mapping.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache from {cache_path}: {e}. Recomputing...")

    # 2. Compute from scratch
    with open(metadata_path, "r") as f:
        data = json.load(f)

    # Extract categories list: [{'id': 123, 'name': '...', 'family': '...', 'genus': '...'}, ...]
    categories = data["categories"]
    df = pd.DataFrame(categories)

    # Ensure columns exist
    required_cols = ["id", "family", "genus"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(
            f"Metadata categories missing required columns. Found: {df.columns}"
        )

    # Rename id to category_id for clarity
    df = df.rename(columns={"id": "category_id"})

    # Encode labels
    # We use LabelEncoder to transform strings/sparse IDs to contiguous integers [0, num_classes-1]

    # Species (category_id)
    species_encoder = LabelEncoder()
    df["species_label"] = species_encoder.fit_transform(df["category_id"])

    # Genus
    genus_encoder = LabelEncoder()
    df["genus_label"] = genus_encoder.fit_transform(df["genus"])

    # Family
    family_encoder = LabelEncoder()
    df["family_label"] = family_encoder.fit_transform(df["family"])

    # Select relevant columns
    df = df[
        [
            "category_id",
            "family",
            "genus",
            "species_label",
            "genus_label",
            "family_label",
        ]
    ]

    # 3. Save to cache
    df.to_parquet(cache_path, index=False)

    return df
