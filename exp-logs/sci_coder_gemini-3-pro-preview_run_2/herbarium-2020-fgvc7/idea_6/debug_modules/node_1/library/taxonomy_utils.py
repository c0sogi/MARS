import os
import json
import pandas as pd
import numpy as np
import random


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def build_taxonomy_mapping(
    metadata_path="./input/nybg2020/train/metadata.json",
    cache_dir="./working/idea_6",
    load_cached_data=True,
):
    """
    Extracts taxonomy information (genus, family) for each species category from the metadata.

    This function reads the competition metadata, extracts the 'categories' list,
    and creates a mapping from the species 'id' to encoded integers for 'genus' and 'family'.
    The result is cached to disk to speed up future runs.

    Args:
        metadata_path (str): Path to the raw metadata json containing category info.
        cache_dir (str): Directory to save/load cached mappings.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: DataFrame containing taxonomy info.
                      Index: 'category_id' (int) - The species ID.
                      Columns:
                        - 'genus_id' (int): Encoded label for genus (0..N_genus-1).
                        - 'family_id' (int): Encoded label for family (0..N_family-1).
                        - 'genus' (str): Original genus name.
                        - 'family' (str): Original family name.
    """
    set_seed(42)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "taxonomy_mapping.parquet")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading taxonomy mapping from {cache_path}...")
            df_mapping = pd.read_parquet(cache_path)
            return df_mapping
        except Exception as e:
            print(f"Failed to load cache: {e}. Rebuilding...")

    # 2. Process from scratch
    print(f"Building taxonomy mapping from {metadata_path}...")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    with open(metadata_path, "r") as f:
        data = json.load(f)

    categories = data.get("categories", [])
    if not categories:
        raise ValueError("No 'categories' field found in metadata.")

    # Create DataFrame from categories list
    # Expected keys in dict: 'id', 'name', 'genus', 'family'
    df = pd.DataFrame(categories)

    # Rename id to category_id for clarity (this is the species ID)
    if "id" in df.columns:
        df = df.rename(columns={"id": "category_id"})

    # Ensure we have the required columns
    required_cols = ["category_id", "genus", "family"]
    if not all(col in df.columns for col in required_cols):
        missing = [c for c in required_cols if c not in df.columns]
        raise ValueError(f"Metadata categories missing columns: {missing}")

    # Encode Genus and Family to integers
    # We sort by name to ensure deterministic encoding (A=0, B=1, ...)
    genus_names = sorted(df["genus"].unique())
    family_names = sorted(df["family"].unique())

    genus_map = {name: i for i, name in enumerate(genus_names)}
    family_map = {name: i for i, name in enumerate(family_names)}

    df["genus_id"] = df["genus"].map(genus_map)
    df["family_id"] = df["family"].map(family_map)

    # Select relevant columns and set index
    df_mapping = df[["category_id", "genus_id", "family_id", "genus", "family"]].copy()
    df_mapping = df_mapping.set_index("category_id").sort_index()

    # 3. Save to cache
    try:
        df_mapping.to_parquet(cache_path)
        print(f"Saved taxonomy mapping to {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")

    return df_mapping


def get_taxonomy_stats(df_mapping):
    """
    Returns statistics about the taxonomy hierarchy.

    Args:
        df_mapping (pd.DataFrame): The dataframe returned by build_taxonomy_mapping.

    Returns:
        dict: Dictionary with counts of species, genera, and families.
    """
    return {
        "num_species": len(df_mapping),
        "num_genera": df_mapping["genus_id"].nunique(),
        "num_families": df_mapping["family_id"].nunique(),
    }
