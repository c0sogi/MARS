import json
import os
import pandas as pd
from library.config import Config


def get_taxonomy_mappings(load_cached_data=True):
    """
    Extracts taxonomy information (Family, Genus) from the raw metadata JSON.
    Creates a mapping from species (category_id) to family_id and genus_id.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        pd.DataFrame: DataFrame containing 'category_id', 'family', 'genus',
                      'family_id', 'genus_id'.
    """
    cache_path = Config.TAXONOMY_MAPPING_PATH

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading taxonomy mappings from {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            # Validate cache schema
            required_cols = [
                "category_id",
                "family",
                "genus",
                "family_id",
                "genus_id",
                "species_id",
            ]
            if all(col in df.columns for col in required_cols):
                return df
            print(f"Cache missing columns. Found: {list(df.columns)}. Recomputing...")
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print("Computing taxonomy mappings from raw metadata...")

    if not os.path.exists(Config.TRAIN_META_JSON):
        raise FileNotFoundError(
            f"Raw metadata file not found at {Config.TRAIN_META_JSON}"
        )

    with open(Config.TRAIN_META_JSON, "r") as f:
        data = json.load(f)

    # The 'categories' list contains dictionaries with 'id', 'name', 'family', 'genus'
    categories = data.get("categories", [])

    if not categories:
        raise ValueError("No categories found in metadata JSON.")

    # Create DataFrame from list of dicts
    df = pd.DataFrame(categories)

    # Rename 'id' to 'category_id' for consistency with other dataframes
    if "id" in df.columns:
        df = df.rename(columns={"id": "category_id"})

    # Ensure necessary columns exist
    required_cols = ["category_id", "family", "genus"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(
            f"Metadata categories missing required columns. Found: {df.columns}"
        )

    # 3. Encode Family and Genus strings to integers
    # Sort uniques to ensure deterministic mapping
    unique_families = sorted(df["family"].astype(str).unique())
    unique_genera = sorted(df["genus"].astype(str).unique())
    unique_species = sorted(df["category_id"].unique())

    family_map = {name: i for i, name in enumerate(unique_families)}
    genus_map = {name: i for i, name in enumerate(unique_genera)}
    species_map = {cid: i for i, cid in enumerate(unique_species)}

    df["family_id"] = df["family"].map(family_map)
    df["genus_id"] = df["genus"].map(genus_map)
    df["species_id"] = df["category_id"].map(species_map)

    # Select and order columns
    result_df = df[
        ["category_id", "family", "genus", "family_id", "genus_id", "species_id"]
    ]

    # 4. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    result_df.to_parquet(cache_path, index=False)
    print(f"Saved taxonomy mappings to {cache_path}")

    # Print statistics
    print(f"Taxonomy Statistics:")
    print(f"  Species (Categories): {len(result_df)}")
    print(f"  Genera: {len(unique_genera)}")
    print(f"  Families: {len(unique_families)}")

    return result_df
