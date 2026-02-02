import os
import json
import pandas as pd
import numpy as np
from library.utils import set_seed


def get_taxonomy_mappings(
    metadata_path="./input/nybg2020/train/metadata.json",
    load_cached_data=True,
    **kwargs,
):
    """
    Parses the metadata JSON to extract hierarchical relationships:
    Species (category_id) -> Genus -> Family.

    Args:
        metadata_path (str): Path to the training metadata JSON file.
        load_cached_data (bool): Whether to attempt loading from cache.
        **kwargs: Additional arguments for flexibility.

    Returns:
        tuple: (species_to_genus, species_to_family, num_genera, num_families)
            - species_to_genus (dict): Map of category_id -> genus_id
            - species_to_family (dict): Map of category_id -> family_id
            - num_genera (int): Total number of unique genera
            - num_families (int): Total number of unique families
    """
    # Ensure reproducibility
    set_seed(42)

    # Define cache directory and file
    cache_dir = "./working/idea_3"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "taxonomy_mapping.parquet")

    df_mapping = None

    # Attempt to load from cache
    if load_cached_data and os.path.exists(cache_file):
        try:
            print(f"Loading taxonomy mappings from cache: {cache_file}")
            df_mapping = pd.read_parquet(cache_file)
        except Exception as e:
            print(f"Failed to load cache ({e}). Proceeding to recompute.")

    # Compute if not loaded
    if df_mapping is None:
        print(f"Processing metadata from {metadata_path}...")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        with open(metadata_path, "r") as f:
            data = json.load(f)

        # 'categories' is a list of dicts: {"id": int, "name": str, "genus": str, "family": str}
        categories = data.get("categories", [])

        if not categories:
            raise ValueError("No 'categories' field found in metadata.")

        # Extract unique Genera and Families
        # Sort to ensure deterministic ID assignment across runs
        unique_genera = sorted(list(set(cat["genus"] for cat in categories)))
        unique_families = sorted(list(set(cat["family"] for cat in categories)))

        # Create Name -> ID maps
        genus_name_to_id = {name: idx for idx, name in enumerate(unique_genera)}
        family_name_to_id = {name: idx for idx, name in enumerate(unique_families)}

        # Build the mapping table
        mapping_rows = []
        for cat in categories:
            cat_id = cat["id"]
            genus_id = genus_name_to_id[cat["genus"]]
            family_id = family_name_to_id[cat["family"]]

            mapping_rows.append(
                {"category_id": cat_id, "genus_id": genus_id, "family_id": family_id}
            )

        df_mapping = pd.DataFrame(mapping_rows)

        # Save to cache
        print(f"Saving taxonomy mappings to cache: {cache_file}")
        df_mapping.to_parquet(cache_file, index=False)

    # Convert DataFrame to required dictionaries and counts
    # We use dictionaries to handle potential non-contiguous category_ids safely
    species_to_genus = dict(zip(df_mapping["category_id"], df_mapping["genus_id"]))
    species_to_family = dict(zip(df_mapping["category_id"], df_mapping["family_id"]))

    num_genera = df_mapping["genus_id"].max() + 1
    num_families = df_mapping["family_id"].max() + 1

    print(
        f"Taxonomy loaded: {len(species_to_genus)} Species, {num_genera} Genera, {num_families} Families."
    )

    return species_to_genus, species_to_family, num_genera, num_families
