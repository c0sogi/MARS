import os
import json
import pandas as pd
import numpy as np


def get_species_to_genus_mapping(
    json_path: str = "./input/nybg2020/train/metadata.json",
    cache_dir: str = "./working/idea_4/",
    load_cached_data: bool = True,
):
    """
    Parses the metadata JSON to map species (category_id) to genus IDs.

    Args:
        json_path (str): Path to the training metadata JSON file.
        cache_dir (str): Directory to store/load the cached mapping.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (species_to_genus_dict, num_genera)
            - species_to_genus_dict: Dict mapping category_id (int) to genus_id (int).
            - num_genera: Total number of unique genera found.
    """
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "species_to_genus.parquet")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_file):
        try:
            # Read parquet file
            # Expecting index to be 'category_id' and a column 'genus_id'
            df = pd.read_parquet(cache_file)

            # Convert back to dictionary: {category_id: genus_id}
            mapping = df["genus_id"].to_dict()

            # Calculate num_genera based on the max ID found (assuming 0-indexed contiguous or similar)
            # However, to be safe and consistent with creation logic, we should infer or store it.
            # Since IDs are 0 to N-1, max() + 1 gives the count.
            if not df.empty:
                num_genera = int(df["genus_id"].max()) + 1
            else:
                num_genera = 0

            print(f"Loaded species-to-genus mapping from cache: {cache_file}")
            return mapping, num_genera

        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing from source...")

    # 2. Compute from scratch
    print(f"Computing species-to-genus mapping from {json_path}...")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Metadata file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    categories = data.get("categories", [])

    if not categories:
        print("Warning: No categories found in metadata.")
        return {}, 0

    # Extract unique genera and assign IDs deterministically
    # Sorting ensures that the ID assignment is reproducible across runs
    unique_genera = sorted(
        list(set(cat["genus"] for cat in categories if "genus" in cat))
    )
    genus_to_id = {genus: idx for idx, genus in enumerate(unique_genera)}

    # Map species (category_id) to genus_id
    species_to_genus = {}
    for cat in categories:
        # Ensure required fields exist
        if "id" in cat and "genus" in cat:
            species_id = cat["id"]
            genus_name = cat["genus"]
            species_to_genus[species_id] = genus_to_id[genus_name]

    # 3. Save to cache
    # Create DataFrame for Parquet storage
    # Index: category_id (species), Column: genus_id
    df_out = pd.DataFrame.from_dict(
        species_to_genus, orient="index", columns=["genus_id"]
    )
    df_out.index.name = "category_id"

    try:
        df_out.to_parquet(cache_file)
        print(f"Saved species-to-genus mapping to {cache_file}")
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_file}: {e}")

    num_genera = len(unique_genera)

    return species_to_genus, num_genera
