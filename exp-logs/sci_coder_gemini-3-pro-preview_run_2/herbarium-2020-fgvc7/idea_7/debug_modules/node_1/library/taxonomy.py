import os
import json
import pandas as pd
import numpy as np
import torch
from library.config import Config


class TaxonomyMapper:
    """
    Handles the hierarchical taxonomy (Species -> Genus -> Family).
    Parses metadata to create mappings between raw category IDs and contiguous model indices.
    """

    def __init__(self, config: Config):
        self.config = config
        self.mapping_df = None
        self.species_to_idx = {}
        self.idx_to_species = {}
        self.species_to_genus_map = None  # Tensor mapping species_idx -> genus_idx
        self.species_to_family_map = None  # Tensor mapping species_idx -> family_idx
        self.num_classes = 0
        self.num_genera = 0
        self.num_families = 0

    def load_or_build(self, load_cached_data: bool = True):
        """
        Loads the taxonomy mapping from cache or builds it from source.

        Args:
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            self
        """
        cache_path = self.config.TAXONOMY_MAP_PATH

        # Logic: IF load_cached_data is True: Try to load.
        # IF loading fails OR load_cached_data is False: Build and save.
        loaded = False
        if load_cached_data and os.path.exists(cache_path):
            try:
                print(f"Loading taxonomy mapping from {cache_path}")
                self.mapping_df = pd.read_parquet(cache_path)
                loaded = True
            except Exception as e:
                print(f"Failed to load cache: {e}")
                loaded = False

        if not loaded:
            print("Building taxonomy mapping from source...")
            self.mapping_df = self._build_mapping()

            # Save to cache
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            self.mapping_df.to_parquet(cache_path, index=False)
            print(f"Saved taxonomy mapping to {cache_path}")

        self._process_mappings()
        return self

    def _build_mapping(self):
        """
        Parses the training metadata JSON to build the taxonomy DataFrame.
        """
        if not os.path.exists(self.config.TRAIN_METADATA_JSON):
            raise FileNotFoundError(
                f"Metadata file not found: {self.config.TRAIN_METADATA_JSON}"
            )

        with open(self.config.TRAIN_METADATA_JSON, "r") as f:
            data = json.load(f)

        # Extract categories list: [{'id': int, 'name': str, 'family': str, 'genus': str}, ...]
        categories = data.get("categories", [])
        if not categories:
            raise ValueError("No categories found in metadata JSON.")

        df = pd.DataFrame(categories)

        # Ensure deterministic order by sorting by category ID
        df = df.sort_values("id").reset_index(drop=True)

        # 1. Create Species Index (0 to N-1)
        df["species_idx"] = range(len(df))

        # 2. Create Genus Index
        # Sort unique genera alphabetically for deterministic mapping
        unique_genera = sorted(df["genus"].unique())
        genus_map = {g: i for i, g in enumerate(unique_genera)}
        df["genus_idx"] = df["genus"].map(genus_map)

        # 3. Create Family Index
        unique_families = sorted(df["family"].unique())
        family_map = {f: i for i, f in enumerate(unique_families)}
        df["family_idx"] = df["family"].map(family_map)

        # Rename 'id' to 'category_id' for clarity and select relevant columns
        df = df.rename(columns={"id": "category_id"})
        out_df = df[
            ["species_idx", "category_id", "genus_idx", "family_idx", "genus", "family"]
        ]

        return out_df

    def _process_mappings(self):
        """
        Populates in-memory lookups and tensors from the DataFrame.
        """
        df = self.mapping_df

        self.num_classes = len(df)
        self.num_genera = int(df["genus_idx"].max()) + 1
        self.num_families = int(df["family_idx"].max()) + 1

        # Create Dictionaries
        # category_id -> species_idx
        self.species_to_idx = dict(zip(df["category_id"], df["species_idx"]))
        # species_idx -> category_id
        self.idx_to_species = dict(zip(df["species_idx"], df["category_id"]))

        # Create Tensors for fast lookup during training
        # We must ensure the tensors are ordered by species_idx (0, 1, 2, ...)
        # The DataFrame is sorted by species_idx implicitly if built from source,
        # but we explicitly sort here to be safe.
        df_sorted = df.sort_values("species_idx")

        # Tensor: index is species_idx, value is genus_idx
        self.species_to_genus_map = torch.tensor(
            df_sorted["genus_idx"].values, dtype=torch.long
        )

        # Tensor: index is species_idx, value is family_idx
        self.species_to_family_map = torch.tensor(
            df_sorted["family_idx"].values, dtype=torch.long
        )

        print(
            f"Taxonomy Processed: {self.num_classes} Species, {self.num_genera} Genera, {self.num_families} Families"
        )
