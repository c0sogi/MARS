import os
import json
import pandas as pd
from library.config import Config


class TaxonomyManager:
    """
    Manages the hierarchical taxonomy data (Species -> Family -> Order).
    Handles loading from raw JSON, generating integer mappings, and caching to Parquet.
    """

    def __init__(self):
        self.json_path = Config.TRAIN_METADATA_JSON
        self.cache_path = Config.TAXONOMY_MAP_PATH
        self.df_taxonomy = None
        self.species_to_family = {}
        self.species_to_order = {}
        self.num_families = 0
        self.num_orders = 0

    def load(self, load_cached_data=True):
        """
        Loads taxonomy data.

        Args:
            load_cached_data (bool): If True, attempts to load from the parquet cache.
                                     If False or cache missing, re-processes raw JSON.
        """
        # 1. Try loading from cache
        if load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading taxonomy mappings from cache: {self.cache_path}")
            self.df_taxonomy = pd.read_parquet(self.cache_path)
        else:
            # 2. Process from scratch
            print(f"Processing taxonomy from raw metadata: {self.json_path}")
            self._process_raw_metadata()

            # 3. Save to cache
            print(f"Saving taxonomy mappings to cache: {self.cache_path}")
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            self.df_taxonomy.to_parquet(self.cache_path, index=False)

        # 4. Populate internal structures
        self._populate_attributes()

    def _process_raw_metadata(self):
        """
        Parses the raw JSON and generates integer IDs for Family and Order.
        """
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Metadata file not found at {self.json_path}")

        with open(self.json_path, "r") as f:
            data = json.load(f)

        # 'categories' list contains dicts with: id, name, family, order
        df = pd.DataFrame(data["categories"])

        # Rename 'id' to 'category_id' for consistency with other files
        if "id" in df.columns:
            df.rename(columns={"id": "category_id"}, inplace=True)

        # Ensure category_id is int
        df["category_id"] = df["category_id"].astype(int)

        # Generate integer IDs for Family and Order
        # We use sort_values to ensure deterministic mapping if codes are assigned based on order
        # However, .cat.codes assigns codes based on alphabetical order of categories by default
        df["family"] = df["family"].astype("category")
        df["order"] = df["order"].astype("category")

        df["family_id"] = df["family"].cat.codes.astype(int)
        df["order_id"] = df["order"].cat.codes.astype(int)

        # Keep only necessary columns for the mapping
        self.df_taxonomy = df[["category_id", "family_id", "order_id"]].copy()

    def _populate_attributes(self):
        """
        Populates helper dictionaries and counts from the loaded DataFrame.
        """
        if self.df_taxonomy is None:
            raise ValueError("Taxonomy data not loaded. Call load() first.")

        # Create mappings: category_id -> family_id/order_id
        # Using dict(zip(...)) is efficient
        self.species_to_family = dict(
            zip(self.df_taxonomy["category_id"], self.df_taxonomy["family_id"])
        )
        self.species_to_order = dict(
            zip(self.df_taxonomy["category_id"], self.df_taxonomy["order_id"])
        )

        # Calculate unique counts
        self.num_families = self.df_taxonomy["family_id"].nunique()
        self.num_orders = self.df_taxonomy["order_id"].nunique()

        print(
            f"Taxonomy loaded: {len(self.species_to_family)} species, "
            f"{self.num_families} families, {self.num_orders} orders."
        )

    def get_mappings(self):
        """
        Returns the mappings for dataset construction.

        Returns:
            tuple: (species_to_family_dict, species_to_order_dict)
        """
        return self.species_to_family, self.species_to_order

    def get_counts(self):
        """
        Returns the number of unique classes for auxiliary heads.

        Returns:
            tuple: (num_families, num_orders)
        """
        return self.num_families, self.num_orders
