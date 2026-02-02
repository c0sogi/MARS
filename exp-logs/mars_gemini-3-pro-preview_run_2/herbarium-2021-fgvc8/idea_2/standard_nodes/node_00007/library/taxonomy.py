import os
import json
import pandas as pd
from library.config import Config


class TaxonomyManager:
    """
    Manages the hierarchical taxonomy of the dataset (Species -> Family -> Order).
    Responsible for loading metadata, encoding auxiliary labels, and providing
    mappings for the dataset and model.
    """

    def __init__(self):
        self.map_df = None
        self.num_families = 0
        self.num_orders = 0
        self._mapping_dict = None

    def process_taxonomy(self, load_cached_data=True):
        """
        Loads or generates the taxonomy mappings.

        Args:
            load_cached_data (bool): If True, attempts to load from parquet cache.
                                     If False or cache missing, regenerates from raw JSON.

        Returns:
            pd.DataFrame: DataFrame containing category_id, family_id, and order_id.
        """
        # Ensure working directory exists
        os.makedirs(os.path.dirname(Config.TAXONOMY_MAP_PATH), exist_ok=True)

        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(Config.TAXONOMY_MAP_PATH):
            try:
                self.map_df = pd.read_parquet(Config.TAXONOMY_MAP_PATH)
                # Recalculate counts from loaded data
                self.num_families = self.map_df["family_id"].max() + 1
                self.num_orders = self.map_df["order_id"].max() + 1
                return self.map_df
            except Exception as e:
                print(f"Failed to load cached taxonomy: {e}. Regenerating...")

        # 2. Generate from Scratch
        print(f"Loading raw metadata from {Config.RAW_TRAIN_METADATA}...")
        with open(Config.RAW_TRAIN_METADATA, "r") as f:
            data = json.load(f)

        # Extract categories list: [{'id': 0, 'family': '...', 'order': '...'}, ...]
        categories = data["categories"]
        df = pd.DataFrame(categories)

        # We need 'id' (which is category_id), 'family', and 'order'
        # Rename 'id' to 'category_id' for consistency with other files
        df = df.rename(columns={"id": "category_id"})

        # Select only relevant columns
        df = df[["category_id", "family", "order"]]

        # Create deterministic integer encodings
        # Sort unique values to ensure ID 0 is always the same class across runs
        unique_families = sorted(df["family"].unique())
        unique_orders = sorted(df["order"].unique())

        family_map = {name: i for i, name in enumerate(unique_families)}
        order_map = {name: i for i, name in enumerate(unique_orders)}

        df["family_id"] = df["family"].map(family_map)
        df["order_id"] = df["order"].map(order_map)

        # Store results
        self.map_df = df
        self.num_families = len(unique_families)
        self.num_orders = len(unique_orders)

        # 3. Save to Cache
        print(f"Saving taxonomy mappings to {Config.TAXONOMY_MAP_PATH}...")
        self.map_df.to_parquet(Config.TAXONOMY_MAP_PATH, index=False)

        return self.map_df

    def get_mappings(self):
        """
        Returns a dictionary for fast lookup of auxiliary labels.

        Returns:
            dict: {category_id: {'family_id': int, 'order_id': int}}
        """
        if self.map_df is None:
            self.process_taxonomy(load_cached_data=True)

        if self._mapping_dict is None:
            # Create a dictionary indexed by category_id for O(1) access
            # orient='index' results in {index: {col: val, ...}}
            # We set index to category_id first
            self._mapping_dict = self.map_df.set_index("category_id")[
                ["family_id", "order_id"]
            ].to_dict(orient="index")

        return self._mapping_dict

    def get_counts(self):
        """
        Returns the number of unique classes for auxiliary tasks.

        Returns:
            tuple: (num_families, num_orders)
        """
        if self.map_df is None:
            self.process_taxonomy(load_cached_data=True)

        return self.num_families, self.num_orders
