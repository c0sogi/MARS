import json
import os
import pandas as pd
from library.config import Config


class TaxonomyManager:
    """
    Manages the hierarchical taxonomy mappings (Species -> Family -> Order).
    Parses the raw metadata to create integer mappings for auxiliary tasks.
    """

    def __init__(self):
        self.metadata_path = Config.TRAIN_METADATA_JSON
        self.cache_path = Config.TAXONOMY_MAP_PATH

    def build_mappings(self, load_cached_data=True):
        """
        Builds or loads the taxonomy mappings.

        Args:
            load_cached_data (bool): If True, attempts to load from parquet cache.

        Returns:
            tuple: (DataFrame mapping, int num_families, int num_orders)
                   The DataFrame contains columns ['category_id', 'family_id', 'order_id'].
        """
        # 1. Try to load from cache
        if load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading taxonomy mappings from cache: {self.cache_path}")
            try:
                df_mappings = pd.read_parquet(self.cache_path)
                num_families = int(df_mappings["family_id"].max()) + 1
                num_orders = int(df_mappings["order_id"].max()) + 1
                print(f"Loaded mappings for {len(df_mappings)} species.")
                return df_mappings, num_families, num_orders
            except Exception as e:
                print(f"Failed to load cache: {e}. Rebuilding...")

        # 2. Build from scratch
        print(f"Building taxonomy mappings from raw metadata: {self.metadata_path}")

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")

        with open(self.metadata_path, "r") as f:
            data = json.load(f)

        categories = data.get("categories", [])
        if not categories:
            raise ValueError("No categories found in metadata JSON.")

        # Extract unique families and orders to assign IDs
        # We sort them to ensure deterministic ID assignment
        all_families = sorted(list(set(cat["family"] for cat in categories)))
        all_orders = sorted(list(set(cat["order"] for cat in categories)))

        family_to_id = {name: i for i, name in enumerate(all_families)}
        order_to_id = {name: i for i, name in enumerate(all_orders)}

        # Build the mapping table
        mapping_rows = []
        for cat in categories:
            # Ensure category_id is int as per dataset specs
            cat_id = int(cat["id"])
            fam_name = cat["family"]
            ord_name = cat["order"]

            mapping_rows.append(
                {
                    "category_id": cat_id,
                    "family_id": family_to_id[fam_name],
                    "order_id": order_to_id[ord_name],
                }
            )

        df_mappings = pd.DataFrame(mapping_rows)

        # 3. Save to cache
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        df_mappings.to_parquet(self.cache_path, index=False)
        print(f"Saved taxonomy mappings to {self.cache_path}")

        num_families = len(all_families)
        num_orders = len(all_orders)

        print(
            f"Taxonomy built: {len(df_mappings)} species, {num_families} families, {num_orders} orders."
        )

        return df_mappings, num_families, num_orders
