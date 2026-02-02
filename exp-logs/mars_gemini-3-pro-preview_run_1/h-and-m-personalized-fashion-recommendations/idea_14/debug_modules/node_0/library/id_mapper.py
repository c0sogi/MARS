import pandas as pd
import numpy as np
import os
from library.config import Config
from library.data_utils import load_processed_data


class IdMapper:
    """
    Manages the mapping between raw string/int IDs (customer_id, article_id)
    and contiguous integer indices required for sparse matrix operations.
    """

    def __init__(self):
        self.user_map = None
        self.item_map = None

        # Fast lookup structures
        self.user_to_idx = {}
        self.idx_to_user = np.array([])
        self.item_to_idx = {}
        self.idx_to_item = np.array([])

    def fit(self, load_cached_data=True):
        """
        Initializes the mappings. Tries to load from cache first to avoid
        loading the full transaction dataset if possible.

        Args:
            load_cached_data (bool): If True, attempts to load pre-computed maps from disk.

        Returns:
            self
        """
        # Check if specific map cache files exist
        # This allows us to skip loading the heavy transactions file if we only need maps
        cache_user_exists = os.path.exists(Config.CACHE_USER_MAP)
        cache_item_exists = os.path.exists(Config.CACHE_ITEM_MAP)

        if load_cached_data and cache_user_exists and cache_item_exists:
            self.user_map = pd.read_parquet(Config.CACHE_USER_MAP)
            self.item_map = pd.read_parquet(Config.CACHE_ITEM_MAP)
        else:
            # Fallback to the main data utility which handles generation, caching,
            # and consistency checks.
            _, self.user_map, self.item_map = load_processed_data(
                load_cached_data=load_cached_data
            )

        # Ensure maps are sorted by index to guarantee array indexing works for inverse_transform
        self.user_map = self.user_map.sort_values("user_idx")
        self.item_map = self.item_map.sort_values("item_idx")

        # Build fast lookup dictionaries for transform
        self.user_to_idx = dict(
            zip(self.user_map["customer_id"], self.user_map["user_idx"])
        )
        self.item_to_idx = dict(
            zip(self.item_map["article_id"], self.item_map["item_idx"])
        )

        # Build arrays for fast inverse_transform
        self.idx_to_user = self.user_map["customer_id"].values
        self.idx_to_item = self.item_map["article_id"].values

        return self

    def transform(self, ids, entity):
        """
        Maps raw IDs to integer indices.

        Args:
            ids: A single ID or a list/array of IDs.
            entity (str): 'user' (for customer_id) or 'item' (for article_id).

        Returns:
            int or np.array: The integer index/indices. Returns -1 for unknown IDs.
        """
        if entity == "user":
            mapping = self.user_to_idx
        elif entity == "item":
            mapping = self.item_to_idx
        else:
            raise ValueError("Entity must be 'user' or 'item'")

        # Helper function for single value lookup
        def get_idx(x):
            return mapping.get(x, -1)

        if isinstance(ids, (list, np.ndarray, pd.Series)):
            return np.array([get_idx(x) for x in ids], dtype=np.int32)
        else:
            return get_idx(ids)

    def inverse_transform(self, indices, entity):
        """
        Maps integer indices back to raw IDs.

        Args:
            indices: A single integer index or a list/array of indices.
            entity (str): 'user' or 'item'.

        Returns:
            str/int or np.array: The raw ID(s).
        """
        if entity == "user":
            arr = self.idx_to_user
        elif entity == "item":
            arr = self.idx_to_item
        else:
            raise ValueError("Entity must be 'user' or 'item'")

        if isinstance(indices, (list, np.ndarray, pd.Series)):
            indices = np.asanyarray(indices, dtype=int)
            return arr[indices]
        else:
            return arr[int(indices)]

    def get_user_count(self):
        """Returns the total number of unique users in the mapping."""
        return len(self.user_map) if self.user_map is not None else 0

    def get_item_count(self):
        """Returns the total number of unique items in the mapping."""
        return len(self.item_map) if self.item_map is not None else 0
