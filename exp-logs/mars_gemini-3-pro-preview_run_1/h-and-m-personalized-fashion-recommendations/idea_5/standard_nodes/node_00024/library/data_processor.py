import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import gc
from datetime import timedelta
from library import config


class DataLoader:
    """
    Handles data ingestion, preprocessing, and mapping generation.
    Implements caching to optimize runtime for iterative development.
    """

    def __init__(self):
        self.train_path = config.TRAIN_DATA_PATH
        self.val_path = config.VAL_DATA_PATH
        self.test_path = config.TEST_DATA_PATH
        self.articles_path = config.ARTICLES_PATH

        self.cache_trans = config.CACHE_TRANSACTIONS_PROCESSED
        self.cache_item_map = config.CACHE_ITEM_MAP
        self.cache_user_map = config.CACHE_USER_MAP

        # Ensure working directory exists
        os.makedirs(config.WORKING_DIR, exist_ok=True)

    def load_data(self, load_cached_data=True):
        """
        Loads data, filtering by time window, and creating integer mappings.

        Args:
            load_cached_data (bool): If True, attempts to load from Parquet cache.

        Returns:
            train_df (pd.DataFrame): Filtered training transactions with integer indices.
            val_df (pd.DataFrame): Validation transactions with integer indices.
            test_df (pd.DataFrame): Test customers with integer indices.
            articles_df (pd.DataFrame): Raw articles metadata.
            user_map (pd.Series): Mapping from customer_id (str) to user_idx (int).
            item_map (pd.Series): Mapping from article_id (int) to item_idx (int).
        """
        # Check cache existence
        if (
            load_cached_data
            and os.path.exists(self.cache_trans)
            and os.path.exists(self.cache_item_map)
            and os.path.exists(self.cache_user_map)
        ):
            print("Loading cached data...")
            try:
                # Load cached files
                train_df = pd.read_parquet(self.cache_trans)
                item_map_df = pd.read_parquet(self.cache_item_map)
                user_map_df = pd.read_parquet(self.cache_user_map)

                # Reconstruct Series from DataFrame
                item_map = pd.Series(
                    data=item_map_df["item_idx"].values,
                    index=item_map_df["article_id"].values,
                )
                user_map = pd.Series(
                    data=user_map_df["user_idx"].values,
                    index=user_map_df["customer_id"].values,
                )

                # Load auxiliary files
                articles_df = pd.read_csv(self.articles_path, dtype={"article_id": int})
                val_df = pd.read_csv(self.val_path)
                test_df = pd.read_csv(self.test_path)

                # Apply mappings to Val and Test (Computed on fly to ensure consistency)
                # Validation Data
                val_df["article_id"] = val_df["article_id"].astype(int)
                val_df = val_df[val_df["article_id"].isin(item_map.index)].copy()
                val_df["item_idx"] = val_df["article_id"].map(item_map).astype("int32")
                val_df = val_df[val_df["customer_id"].isin(user_map.index)].copy()
                val_df["user_idx"] = val_df["customer_id"].map(user_map).astype("int32")

                # Test Data
                test_df = test_df[test_df["customer_id"].isin(user_map.index)].copy()
                test_df["user_idx"] = (
                    test_df["customer_id"].map(user_map).astype("int32")
                )

                return train_df, val_df, test_df, articles_df, user_map, item_map
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing from scratch...")

        print("Computing data from scratch...")
        return self._process_data()

    def _process_data(self):
        """
        Internal method to process raw CSVs, generate mappings, and save to cache.
        """
        # 1. Load Articles (Item Universe)
        print("Loading articles...")
        articles_df = pd.read_csv(self.articles_path, dtype={"article_id": int})

        # Create Item Map (All available articles map to 0..N-1)
        articles_df["item_idx"] = np.arange(len(articles_df), dtype="int32")
        item_map = pd.Series(
            data=articles_df["item_idx"].values, index=articles_df["article_id"].values
        )

        # 2. Load Train
        print("Loading training transactions...")
        train_df = pd.read_csv(self.train_path, dtype={"article_id": int})
        train_df["t_dat"] = pd.to_datetime(train_df["t_dat"])

        # Filter by Time Window (Last N weeks)
        max_date = train_df["t_dat"].max()
        cutoff_date = max_date - timedelta(weeks=config.TRAIN_WEEKS)
        print(f"Filtering transactions after {cutoff_date}...")
        train_df = train_df[train_df["t_dat"] > cutoff_date].copy()

        # 3. Load Test and Val (User Universe)
        print("Loading validation and test sets...")
        test_df = pd.read_csv(self.test_path)
        val_df = pd.read_csv(self.val_path)

        # Get all unique customers from Train (filtered), Val, and Test
        # This ensures our user_idx covers everyone we need to predict for or evaluate on.
        unique_users = pd.concat(
            [train_df["customer_id"], val_df["customer_id"], test_df["customer_id"]]
        ).unique()

        print(f"Total unique users: {len(unique_users)}")

        # Create User Map
        user_map_df = pd.DataFrame({"customer_id": unique_users})
        user_map_df["user_idx"] = np.arange(len(user_map_df), dtype="int32")
        user_map = pd.Series(
            data=user_map_df["user_idx"].values, index=user_map_df["customer_id"].values
        )

        # 4. Apply Mappings to Train
        print("Mapping training data...")
        # Filter train items to those in articles (safety check)
        train_df = train_df[train_df["article_id"].isin(item_map.index)].copy()
        train_df["item_idx"] = train_df["article_id"].map(item_map).astype("int32")
        train_df["user_idx"] = train_df["customer_id"].map(user_map).astype("int32")

        # 5. Apply Mappings to Val/Test
        print("Mapping validation and test data...")
        val_df["article_id"] = val_df["article_id"].astype(int)
        val_df = val_df[val_df["article_id"].isin(item_map.index)].copy()
        val_df["item_idx"] = val_df["article_id"].map(item_map).astype("int32")
        val_df = val_df[val_df["customer_id"].isin(user_map.index)].copy()
        val_df["user_idx"] = val_df["customer_id"].map(user_map).astype("int32")

        test_df = test_df[test_df["customer_id"].isin(user_map.index)].copy()
        test_df["user_idx"] = test_df["customer_id"].map(user_map).astype("int32")

        # 6. Save Cache
        print("Saving cache to disk...")
        train_df.to_parquet(self.cache_trans, index=False)

        item_map_save = pd.DataFrame(
            {"article_id": item_map.index, "item_idx": item_map.values}
        )
        item_map_save.to_parquet(self.cache_item_map, index=False)

        user_map_df.to_parquet(self.cache_user_map, index=False)

        # Clean up
        gc.collect()

        return train_df, val_df, test_df, articles_df, user_map, item_map


def get_interaction_matrix(
    df, n_users, n_items, user_col="user_idx", item_col="item_idx"
):
    """
    Constructs a binary User-Item interaction matrix (CSR).
    Used for calculating Item-Item similarity (X^T X).

    Args:
        df (pd.DataFrame): Transaction dataframe.
        n_users (int): Total number of users (dimension 0).
        n_items (int): Total number of items (dimension 1).
        user_col (str): Column name for user indices.
        item_col (str): Column name for item indices.

    Returns:
        sp.csr_matrix: Sparse binary interaction matrix.
    """
    # Drop duplicates to ensure binary interaction (bought at least once in the window)
    df_uniq = df[[user_col, item_col]].drop_duplicates()

    data = np.ones(len(df_uniq), dtype=config.PRECISION)
    row = df_uniq[user_col].values
    col = df_uniq[item_col].values

    mat = sp.csr_matrix(
        (data, (row, col)), shape=(n_users, n_items), dtype=config.PRECISION
    )
    return mat


def get_variant_matrix(articles_df, item_map):
    """
    Constructs the Variant Similarity Matrix (S_variant).
    S_ij = 1 if item i and item j share the same product_code, else 0.
    This effectively links different colors of the same product.

    Args:
        articles_df (pd.DataFrame): Articles metadata.
        item_map (pd.Series): Mapping from article_id to item_idx.

    Returns:
        sp.csr_matrix: Sparse binary item-item similarity matrix.
    """
    print("Constructing Variant Matrix...")
    # Work on a copy
    df = articles_df.copy()

    # Ensure item_idx is present
    if "item_idx" not in df.columns:
        # Filter to items in map
        df = df[df["article_id"].isin(item_map.index)]
        df["item_idx"] = df["article_id"].map(item_map).astype("int32")

    # Encode product_code to integer for sparse matrix construction
    df["product_code_idx"] = df["product_code"].astype("category").cat.codes

    n_items = len(item_map)
    n_products = df["product_code_idx"].max() + 1

    # Create Item-Product Matrix A (Items x Products)
    # A_ip = 1 if item i belongs to product p
    row = df["item_idx"].values
    col = df["product_code_idx"].values
    data = np.ones(len(row), dtype=config.PRECISION)

    A = sp.csr_matrix(
        (data, (row, col)), shape=(n_items, n_products), dtype=config.PRECISION
    )

    # S_variant = A * A^T
    # This creates a clique for each product code (all items with same product_code are connected)
    S_variant = A.dot(A.T)

    # Remove diagonal (self-similarity)
    S_variant.setdiag(0)

    # Ensure binary (1.0)
    S_variant.data = np.ones_like(S_variant.data, dtype=config.PRECISION)

    return S_variant
