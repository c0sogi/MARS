import pandas as pd
import numpy as np
import os
from pathlib import Path
from library.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    WORKING_DIR,
    HISTORY_WEEKS,
)


class IdEncoder:
    """
    Encodes string identifiers (customer_id, article_id) to integers and back.
    Persists mappings to disk to ensure consistency across runs.
    """

    def __init__(self):
        self.customer_to_idx = {}
        self.idx_to_customer = {}
        self.article_to_idx = {}
        self.idx_to_article = {}

    def fit(self, customers, articles, load_cached_data=True):
        """
        Fits the encoder on the provided customer and article lists.
        If cached mappings exist and load_cached_data is True, loads them instead.
        """
        cust_map_path = WORKING_DIR / "customer_map.parquet"
        art_map_path = WORKING_DIR / "article_map.parquet"

        WORKING_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Try to load from cache
        if load_cached_data and cust_map_path.exists() and art_map_path.exists():
            print("Loading ID mappings from cache...")
            cust_df = pd.read_parquet(cust_map_path)
            art_df = pd.read_parquet(art_map_path)

            self.customer_to_idx = dict(zip(cust_df["customer_id"], cust_df["id"]))
            self.idx_to_customer = dict(zip(cust_df["id"], cust_df["customer_id"]))

            self.article_to_idx = dict(zip(art_df["article_id"], art_df["id"]))
            self.idx_to_article = dict(zip(art_df["id"], art_df["article_id"]))
            return

        # 2. Compute from scratch
        print("Computing ID mappings...")
        # Use numpy unique for sorting and speed
        unique_customers = np.unique(customers)
        unique_articles = np.unique(articles)

        # Create dictionaries
        self.customer_to_idx = {cid: i for i, cid in enumerate(unique_customers)}
        self.idx_to_customer = {i: cid for i, cid in enumerate(unique_customers)}

        self.article_to_idx = {aid: i for i, aid in enumerate(unique_articles)}
        self.idx_to_article = {i: aid for i, aid in enumerate(unique_articles)}

        # 3. Save to cache
        # Convert to DataFrame for Parquet storage
        cust_df = pd.DataFrame(
            {
                "customer_id": list(self.customer_to_idx.keys()),
                "id": list(self.customer_to_idx.values()),
            }
        )
        # Ensure IDs are strings
        cust_df["customer_id"] = cust_df["customer_id"].astype(str)

        art_df = pd.DataFrame(
            {
                "article_id": list(self.article_to_idx.keys()),
                "id": list(self.article_to_idx.values()),
            }
        )
        art_df["article_id"] = art_df["article_id"].astype(str)

        cust_df.to_parquet(cust_map_path, index=False)
        art_df.to_parquet(art_map_path, index=False)
        print("ID mappings saved to cache.")

    def transform_customers(self, customers):
        """
        Transforms a list/series of customer_ids to their integer encodings.
        Unknown customers are mapped to -1.
        """
        if not isinstance(customers, pd.Series):
            customers = pd.Series(customers)
        # Use map for efficiency with large arrays
        return customers.map(self.customer_to_idx).fillna(-1).astype(int).values

    def transform_articles(self, articles):
        """
        Transforms a list/series of article_ids to their integer encodings.
        Unknown articles are mapped to -1.
        """
        if not isinstance(articles, pd.Series):
            articles = pd.Series(articles)
        return articles.map(self.article_to_idx).fillna(-1).astype(int).values

    def inverse_transform_articles(self, article_idxs):
        """
        Transforms a list/array of integer encodings back to article_id strings.
        """
        return [self.idx_to_article.get(idx, "") for idx in article_idxs]


def load_and_filter_data(load_cached_data=True):
    """
    Loads training and validation data.
    Filters training data to keep only the last HISTORY_WEEKS of transactions.
    Uses caching to speed up subsequent runs.
    """
    filtered_train_path = WORKING_DIR / "filtered_train.parquet"

    # 1. Try to load filtered train from cache
    if load_cached_data and filtered_train_path.exists():
        print("Loading filtered training data from cache...")
        train_df = pd.read_parquet(filtered_train_path)
        val_df = pd.read_parquet(VAL_DATA_PATH)
        val_df["t_dat"] = pd.to_datetime(val_df["t_dat"])
        return train_df, val_df

    # 2. Load raw data
    print("Loading raw data from metadata...")
    train_df = pd.read_parquet(TRAIN_DATA_PATH)
    val_df = pd.read_parquet(VAL_DATA_PATH)

    # 3. Filter Training Data
    print(f"Filtering training data to last {HISTORY_WEEKS} weeks...")
    # Ensure t_dat is datetime
    train_df["t_dat"] = pd.to_datetime(train_df["t_dat"])
    val_df["t_dat"] = pd.to_datetime(val_df["t_dat"])

    max_date = train_df["t_dat"].max()
    cutoff_date = max_date - pd.Timedelta(weeks=HISTORY_WEEKS)

    # Apply filter
    train_df = train_df[train_df["t_dat"] > cutoff_date].copy()

    # 4. Save to cache
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(filtered_train_path, index=False)
    print("Filtered training data saved to cache.")

    return train_df, val_df


def get_target_customers():
    """
    Loads the list of customers requiring predictions (test set).
    """
    return pd.read_parquet(TEST_DATA_PATH)
