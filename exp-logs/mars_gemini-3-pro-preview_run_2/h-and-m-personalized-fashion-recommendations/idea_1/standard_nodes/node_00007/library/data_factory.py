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
    Implements a Time-based Split (Cite solution_lesson_node_00003) to ensure
    validation users have history in the training set.
    """
    train_cache = WORKING_DIR / "time_split_train.parquet"
    val_cache = WORKING_DIR / "time_split_val.parquet"

    # 1. Try to load time-split data from cache
    if load_cached_data and train_cache.exists() and val_cache.exists():
        print("Loading time-split data from cache...")
        train_df = pd.read_parquet(train_cache)
        val_df = pd.read_parquet(val_cache)
        return train_df, val_df

    # 2. Load raw data and combine
    print("Loading raw data from metadata and combining...")
    # The original files were split by customer, which is incorrect for this task.
    # We combine them to perform a proper time-based split.
    df1 = pd.read_parquet(TRAIN_DATA_PATH)
    df2 = pd.read_parquet(VAL_DATA_PATH)
    full_df = pd.concat([df1, df2], axis=0)

    full_df["t_dat"] = pd.to_datetime(full_df["t_dat"])

    # 3. Time-based Split
    # Validation = Last 7 days of available data
    max_date = full_df["t_dat"].max()
    val_start_date = max_date - pd.Timedelta(days=7)

    print(f"Performing Time-based Split (Cite solution_lesson_node_00003)...")
    print(f"  Max Date: {max_date}")
    print(f"  Validation Start: {val_start_date}")

    val_df = full_df[full_df["t_dat"] > val_start_date].copy()
    train_df = full_df[full_df["t_dat"] <= val_start_date].copy()

    # 4. Filter Training Data History
    print(f"Filtering training data to last {HISTORY_WEEKS} weeks before validation...")
    train_start_date = val_start_date - pd.Timedelta(weeks=HISTORY_WEEKS)
    train_df = train_df[train_df["t_dat"] > train_start_date].copy()

    # 5. Save to cache
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    print("Time-split data saved to cache.")

    return train_df, val_df


def get_target_customers():
    """
    Loads the list of customers requiring predictions (test set).
    """
    return pd.read_parquet(TEST_DATA_PATH)
