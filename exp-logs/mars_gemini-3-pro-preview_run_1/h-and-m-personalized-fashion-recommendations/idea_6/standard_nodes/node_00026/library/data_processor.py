import pandas as pd
import numpy as np
import os
import gc
from datetime import timedelta
from library.config import Config


def load_and_filter_data(load_cached_data=True):
    """
    Loads training and validation data from metadata, filters by time window,
    and loads the test customer list.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache paths
    cache_train = os.path.join(Config.CACHE_DIR, "train_filtered.parquet")
    cache_val = os.path.join(Config.CACHE_DIR, "val_filtered.parquet")
    cache_test = os.path.join(Config.CACHE_DIR, "test_customers.parquet")

    # Check cache
    if (
        load_cached_data
        and os.path.exists(cache_train)
        and os.path.exists(cache_val)
        and os.path.exists(cache_test)
    ):
        print("Loading filtered data from cache...")
        train_df = pd.read_parquet(cache_train)
        val_df = pd.read_parquet(cache_val)
        test_df = pd.read_parquet(cache_test)
        return train_df, val_df, test_df

    print("Loading and filtering data from scratch...")

    # Load metadata
    # Using specific dtypes to save memory
    train_path = os.path.join(Config.METADATA_DIR, "train.csv")
    val_path = os.path.join(Config.METADATA_DIR, "val.csv")
    test_path = os.path.join(Config.METADATA_DIR, "test.csv")

    train_df = pd.read_csv(
        train_path, dtype={"article_id": "int32", "sales_channel_id": "int8"}
    )
    val_df = pd.read_csv(
        val_path, dtype={"article_id": "int32", "sales_channel_id": "int8"}
    )
    test_df = pd.read_csv(test_path)

    # Convert dates
    train_df["t_dat"] = pd.to_datetime(train_df["t_dat"])
    val_df["t_dat"] = pd.to_datetime(val_df["t_dat"])

    # Calculate cutoff date
    # We use the maximum date across both datasets to define the "current" time
    # Then subtract the configured training weeks
    max_date = max(train_df["t_dat"].max(), val_df["t_dat"].max())
    cutoff_date = max_date - timedelta(days=Config.TRAIN_WEEKS * 7)

    print(f"Filtering transactions after {cutoff_date}...")

    # Filter data
    train_filtered = train_df[train_df["t_dat"] > cutoff_date].reset_index(drop=True)
    val_filtered = val_df[val_df["t_dat"] > cutoff_date].reset_index(drop=True)

    # Save to cache
    print("Saving filtered data to cache...")
    train_filtered.to_parquet(cache_train, index=False)
    val_filtered.to_parquet(cache_val, index=False)
    test_df.to_parquet(cache_test, index=False)

    return train_filtered, val_filtered, test_df


def create_mappings(train_df, val_df, test_df, load_cached_data=True):
    """
    Creates bidirectional mappings for customers and articles.
    Ensures all customers in test set are included.

    Args:
        train_df (pd.DataFrame): Filtered training transactions.
        val_df (pd.DataFrame): Filtered validation transactions.
        test_df (pd.DataFrame): Test set customers.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (user_to_idx, idx_to_user, item_to_idx, idx_to_item)
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    cache_user_map = os.path.join(Config.CACHE_DIR, "user_map.parquet")
    cache_item_map = os.path.join(Config.CACHE_DIR, "item_map.parquet")

    if (
        load_cached_data
        and os.path.exists(cache_user_map)
        and os.path.exists(cache_item_map)
    ):
        print("Loading mappings from cache...")
        user_map_df = pd.read_parquet(cache_user_map)
        item_map_df = pd.read_parquet(cache_item_map)

        user_to_idx = dict(zip(user_map_df["customer_id"], user_map_df["user_idx"]))
        idx_to_user = dict(zip(user_map_df["user_idx"], user_map_df["customer_id"]))
        item_to_idx = dict(zip(item_map_df["article_id"], item_map_df["item_idx"]))
        idx_to_item = dict(zip(item_map_df["item_idx"], item_map_df["article_id"]))

        return user_to_idx, idx_to_user, item_to_idx, idx_to_item

    print("Generating mappings...")

    # Users: Union of Train, Val, and Test
    # It is critical to include test users so we can generate predictions for them
    unique_users = pd.concat(
        [train_df["customer_id"], val_df["customer_id"], test_df["customer_id"]]
    ).unique()
    # Sort for determinism
    unique_users.sort()

    # Items: Union of Train and Val
    # We only map items that appear in the filtered history
    unique_items = pd.concat([train_df["article_id"], val_df["article_id"]]).unique()
    unique_items.sort()

    # Create DataFrames
    user_map_df = pd.DataFrame(
        {
            "customer_id": unique_users,
            "user_idx": np.arange(len(unique_users), dtype=np.int32),
        }
    )

    item_map_df = pd.DataFrame(
        {
            "article_id": unique_items,
            "item_idx": np.arange(len(unique_items), dtype=np.int32),
        }
    )

    # Cache
    print("Saving mappings to cache...")
    user_map_df.to_parquet(cache_user_map, index=False)
    item_map_df.to_parquet(cache_item_map, index=False)

    # Create dictionaries
    user_to_idx = dict(zip(user_map_df["customer_id"], user_map_df["user_idx"]))
    idx_to_user = dict(zip(user_map_df["user_idx"], user_map_df["customer_id"]))
    item_to_idx = dict(zip(item_map_df["article_id"], item_map_df["item_idx"]))
    idx_to_item = dict(zip(item_map_df["item_idx"], item_map_df["article_id"]))

    return user_to_idx, idx_to_user, item_to_idx, idx_to_item


def process_customer_cohorts(user_to_idx, load_cached_data=True):
    """
    Maps every user index to an age cohort index.

    Args:
        user_to_idx (dict): Mapping from customer_id to user_idx.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        np.ndarray: Array where index i contains the cohort ID for user_idx i.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    cache_path = os.path.join(Config.CACHE_DIR, "user_cohorts.npy")

    if load_cached_data and os.path.exists(cache_path):
        print("Loading cohort data from cache...")
        return np.load(cache_path)

    print("Processing customer cohorts...")

    # Load raw customer data
    customers = pd.read_csv(os.path.join(Config.INPUT_DIR, "customers.csv"))

    # Handle missing age: fill with median
    median_age = customers["age"].median()
    customers["age"] = customers["age"].fillna(median_age)

    # Create Age Bins
    # Bins: 0-10, 10-20, ..., up to 100+
    bins = np.arange(0, 110, Config.AGE_BIN_SIZE)
    # labels=False returns integer indicators
    customers["cohort_idx"] = pd.cut(
        customers["age"], bins=bins, labels=False, right=False
    )

    # Fill any remaining NaNs (e.g. age > 110) with the last bin or 0
    customers["cohort_idx"] = customers["cohort_idx"].fillna(0).astype(np.int8)

    # We need to map these cohort indices to our user_idx space
    # Create a dataframe for the mapping
    user_ids = list(user_to_idx.keys())
    user_indices = list(user_to_idx.values())

    mapping_df = pd.DataFrame({"customer_id": user_ids, "user_idx": user_indices})

    # Merge with customers to get cohort info
    # Note: customers.csv might not contain all users in user_to_idx if data is messy,
    # or user_to_idx might be subset. Left join ensures we keep all mapped users.
    merged_df = mapping_df.merge(
        customers[["customer_id", "cohort_idx"]], on="customer_id", how="left"
    )

    # Fill missing cohorts for users not found in customers.csv
    # Use the mode (most common cohort)
    mode_cohort = merged_df["cohort_idx"].mode()[0]
    merged_df["cohort_idx"] = (
        merged_df["cohort_idx"].fillna(mode_cohort).astype(np.int8)
    )

    # Sort by user_idx to ensure the array is aligned: index i corresponds to user_idx i
    merged_df = merged_df.sort_values("user_idx")

    # Extract the array
    cohort_array = merged_df["cohort_idx"].values

    # Save
    print("Saving cohort data to cache...")
    np.save(cache_path, cohort_array)

    return cohort_array
