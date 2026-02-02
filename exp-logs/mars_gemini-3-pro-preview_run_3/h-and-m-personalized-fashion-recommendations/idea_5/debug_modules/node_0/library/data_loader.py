import os
import pandas as pd
import numpy as np
from library import config


def _ensure_working_dir():
    """Ensures the working directory exists."""
    os.makedirs(config.WORKING_DIR, exist_ok=True)


def get_image_paths(article_ids):
    """
    Converts a list/series of article_ids (int or str) to their corresponding image paths.
    Format: images/xxx/0xxxxxxxx.jpg
    """
    # Ensure article_ids are strings with leading zeros (10 chars)
    # If input is a pandas Series, use vectorized string operations
    if isinstance(article_ids, pd.Series):
        ids_str = article_ids.astype(str).str.zfill(10)
        folders = ids_str.str[:3]
        return "images/" + folders + "/" + ids_str + ".jpg"

    # Fallback for list/numpy array
    paths = []
    for aid in article_ids:
        s = str(aid).zfill(10)
        folder = s[:3]
        paths.append(f"images/{folder}/{s}.jpg")
    return paths


def load_transactions(split="train", load_cached_data=True):
    """
    Loads transaction data for the specified split ('train' or 'val').

    Args:
        split (str): 'train' or 'val'.
        load_cached_data (bool): If True, tries to load processed parquet from working dir.

    Returns:
        pd.DataFrame: Transaction data.
    """
    _ensure_working_dir()
    cache_path = config.WORKING_DIR / f"transactions_{split}_processed.parquet"

    if load_cached_data and cache_path.exists():
        print(f"Loading cached {split} transactions from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Processing {split} transactions from metadata...")
    if split == "train":
        source_path = config.TRAIN_METADATA_PATH
    elif split == "val":
        source_path = config.VAL_METADATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")

    df = pd.read_parquet(source_path)

    # Type optimization
    df["t_dat"] = pd.to_datetime(df["t_dat"])

    # article_id fits in int32 (max ~9e8 < 2e9)
    if "article_id" in df.columns:
        df["article_id"] = df["article_id"].astype("int32")

    if "sales_channel_id" in df.columns:
        df["sales_channel_id"] = df["sales_channel_id"].astype("int8")

    if "price" in df.columns:
        df["price"] = df["price"].astype("float32")

    # Save to cache
    print(f"Saving processed {split} transactions to {cache_path}...")
    df.to_parquet(cache_path, index=False)

    return df


def load_articles(load_cached_data=True):
    """
    Loads article metadata. Adds 'image_path' column.

    Args:
        load_cached_data (bool): If True, tries to load processed parquet from working dir.

    Returns:
        pd.DataFrame: Articles data.
    """
    _ensure_working_dir()
    cache_path = config.WORKING_DIR / "articles_processed.parquet"

    if load_cached_data and cache_path.exists():
        print(f"Loading cached articles from {cache_path}...")
        return pd.read_parquet(cache_path)

    print("Processing articles from raw CSV...")
    df = pd.read_csv(config.ARTICLES_CSV)

    # Add image_path column
    df["image_path"] = get_image_paths(df["article_id"])

    # Type optimization
    df["article_id"] = df["article_id"].astype("int32")

    # Convert object columns with low cardinality to category
    for col in df.select_dtypes(include=["object"]).columns:
        if col != "detail_desc" and col != "image_path":
            num_unique = df[col].nunique()
            if num_unique < 1000:
                df[col] = df[col].astype("category")

    # Save to cache
    print(f"Saving processed articles to {cache_path}...")
    df.to_parquet(cache_path, index=False)

    return df


def load_customers(load_cached_data=True):
    """
    Loads customer metadata.

    Args:
        load_cached_data (bool): If True, tries to load processed parquet from working dir.

    Returns:
        pd.DataFrame: Customer data.
    """
    _ensure_working_dir()
    cache_path = config.WORKING_DIR / "customers_processed.parquet"

    if load_cached_data and cache_path.exists():
        print(f"Loading cached customers from {cache_path}...")
        return pd.read_parquet(cache_path)

    print("Processing customers from raw CSV...")
    df = pd.read_csv(config.CUSTOMERS_CSV)

    # Fill NaNs for categorical columns
    df["club_member_status"] = df["club_member_status"].fillna("NONE")
    df["fashion_news_frequency"] = df["fashion_news_frequency"].fillna("NONE")

    # Type optimization
    cat_cols = ["club_member_status", "fashion_news_frequency", "postal_code"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    if "age" in df.columns:
        df["age"] = df["age"].fillna(-1).astype("int8")

    # Save to cache
    print(f"Saving processed customers to {cache_path}...")
    df.to_parquet(cache_path, index=False)

    return df


def load_test_customers(load_cached_data=True):
    """
    Loads the list of customers for whom predictions are required (submission format).

    Args:
        load_cached_data (bool): If True, tries to load processed parquet from working dir.

    Returns:
        pd.DataFrame: Test customer IDs.
    """
    _ensure_working_dir()
    cache_path = config.WORKING_DIR / "test_customers_processed.parquet"

    if load_cached_data and cache_path.exists():
        print(f"Loading cached test customers from {cache_path}...")
        return pd.read_parquet(cache_path)

    print("Loading test customers from metadata...")
    # The test metadata is already in parquet format in the metadata dir
    df = pd.read_parquet(config.TEST_METADATA_PATH)

    # Just ensure we have the customer_id column and it's clean
    if "prediction" in df.columns:
        df = df.drop(columns=["prediction"])

    # Save to cache (mostly for consistency)
    print(f"Saving processed test customers to {cache_path}...")
    df.to_parquet(cache_path, index=False)

    return df
