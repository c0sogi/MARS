import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and Pandas to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def load_metadata(split: str = "train") -> pd.DataFrame:
    """
    Loads the pre-generated metadata (transactions/submission) for the specified split.

    Args:
        split (str): One of 'train', 'val', or 'test'.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    if split == "train":
        path = Config.TRAIN_METADATA
    elif split == "val":
        path = Config.VAL_METADATA
    elif split == "test":
        path = Config.TEST_METADATA
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    if not path.exists():
        raise FileNotFoundError(
            f"Metadata file not found at {path}. Please run metadata generation first."
        )

    return pd.read_parquet(path)


def load_articles(load_cached_data: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Loads and preprocesses article data. Generates a dense index mapping for article_ids.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (processed_articles_df, article_to_idx_map)
            - processed_articles_df: DataFrame with encoded features and 'article_idx' column.
            - article_to_idx_map: Dictionary mapping raw article_id to dense integer index.
    """
    cache_path_df = Config.WORKING_DIR / "articles_processed.parquet"

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Try loading from cache
    if (
        load_cached_data
        and cache_path_df.exists()
        and Config.ARTICLE_ID_MAP_PATH.exists()
    ):
        df = pd.read_parquet(cache_path_df)
        # Load map from npy (saved as 2 arrays: keys and values, or just assume index is value)
        # To be safe and flexible, we reconstruct the dict from the df or load specific map file
        # Here we load the specific map file if we saved it as a dictionary pickle or similar,
        # but the requirement says no pickle.
        # Strategy: The df has 'article_id' and 'article_idx'. We can reconstruct the map.
        article_map = dict(zip(df["article_id"], df["article_idx"]))
        return df, article_map

    # Process from scratch
    df = pd.read_csv(Config.ARTICLES_CSV)

    # 1. Handle Missing Values
    # For categorical columns, fill with "Unknown"
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    # 2. Label Encoding for Categorical Features (useful for LightGBM)
    # We keep the original IDs but also create encoded versions for features
    le = LabelEncoder()
    for col in cat_cols:
        # Create a new column with _idx suffix
        df[f"{col}_idx"] = le.fit_transform(df[col].astype(str))

    # 3. Create Dense Index for Matrix Operations
    # We sort by article_id to ensure deterministic mapping
    unique_articles = sorted(df["article_id"].unique())
    article_map = {aid: i for i, aid in enumerate(unique_articles)}

    # Map back to dataframe
    df["article_idx"] = df["article_id"].map(article_map)

    # 4. Generate Image Paths (Static feature)
    # Format: images/0xx/0xxxxxxxxx.jpg
    # Vectorized operation
    aid_str = df["article_id"].astype(str).str.zfill(10)
    subfolders = aid_str.str[:3]
    filenames = aid_str + ".jpg"
    df["image_path"] = "images/" + subfolders + "/" + filenames

    # 5. Save to Cache
    df.to_parquet(cache_path_df, index=False)

    # Save the map keys as npy for fast loading if needed, though reconstructing from DF is fast enough.
    # We satisfy the Config path requirement by saving the sorted unique IDs.
    # The index in this array corresponds to the dense ID.
    np.save(Config.ARTICLE_ID_MAP_PATH, np.array(unique_articles))

    return df, article_map


def load_customers(load_cached_data: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Loads and preprocesses customer data. Generates a dense index mapping for customer_ids.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (processed_customers_df, customer_to_idx_map)
            - processed_customers_df: DataFrame with encoded features and 'customer_idx' column.
            - customer_to_idx_map: Dictionary mapping raw customer_id (hash) to dense integer index.
    """
    cache_path_df = Config.WORKING_DIR / "customers_processed.parquet"

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    if (
        load_cached_data
        and cache_path_df.exists()
        and Config.CUSTOMER_ID_MAP_PATH.exists()
    ):
        df = pd.read_parquet(cache_path_df)
        customer_map = dict(zip(df["customer_id"], df["customer_idx"]))
        return df, customer_map

    # Process from scratch
    df = pd.read_csv(Config.CUSTOMERS_CSV)

    # 1. Handle Missing Values
    # FN and Active are 1.0 or NaN. Fill NaN with 0.
    df["FN"] = df["FN"].fillna(0)
    df["Active"] = df["Active"].fillna(0)

    # Age: Fill with median
    median_age = df["age"].median()
    df["age"] = df["age"].fillna(median_age)

    # Club Member Status & Fashion News Frequency
    df["club_member_status"] = df["club_member_status"].fillna("Unknown")
    df["fashion_news_frequency"] = df["fashion_news_frequency"].fillna("Unknown")

    # 2. Label Encoding
    le = LabelEncoder()
    df["club_member_status_idx"] = le.fit_transform(
        df["club_member_status"].astype(str)
    )

    # Normalize fashion news frequency (sometimes has 'None' vs 'NONE')
    df["fashion_news_frequency"] = df["fashion_news_frequency"].astype(str).str.upper()
    df["fashion_news_frequency_idx"] = le.fit_transform(df["fashion_news_frequency"])

    # 3. Create Dense Index
    # We must include customers from the sample submission as well to ensure we can map them.
    # However, customers.csv usually contains all customers.
    # We sort to ensure deterministic mapping.
    unique_customers = sorted(df["customer_id"].unique())
    customer_map = {cid: i for i, cid in enumerate(unique_customers)}

    df["customer_idx"] = df["customer_id"].map(customer_map)

    # 4. Save to Cache
    df.to_parquet(cache_path_df, index=False)

    # Save the sorted unique IDs to NPY. Index in array = dense ID.
    np.save(Config.CUSTOMER_ID_MAP_PATH, np.array(unique_customers))

    return df, customer_map


def get_article_image_paths(article_ids: list) -> list:
    """
    Generates relative image paths for a list of article IDs.

    Args:
        article_ids (list or np.array): List of article IDs (int or str).

    Returns:
        list: List of string paths (e.g., 'images/010/0101234567.jpg').
    """
    paths = []
    for aid in article_ids:
        s = str(aid).zfill(10)
        folder = s[:3]
        paths.append(f"images/{folder}/{s}.jpg")
    return paths
