import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from datetime import timedelta
import library.config as config


def load_and_preprocess_articles(load_cached_data=True):
    """
    Loads articles metadata, performs preprocessing (LabelEncoding), and returns the dataframe.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        pd.DataFrame: Preprocessed articles dataframe with additional '{col}_idx' columns.
    """
    cache_path = config.WORKING_DIR / "articles_preprocessed.parquet"

    if load_cached_data and cache_path.exists():
        print(f"Loading preprocessed articles from {cache_path}")
        return pd.read_parquet(cache_path)

    print("Preprocessing articles...")
    df = pd.read_csv(config.ARTICLES_CSV)

    # Fill missing values for object columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("Unknown")

    # Label Encode categorical columns useful for the ranker
    categorical_cols = [
        "product_type_name",
        "product_group_name",
        "graphical_appearance_name",
        "colour_group_name",
        "perceived_colour_value_name",
        "perceived_colour_master_name",
        "department_name",
        "index_name",
        "index_group_name",
        "section_name",
        "garment_group_name",
    ]

    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            # Convert to string to ensure uniformity before encoding
            df[col] = df[col].astype(str)
            df[f"{col}_idx"] = le.fit_transform(df[col])

    # Save to cache
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


def load_and_preprocess_customers(load_cached_data=True):
    """
    Loads customer metadata, performs preprocessing, and returns the dataframe.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        pd.DataFrame: Preprocessed customers dataframe with imputed values and encoded features.
    """
    cache_path = config.WORKING_DIR / "customers_preprocessed.parquet"

    if load_cached_data and cache_path.exists():
        print(f"Loading preprocessed customers from {cache_path}")
        return pd.read_parquet(cache_path)

    print("Preprocessing customers...")
    df = pd.read_csv(config.CUSTOMERS_CSV)

    # Handle Missing Values
    # FN and Active: NaN -> 0
    if "FN" in df.columns:
        df["FN"] = df["FN"].fillna(0)
    if "Active" in df.columns:
        df["Active"] = df["Active"].fillna(0)

    # Club member status
    if "club_member_status" in df.columns:
        df["club_member_status"] = df["club_member_status"].fillna("NONE")

    # Fashion news frequency
    if "fashion_news_frequency" in df.columns:
        df["fashion_news_frequency"] = df["fashion_news_frequency"].fillna("NONE")

    # Age: Fill with median
    if "age" in df.columns and df["age"].isnull().sum() > 0:
        median_age = df["age"].median()
        df["age"] = df["age"].fillna(median_age)

    # Label Encode
    le = LabelEncoder()
    cat_cols = ["club_member_status", "fashion_news_frequency"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[f"{col}_idx"] = le.fit_transform(df[col])

    # Save to cache
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


def load_train_data_split(val_days=config.VAL_DAYS, load_cached_data=True):
    """
    Loads the training metadata and splits it into:
    1. Retrieval History (Data <= Max Date - val_days)
    2. Ranker Target (Data > Max Date - val_days)

    Args:
        val_days (int): Number of days to hold out for validation/ranker targets.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (df_history, df_target)
    """
    cache_history = config.WORKING_DIR / "train_history.parquet"
    cache_target = config.WORKING_DIR / "train_target.parquet"

    if load_cached_data and cache_history.exists() and cache_target.exists():
        print("Loading train data splits from cache...")
        return pd.read_parquet(cache_history), pd.read_parquet(cache_target)

    print("Loading and splitting training data...")
    # Load full training metadata
    df = pd.read_parquet(config.TRAIN_METADATA_PATH)

    # Convert date
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])

    # Determine split point
    max_date = df[config.DATE_COL].max()
    split_date = max_date - timedelta(days=val_days)

    print(f"Max Date: {max_date}")
    print(f"Split Date: {split_date}")

    # Split
    # Target: Items purchased in the last 'val_days' (strictly > split_date)
    df_target = df[df[config.DATE_COL] > split_date].copy()

    # History: Items purchased before the target period
    df_history = df[df[config.DATE_COL] <= split_date].copy()

    print(f"History samples: {len(df_history)}")
    print(f"Target samples: {len(df_target)}")

    # Save to cache
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    df_history.to_parquet(cache_history, index=False)
    df_target.to_parquet(cache_target, index=False)

    return df_history, df_target


def load_full_train_data():
    """
    Loads the full training dataset without splitting.
    Used for final inference where we train the retrieval model on the entire history.
    """
    print("Loading full training data...")
    df = pd.read_parquet(config.TRAIN_METADATA_PATH)
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
    return df


def load_test_users():
    """
    Loads the list of customers to predict for (from test metadata).

    Returns:
        np.array: Array of unique customer_ids.
    """
    print("Loading test users...")
    df = pd.read_parquet(config.TEST_METADATA_PATH)
    return df[config.USER_COL].unique()
