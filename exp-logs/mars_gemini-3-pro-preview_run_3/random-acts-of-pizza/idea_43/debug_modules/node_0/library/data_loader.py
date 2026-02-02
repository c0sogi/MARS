import os
import pandas as pd
import numpy as np
from library.config import Config


def load_datasets(debug=False):
    """
    Loads train, validation, and test datasets from the metadata directory.

    Args:
        debug (bool): If True, loads a small subset of the data for debugging purposes.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    if not os.path.exists(Config.TRAIN_PATH):
        raise FileNotFoundError(f"Train metadata not found at {Config.TRAIN_PATH}")
    if not os.path.exists(Config.VAL_PATH):
        raise FileNotFoundError(f"Validation metadata not found at {Config.VAL_PATH}")
    if not os.path.exists(Config.TEST_PATH):
        raise FileNotFoundError(f"Test metadata not found at {Config.TEST_PATH}")

    train_df = pd.read_parquet(Config.TRAIN_PATH)
    val_df = pd.read_parquet(Config.VAL_PATH)
    test_df = pd.read_parquet(Config.TEST_PATH)

    if debug:
        train_df = train_df.sample(
            n=min(200, len(train_df)), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(50, len(val_df)), random_state=Config.SEED
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(50, len(test_df)), random_state=Config.SEED
        ).reset_index(drop=True)

    return train_df, val_df, test_df


def get_text_data(df):
    """
    Extracts and preprocesses text columns defined in Config.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe containing only the text columns, with NaNs filled.
    """
    # Ensure we only fetch columns that exist in the dataframe
    cols = [c for c in Config.TEXT_COLS if c in df.columns]

    if not cols:
        raise ValueError(
            f"None of the configured text columns {Config.TEXT_COLS} found in dataframe."
        )

    text_df = df[cols].copy()

    # Fill missing text with empty string
    for col in cols:
        text_df[col] = text_df[col].fillna("").astype(str)

    return text_df


def get_metadata(df):
    """
    Extracts safe metadata features (allow-list approach) to prevent leakage.
    Explicitly excludes retrieval-time features.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe containing safe numerical and list-based metadata.
    """
    # Explicit Allow-List of safe features available at request time
    # This aligns with the "Augmented Global Metadata" strategy
    safe_features = [
        "unix_timestamp_of_request_utc",
        "requester_account_age_in_days_at_request",
        "requester_days_since_first_post_on_raop_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_comments_in_raop_at_request",
        "requester_number_of_posts_at_request",
        "requester_number_of_posts_on_raop_at_request",
        "requester_number_of_subreddits_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_upvotes_plus_downvotes_at_request",
        # Included for behavioral analysis (NMF)
        "requester_subreddits_at_request",
    ]

    # Filter for columns that actually exist in the dataframe
    selected_features = [col for col in safe_features if col in df.columns]

    # Verification: Ensure no retrieval-time leakage
    for col in selected_features:
        if "at_retrieval" in col:
            # This should technically not happen given the hardcoded list above,
            # but serves as a double-check if the list is modified later.
            raise ValueError(
                f"Potential leakage detected: {col} is a retrieval-time feature."
            )

    meta_df = df[selected_features].copy()

    # Handle the subreddits list column if it contains None/NaN
    if "requester_subreddits_at_request" in meta_df.columns:
        # Ensure it's a list or empty list, not NaN
        meta_df["requester_subreddits_at_request"] = meta_df[
            "requester_subreddits_at_request"
        ].apply(lambda x: x if isinstance(x, (list, np.ndarray)) else [])

    return meta_df


def get_target(df):
    """
    Extracts the target variable.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.Series: Target variable.
    """
    target_col = "requester_received_pizza"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    return df[target_col].astype(int)
