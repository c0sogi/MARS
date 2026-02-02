import os
import ast
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed


def extract_text_meta_features(df):
    """
    Extracts meta-features from the request text, such as length and capitalization ratio.

    Args:
        df (pd.DataFrame): Input DataFrame containing the text column.

    Returns:
        pd.DataFrame: DataFrame with added text meta-features.
    """
    text_col = Config.TEXT_COL

    # Ensure text column exists and fill NaNs
    if text_col not in df.columns:
        # If the edit-aware column is missing, try falling back to 'request_text' or empty
        if "request_text" in df.columns:
            texts = df["request_text"].fillna("").astype(str)
        else:
            texts = pd.Series([""] * len(df), index=df.index)
    else:
        texts = df[text_col].fillna("").astype(str)

    # Feature: Character Count
    df["text_len_char"] = texts.apply(len)

    # Feature: Word Count
    df["text_len_words"] = texts.apply(lambda x: len(x.split()))

    # Feature: Caps Ratio (Uppercase / Total Length)
    # Handle division by zero for empty strings
    def get_caps_ratio(text):
        if len(text) == 0:
            return 0.0
        return sum(1 for c in text if c.isupper()) / len(text)

    df["text_caps_ratio"] = texts.apply(get_caps_ratio)

    return df


def compute_interaction_ratios(df):
    """
    Computes interaction ratios based on user history, such as upvote ratios
    and RAOP-specific activity fractions.

    Args:
        df (pd.DataFrame): Input DataFrame with numerical history columns.

    Returns:
        pd.DataFrame: DataFrame with added ratio features.
    """
    # 1. Vote Balance Ratio
    # plus = up + down, minus = up - down
    # Ratio = (up - down) / (up + down) = minus / plus
    # Range: [-1, 1]. 0 means neutral or no votes.
    if (
        "requester_upvotes_minus_downvotes_at_request" in df.columns
        and "requester_upvotes_plus_downvotes_at_request" in df.columns
    ):

        minus = df["requester_upvotes_minus_downvotes_at_request"]
        plus = df["requester_upvotes_plus_downvotes_at_request"]

        # Avoid division by zero
        df["vote_balance_ratio"] = np.where(plus > 0, minus / plus, 0.0)

    # 2. RAOP Comment Engagement
    # comments_in_raop / total_comments
    if (
        "requester_number_of_comments_in_raop_at_request" in df.columns
        and "requester_number_of_comments_at_request" in df.columns
    ):

        raop_comments = df["requester_number_of_comments_in_raop_at_request"]
        total_comments = df["requester_number_of_comments_at_request"]

        df["raop_comment_ratio"] = np.where(
            total_comments > 0, raop_comments / total_comments, 0.0
        )

    # 3. RAOP Post Engagement
    # posts_in_raop / total_posts
    if (
        "requester_number_of_posts_on_raop_at_request" in df.columns
        and "requester_number_of_posts_at_request" in df.columns
    ):

        raop_posts = df["requester_number_of_posts_on_raop_at_request"]
        total_posts = df["requester_number_of_posts_at_request"]

        df["raop_post_ratio"] = np.where(total_posts > 0, raop_posts / total_posts, 0.0)

    return df


def process_subreddit_lists(df):
    """
    Parses subreddit history lists and extracts summary features like count.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with processed list column and count feature.
    """
    col_name = Config.SUBREDDIT_COL

    if col_name in df.columns:
        # Ensure the column contains lists (parse if string)
        def ensure_list(x):
            if isinstance(x, list):
                return x
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except (ValueError, SyntaxError):
                    return []
            return []

        # Apply parsing if necessary (though data_loader usually handles this)
        # We re-apply to be safe if this module is used independently
        df[col_name] = df[col_name].apply(ensure_list)

        # Feature: Number of unique subreddits posted in
        df["num_subreddits"] = df[col_name].apply(len)

    else:
        # Create empty placeholder if missing
        df[col_name] = [[] for _ in range(len(df))]
        df["num_subreddits"] = 0

    return df


def generate_features(train_df, val_df, test_df, load_cached_data=True):
    """
    Main function to generate tabular and metadata features.
    Handles caching to avoid re-computation.

    Args:
        train_df, val_df, test_df (pd.DataFrame): Input DataFrames.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        tuple: (train_df, val_df, test_df) with added features.
    """
    set_seed(Config.SEED)

    # Define cache paths
    train_feat_path = os.path.join(Config.WORKING_DIR, "train_features.parquet")
    val_feat_path = os.path.join(Config.WORKING_DIR, "val_features.parquet")
    test_feat_path = os.path.join(Config.WORKING_DIR, "test_features.parquet")

    # Check cache
    if load_cached_data:
        if (
            os.path.exists(train_feat_path)
            and os.path.exists(val_feat_path)
            and os.path.exists(test_feat_path)
        ):
            print("Loading engineered features from cache...")
            train_out = pd.read_parquet(train_feat_path)
            val_out = pd.read_parquet(val_feat_path)
            test_out = pd.read_parquet(test_feat_path)
            return train_out, val_out, test_out
        else:
            print("Feature cache not found. Generating features...")
    else:
        print("Ignoring feature cache. Generating features...")

    # Process each split
    datasets = [train_df, val_df, test_df]
    processed_datasets = []

    for df in datasets:
        # Create a copy to avoid modifying original if passed by reference
        df_proc = df.copy()

        # 1. Text Meta Features
        df_proc = extract_text_meta_features(df_proc)

        # 2. Interaction Ratios
        df_proc = compute_interaction_ratios(df_proc)

        # 3. Subreddit List Processing
        df_proc = process_subreddit_lists(df_proc)

        processed_datasets.append(df_proc)

    train_out, val_out, test_out = processed_datasets

    # Save to cache
    print(f"Saving engineered features to {Config.WORKING_DIR}...")
    train_out.to_parquet(train_feat_path, index=False)
    val_out.to_parquet(val_feat_path, index=False)
    test_out.to_parquet(test_feat_path, index=False)

    return train_out, val_out, test_out
