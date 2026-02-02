import pandas as pd
import numpy as np
import os
import gc
from pathlib import Path
import library.config as config
import library.data_loader as data_loader


class FeatureEngineer:
    def __init__(self):
        pass

    def generate_features(
        self,
        candidates_df,
        df_history,
        mode="train",
        df_target=None,
        load_cached_data=True,
        max_users=None,
    ):
        """
        Generates a feature matrix for the ranker by enriching candidates with
        user/item metadata and dynamic history-based features.

        Args:
            candidates_df (pd.DataFrame): DataFrame containing [customer_id, article_id, retrieval_score, rank].
            df_history (pd.DataFrame): Historical transactions used for computing dynamic features.
            mode (str): 'train' or 'test'. Determines caching path and label generation.
            df_target (pd.DataFrame): Ground truth transactions (required if mode='train').
            load_cached_data (bool): If True, attempts to load result from cache.
            max_users (int): Optional limit on the number of users to process (for debugging).

        Returns:
            pd.DataFrame: The enriched feature matrix (including 'label' column if mode='train').
        """
        # 1. Determine Cache Path
        if mode == "train":
            cache_path = config.CACHE_FEATURES_TRAIN
        else:
            cache_path = config.CACHE_FEATURES_TEST

        # 2. Check Cache
        if load_cached_data and cache_path.exists():
            print(f"Loading features from {cache_path}")
            return pd.read_parquet(cache_path)

        print(f"Generating features for mode={mode}...")

        # 3. Apply Debugging/Sampling Limits
        if max_users is not None:
            unique_users = candidates_df[config.USER_COL].unique()
            if len(unique_users) > max_users:
                print(f"Sampling {max_users} users for feature generation...")
                selected_users = unique_users[:max_users]
                candidates_df = candidates_df[
                    candidates_df[config.USER_COL].isin(selected_users)
                ].copy()

        # Create working copy
        df = candidates_df.copy()

        # Ensure correct data types for merge keys
        # article_id should be int64 to match preprocessed metadata
        df[config.ITEM_COL] = df[config.ITEM_COL].astype(np.int64)

        # 4. Load and Merge Static Metadata
        print("Merging static user and item metadata...")

        # --- Customers ---
        cust_df = data_loader.load_and_preprocess_customers(load_cached_data=True)
        # Select relevant columns: ID, numericals, and encoded categoricals
        cust_cols = [config.USER_COL, "age", "FN", "Active"] + [
            c for c in cust_df.columns if c.endswith("_idx")
        ]
        # Intersect with available columns to be safe
        cust_cols = [c for c in cust_cols if c in cust_df.columns]

        df = df.merge(cust_df[cust_cols], on=config.USER_COL, how="left")

        # --- Articles ---
        art_df = data_loader.load_and_preprocess_articles(load_cached_data=True)
        # Select relevant columns: ID and encoded categoricals
        art_cols = [config.ITEM_COL] + [c for c in art_df.columns if c.endswith("_idx")]
        art_cols = [c for c in art_cols if c in art_df.columns]

        df = df.merge(art_df[art_cols], on=config.ITEM_COL, how="left")

        # 5. Calculate Dynamic Features
        print("Calculating dynamic features...")

        # --- A. Item Popularity ---
        # Count occurrences in the provided history
        pop_counts = (
            df_history[config.ITEM_COL].value_counts().rename("item_pop_global")
        )
        df = df.merge(pop_counts, left_on=config.ITEM_COL, right_index=True, how="left")
        df["item_pop_global"] = df["item_pop_global"].fillna(0)

        # --- B. Days Since Last Purchase ---
        if not df_history.empty:
            # Determine reference date (last date in history)
            # Ensure date column is datetime
            if not np.issubdtype(df_history[config.DATE_COL].dtype, np.datetime64):
                df_history[config.DATE_COL] = pd.to_datetime(
                    df_history[config.DATE_COL]
                )

            ref_date = df_history[config.DATE_COL].max()

            # Find last purchase date for each user-item pair in history
            # Grouping by user and item to find max date
            last_dates = (
                df_history.groupby([config.USER_COL, config.ITEM_COL])[config.DATE_COL]
                .max()
                .reset_index()
            )
            last_dates.rename(
                columns={config.DATE_COL: "last_purchase_date"}, inplace=True
            )

            # Merge into candidates
            df = df.merge(last_dates, on=[config.USER_COL, config.ITEM_COL], how="left")

            # Calculate difference in days
            df["days_since_last_purchase"] = (
                ref_date - df["last_purchase_date"]
            ).dt.days

            # Fill NaNs (items never purchased by this user) with a large number
            df["days_since_last_purchase"] = df["days_since_last_purchase"].fillna(9999)

            # Clean up temporary date column
            df.drop(columns=["last_purchase_date"], inplace=True)
        else:
            # Fallback if history is empty
            df["days_since_last_purchase"] = 9999

        # 6. Generate Labels (Train Mode Only)
        if mode == "train":
            if df_target is None:
                raise ValueError("df_target must be provided when mode='train'.")

            print("Generating training labels...")
            # Identify positive pairs (User, Item) present in the target set
            positives = (
                df_target[[config.USER_COL, config.ITEM_COL]].drop_duplicates().copy()
            )
            positives["label"] = 1

            # Merge to assign labels
            df = df.merge(positives, on=[config.USER_COL, config.ITEM_COL], how="left")
            df["label"] = df["label"].fillna(0).astype(int)

            # Print stats
            pos_count = df["label"].sum()
            total_count = len(df)
            print(
                f"Label generation complete. Positives: {pos_count} ({pos_count/total_count:.5f} rate)"
            )

        # 7. Save to Cache
        print(f"Saving features to {cache_path}")
        os.makedirs(config.WORKING_DIR, exist_ok=True)
        df.to_parquet(cache_path, index=False)

        # Explicit garbage collection
        gc.collect()

        return df
