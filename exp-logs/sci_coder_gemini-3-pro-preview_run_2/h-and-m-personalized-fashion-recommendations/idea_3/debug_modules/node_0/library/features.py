import pandas as pd
import numpy as np
import os
import gc
from library.config import Config
from library.utils import Timer, reduce_mem_usage


class UserProfiler:
    """
    Computes user preference vectors for metadata attributes based on purchase history.
    Used to generate Interaction Features (Affinity) for the Ranker.
    """

    def __init__(self, history_df, articles_df):
        """
        Args:
            history_df (pd.DataFrame): Transaction history with customer_id_idx and article_id_idx.
            articles_df (pd.DataFrame): Article metadata with article_id_idx and attribute columns.
        """
        self.history_df = history_df
        self.articles_df = articles_df
        self.profiles = {}  # Cache for profiles {col_name: dataframe}

    def get_profile(self, col_name):
        """
        Computes or retrieves the user profile for a specific attribute column.
        Returns a DataFrame: [customer_id_idx, {col_name}, {col_name}_affinity]
        """
        if col_name in self.profiles:
            return self.profiles[col_name]

        with Timer(f"Profiling User Affinity: {col_name}"):
            # Merge history with article attribute
            # We only need customer and the specific attribute
            # Use inner join to drop transactions for articles not in articles_df
            merged = self.history_df[["customer_id_idx", "article_id_idx"]].merge(
                self.articles_df[["article_id_idx", col_name]],
                on="article_id_idx",
                how="inner",
            )

            # Count occurrences of (user, attribute_value)
            user_attr_counts = (
                merged.groupby(["customer_id_idx", col_name])
                .size()
                .reset_index(name="count")
            )

            # Calculate total counts per user for normalization
            user_total_counts = (
                merged.groupby("customer_id_idx").size().reset_index(name="total")
            )

            # Merge to calculate affinity (probability)
            profile = user_attr_counts.merge(user_total_counts, on="customer_id_idx")
            profile[f"{col_name}_affinity"] = profile["count"] / profile["total"]

            # Keep only relevant columns
            profile = profile[["customer_id_idx", col_name, f"{col_name}_affinity"]]
            profile = reduce_mem_usage(profile, verbose=False)

            self.profiles[col_name] = profile

            del merged, user_attr_counts, user_total_counts
            gc.collect()

            return profile


class FeatureEngineer:
    """
    Stage 2: Feature Engineering.
    Enriches candidate pairs with:
    1. Behavioral Features (from Retrieval stage)
    2. User Features (Age, Club Status)
    3. Item Features (Product Type, etc.)
    4. Interaction Features (User-Item Affinity)
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        self.affinity_cols = Config.AFFINITY_COLS

    def generate_features(
        self,
        candidates_df,
        train_df,
        articles_df,
        customers_df,
        mode="train",
        labeled_data=None,
        load_cached_data=True,
    ):
        """
        Main method to generate the feature matrix for Ranking.

        Args:
            candidates_df (pd.DataFrame): [customer_id_idx, article_id_idx, scores...]
            train_df (pd.DataFrame): Historical transactions for User Profiling.
            articles_df (pd.DataFrame): Article metadata.
            customers_df (pd.DataFrame): Customer metadata.
            mode (str): 'train' or 'test'.
            labeled_data (pd.DataFrame, optional): Validation transactions for creating labels (only in train mode).
            load_cached_data (bool): Whether to use disk caching.

        Returns:
            pd.DataFrame: Enriched dataframe ready for LightGBM.
        """
        cache_file = self.working_dir / f"features_{mode}.parquet"

        if load_cached_data and cache_file.exists():
            print(f"[FeatureEngineer] Loading cached features from {cache_file}")
            return pd.read_parquet(cache_file)

        print(f"[FeatureEngineer] Generating features for {mode}...")

        # Initialize Profiler with the provided history
        # Note: In 'train' mode, train_df should exclude the validation period (handled by caller)
        profiler = UserProfiler(train_df, articles_df)

        # 1. Base Features (Candidates)
        df = candidates_df.copy()

        # 2. Item Features
        # We need affinity cols for interaction features, plus others for direct features
        # We merge affinity cols + product_type_no + perceived_colour_value_id
        item_cols = list(
            set(
                self.affinity_cols
                + ["product_type_no", "perceived_colour_value_id", "department_no"]
            )
        )
        # Ensure we don't duplicate if they overlap and exist in articles
        item_cols = [c for c in item_cols if c in articles_df.columns]

        with Timer("Merge Item Features"):
            df = df.merge(
                articles_df[["article_id_idx"] + item_cols],
                on="article_id_idx",
                how="left",
            )

        # 3. User Features
        # Age, club_member_status
        cust_cols = ["customer_id_idx", "age", "club_member_status"]
        cust_cols = [c for c in cust_cols if c in customers_df.columns]

        with Timer("Merge User Features"):
            cust_df_clean = customers_df[cust_cols].copy()
            # Simple encoding for club_member_status if it's object/string
            if (
                "club_member_status" in cust_df_clean.columns
                and cust_df_clean["club_member_status"].dtype == "object"
            ):
                cust_df_clean["club_member_status"] = (
                    cust_df_clean["club_member_status"].astype("category").cat.codes
                )

            df = df.merge(cust_df_clean, on="customer_id_idx", how="left")

        # 4. Interaction Features (Affinity)
        for col in self.affinity_cols:
            # Get profile [customer_id_idx, col, score]
            profile_df = profiler.get_profile(col)

            with Timer(f"Merge Interaction: {col}"):
                # Merge on [customer, col_value]
                # The candidate item has 'col_value' (merged in step 2)
                # The profile has 'col_value'
                df = df.merge(profile_df, on=["customer_id_idx", col], how="left")

                # Fill NaNs: If user has no history for this attribute value, affinity is 0
                df[f"{col}_affinity"] = df[f"{col}_affinity"].fillna(0.0)

        # 5. Label Generation (Train Mode Only)
        if mode == "train" and labeled_data is not None:
            with Timer("Generating Labels"):
                # labeled_data is val_df [customer_id_idx, article_id_idx, ...]
                # Create a set of positive pairs
                positives = labeled_data[
                    ["customer_id_idx", "article_id_idx"]
                ].drop_duplicates()
                positives["label"] = 1

                df = df.merge(
                    positives, on=["customer_id_idx", "article_id_idx"], how="left"
                )
                df["label"] = df["label"].fillna(0).astype(np.int8)

        # 6. Cleanup & Save
        with Timer("Final Cleanup & Caching"):
            df = reduce_mem_usage(df)

            os.makedirs(self.working_dir, exist_ok=True)
            df.to_parquet(cache_file, index=False)

        return df
