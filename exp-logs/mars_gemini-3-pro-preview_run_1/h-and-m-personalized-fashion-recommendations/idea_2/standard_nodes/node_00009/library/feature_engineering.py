import os
import gc
import pandas as pd
import numpy as np
from library.utils import reduce_mem_usage, set_seed


class FeatureFactory:
    def __init__(
        self,
        transactions_df,
        articles_df,
        customers_df,
        cache_dir="./working/idea_2",
    ):
        """
        Initialize the FeatureFactory.

        Args:
            transactions_df (pd.DataFrame): Transactions with 'week' column.
            articles_df (pd.DataFrame): Articles metadata.
            customers_df (pd.DataFrame): Customers metadata.
            cache_dir (str): Directory to store cached feature files.
        """
        self.transactions = transactions_df
        self.articles = articles_df
        self.customers = customers_df
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        set_seed(42)

        # Pre-process customers categorical columns to int
        self._process_customer_features()

    def _process_customer_features(self):
        """
        Encodes categorical features in customers dataframe.
        """
        cat_cols = ["club_member_status", "fashion_news_frequency", "postal_code"]
        for col in cat_cols:
            if col in self.customers.columns:
                self.customers[col] = pd.factorize(self.customers[col])[0].astype(
                    np.int32
                )

    def create_features(self, candidates_df, target_week, load_cached_data=True):
        """
        Generates features for the candidate set based on history relative to target_week.

        Args:
            candidates_df (pd.DataFrame): DataFrame with [customer_id, article_id, scores...]
            target_week (int): The week index to predict (0=Validation, 1=Train, -1=Test).
            load_cached_data (bool): Whether to load from cache if available.

        Returns:
            pd.DataFrame: Enriched dataframe with features and labels (if applicable).
        """
        cache_file = os.path.join(
            self.cache_dir, f"features_week_{target_week}.parquet"
        )

        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading features from cache: {cache_file}")
            return pd.read_parquet(cache_file)

        print(f"Computing features for target week {target_week}...")

        # 1. Define History and Reference Date
        # If target_week is -1 (Test), we use all available data (week >= 0) as history
        # If target_week is 0 (Val), we use week > 0 as history
        if target_week == -1:
            history_mask = self.transactions["week"] >= 0
        else:
            history_mask = self.transactions["week"] > target_week

        history_df = self.transactions[history_mask].copy()

        if history_df.empty:
            raise ValueError(f"No history available for target week {target_week}")

        # Reference date for recency calculation is the max date in history
        ref_date = history_df["t_dat"].max()

        # 2. User Stats Features
        print("Computing User Stats...")
        user_stats = (
            history_df.groupby("customer_id")
            .agg(
                user_purchase_count=("article_id", "count"),
                user_mean_price=("price", "mean"),
            )
            .reset_index()
        )
        # Optimize types
        user_stats["user_purchase_count"] = user_stats["user_purchase_count"].astype(
            np.int16
        )
        user_stats["user_mean_price"] = user_stats["user_mean_price"].astype(np.float32)

        # 3. Item Recency Features
        print("Computing Item Recency...")
        # Last purchase date of specific item by user
        last_purchase = (
            history_df.groupby(["customer_id", "article_id"])["t_dat"]
            .max()
            .reset_index()
        )
        last_purchase["days_since_last_item"] = (
            ref_date - last_purchase["t_dat"]
        ).dt.days
        last_purchase = last_purchase.drop(columns=["t_dat"])

        # 4. Category Recency Features
        print("Computing Category Recency...")
        # We use index_group_no as a high-level category proxy
        # Merge article info to history to get category
        history_cat = history_df[["customer_id", "article_id", "t_dat"]].merge(
            self.articles[["article_id", "index_group_no"]],
            on="article_id",
            how="left",
        )

        last_cat_purchase = (
            history_cat.groupby(["customer_id", "index_group_no"])["t_dat"]
            .max()
            .reset_index()
        )
        last_cat_purchase["days_since_last_cat"] = (
            ref_date - last_cat_purchase["t_dat"]
        ).dt.days
        last_cat_purchase = last_cat_purchase.drop(columns=["t_dat"])

        # 5. Merge Features onto Candidates
        print("Merging features...")
        df = candidates_df.copy()

        # Merge User Stats
        df = df.merge(user_stats, on="customer_id", how="left")
        # Fill NaNs for users with no history (cold start in test)
        df["user_purchase_count"] = df["user_purchase_count"].fillna(0)
        df["user_mean_price"] = df["user_mean_price"].fillna(0)

        # Merge Item Recency
        df = df.merge(last_purchase, on=["customer_id", "article_id"], how="left")
        df["days_since_last_item"] = df["days_since_last_item"].fillna(999)

        # Merge Article Metadata to get index_group_no for Category Recency
        # Select relevant article columns
        art_cols = [
            "article_id",
            "product_type_no",
            "graphical_appearance_no",
            "colour_group_code",
            "perceived_colour_value_id",
            "perceived_colour_master_id",
            "department_no",
            "index_group_no",
            "section_no",
            "garment_group_no",
        ]
        df = df.merge(self.articles[art_cols], on="article_id", how="left")

        # Merge Category Recency
        df = df.merge(
            last_cat_purchase, on=["customer_id", "index_group_no"], how="left"
        )
        df["days_since_last_cat"] = df["days_since_last_cat"].fillna(999)

        # Merge Customer Metadata
        cust_cols = [
            "customer_id",
            "age",
            "club_member_status",
            "fashion_news_frequency",
        ]
        df = df.merge(self.customers[cust_cols], on="customer_id", how="left")

        # Fill NaNs in customer data
        df["age"] = df["age"].fillna(df["age"].mean())
        df["club_member_status"] = df["club_member_status"].fillna(-1)
        df["fashion_news_frequency"] = df["fashion_news_frequency"].fillna(-1)

        # 6. Generate Labels (if training/validation)
        if target_week >= 0:
            print("Generating labels...")
            target_transactions = self.transactions[
                self.transactions["week"] == target_week
            ]

            # Create a set of purchased pairs
            # Using a set of tuples is faster than merge for just existence check
            purchased_pairs = set(
                zip(
                    target_transactions["customer_id"],
                    target_transactions["article_id"],
                )
            )

            # Apply label
            # We iterate or use map. Map with a set is efficient.
            # Construct tuple series
            pairs = list(zip(df["customer_id"], df["article_id"]))
            df["label"] = [1 if p in purchased_pairs else 0 for p in pairs]
            df["label"] = df["label"].astype(np.int8)
        else:
            # Test set, no labels
            df["label"] = np.nan

        # 7. Cleanup
        print("Optimizing memory...")
        df = reduce_mem_usage(df, verbose=False)

        # Save to cache
        print(f"Saving features to {cache_file}")
        df.to_parquet(cache_file, index=False)

        # Explicit GC
        del history_df, user_stats, last_purchase, last_cat_purchase
        gc.collect()

        return df

    def get_ranking_data(self, feature_df, features_columns=None):
        """
        Prepares the data for LightGBM Ranker.
        Sorts by customer_id to ensure groups are contiguous.

        Args:
            feature_df (pd.DataFrame): The dataframe with features and labels.
            features_columns (list): List of column names to use as features.
                                     If None, uses all except IDs and label.

        Returns:
            tuple: (X, y, group)
        """
        # Sort by customer_id for LightGBM grouping
        print("Sorting data for ranking...")
        df = feature_df.sort_values("customer_id").reset_index(drop=True)

        # Define feature columns if not provided
        if features_columns is None:
            exclude_cols = ["customer_id", "article_id", "label"]
            features_columns = [c for c in df.columns if c not in exclude_cols]

        X = df[features_columns]

        if "label" in df.columns and not df["label"].isna().all():
            y = df["label"].to_numpy()
        else:
            y = None

        # Calculate groups (number of rows per customer)
        # Since we sorted by customer_id, we can just count occurrences
        group = df.groupby("customer_id", sort=False).size().to_numpy()

        return X, y, group
