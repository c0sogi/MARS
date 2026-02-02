import os
import gc
import numpy as np
import pandas as pd
from datetime import timedelta
from library.config import Config
from library.utils import Timer, print_memory_usage


class FeatureEngine:
    """
    Transforms retrieved candidates into a tabular dataset for the LightGBM ranker.
    Enriches candidates with:
    1. Retrieval Scores & Ranks
    2. Item Metadata (Category, Colour, etc.)
    3. User Metadata (Age, etc.)
    4. Dynamic Context Features (Popularity, Item Age, User Activity)
    5. Binary Labels (Purchased/Not Purchased) for training
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        self.working_dir.mkdir(parents=True, exist_ok=True)

    def generate_train_data(
        self,
        retriever,
        data_loader,
        train_df: pd.DataFrame,
        articles_df: pd.DataFrame,
        customers_df: pd.DataFrame,
        load_cached_data: bool = True,
    ):
        """
        Generates the training dataset for the ranker using a Sliding Window strategy.
        """
        if load_cached_data and Config.CACHE_RANKER_TRAIN.exists():
            print("Loading Ranker Train data from cache...")
            return pd.read_parquet(Config.CACHE_RANKER_TRAIN)

        print("Generating Ranker Train data (Sliding Windows)...")

        all_features = []

        # Iterate over sliding windows from the training set
        for window in data_loader.get_sliding_windows(train_df):
            window_id = window["window_id"]
            history_df = window["history_df"]
            target_df = window["target_df"]
            target_start = window["target_start_date"]

            print(f"Processing Window {window_id}...")

            # 1. Generate Candidates via Retrieval
            # We compute user vectors based strictly on history within this window
            user_vectors = retriever.compute_user_vectors(history_df, target_start)
            candidates_df = retriever.retrieve_candidates(user_vectors)

            if candidates_df.empty:
                print(f"Warning: No candidates retrieved for Window {window_id}.")
                continue

            # 2. Compute Features
            features_df = self._compute_features(
                candidates_df, history_df, articles_df, customers_df, target_start
            )

            # 3. Generate Labels
            # Label = 1 if (customer, article) is in target_df, else 0
            # Create a set of positive pairs for fast lookup
            # We use a composite key or merge. Merge is safer.
            target_pairs = target_df[["customer_id", "article_id"]].drop_duplicates()
            target_pairs["label"] = 1

            # Left join candidates with targets
            features_df = features_df.merge(
                target_pairs, on=["customer_id", "article_id"], how="left"
            )
            features_df["label"] = features_df["label"].fillna(0).astype("int8")

            # Downsample negatives?
            # LightGBM with LambdaRank handles class imbalance well, but we can save memory.
            # For now, we keep all retrieved candidates to ensure the model learns to rank
            # hard negatives (items retrieved by stage 1 but not bought).

            all_features.append(features_df)

            # Cleanup
            del user_vectors, candidates_df, features_df, history_df, target_df
            gc.collect()

        if not all_features:
            raise ValueError("No training data generated from sliding windows.")

        # Concatenate all windows
        full_train_df = pd.concat(all_features, ignore_index=True)

        print(f"Saving Ranker Train data: {full_train_df.shape}")
        full_train_df.to_parquet(Config.CACHE_RANKER_TRAIN, index=False)

        return full_train_df

    def generate_val_data(
        self,
        retriever,
        val_df: pd.DataFrame,
        articles_df: pd.DataFrame,
        customers_df: pd.DataFrame,
        load_cached_data: bool = True,
    ):
        """
        Generates the validation dataset using the hold-out validation users.
        We simulate a prediction for the last 7 days of the validation set.
        """
        if load_cached_data and Config.CACHE_RANKER_VAL.exists():
            print("Loading Ranker Val data from cache...")
            return pd.read_parquet(Config.CACHE_RANKER_VAL)

        print("Generating Ranker Validation data...")

        # Split Val DF into History and Target (Last 7 days)
        max_date = val_df["t_dat"].max()
        target_start = max_date - timedelta(days=Config.SLIDING_WINDOW_SIZE)

        history_df = val_df[val_df["t_dat"] < target_start].copy()
        target_df = val_df[val_df["t_dat"] >= target_start].copy()

        # 1. Retrieval
        user_vectors = retriever.compute_user_vectors(history_df, target_start)
        candidates_df = retriever.retrieve_candidates(user_vectors)

        # 2. Features
        features_df = self._compute_features(
            candidates_df, history_df, articles_df, customers_df, target_start
        )

        # 3. Labels
        target_pairs = target_df[["customer_id", "article_id"]].drop_duplicates()
        target_pairs["label"] = 1

        features_df = features_df.merge(
            target_pairs, on=["customer_id", "article_id"], how="left"
        )
        features_df["label"] = features_df["label"].fillna(0).astype("int8")

        print(f"Saving Ranker Val data: {features_df.shape}")
        features_df.to_parquet(Config.CACHE_RANKER_VAL, index=False)

        return features_df

    def generate_test_data(
        self,
        retriever,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_customer_ids: pd.DataFrame,
        articles_df: pd.DataFrame,
        customers_df: pd.DataFrame,
        load_cached_data: bool = True,
    ):
        """
        Generates the test dataset for final inference.
        Uses the full available history (Train + Val) to predict for Test customers.
        """
        if load_cached_data and Config.CACHE_RANKER_TEST.exists():
            print("Loading Ranker Test data from cache...")
            return pd.read_parquet(Config.CACHE_RANKER_TEST)

        print("Generating Ranker Test data...")

        # Combine Train and Val for full history
        full_history_df = pd.concat([train_df, val_df], ignore_index=True)
        max_date = full_history_df["t_dat"].max()
        prediction_date = max_date + timedelta(days=1)  # The 'next' day

        # 1. Retrieval
        # We need to filter user vectors to only those in test_customer_ids to save memory/time
        # But compute_user_vectors computes for all users in history.
        # We can pass the full history, then filter the resulting user_vectors or candidates.
        # Efficient approach: Compute full U, but only dot product for test users?
        # SparseRetriever.retrieve_candidates computes for all rows in U.
        # Let's filter U to keep only test customers.

        print("Computing User Vectors for Test...")
        user_vectors = retriever.compute_user_vectors(full_history_df, prediction_date)

        # Filter User Vectors for requested test customers
        # test_customer_ids is a DataFrame with 'customer_id' (int mapped)
        target_cust_indices = test_customer_ids["customer_id"].unique()

        # Create a mask or slice. Slicing CSR by rows is efficient.
        # However, our user_vectors are shape (N_total_customers, N_items).
        # We need to retrieve only for target_cust_indices.
        # If we slice, the row indices change (0..M). We need to track original IDs.

        # Better approach: The retriever calculates for everyone. We filter the RESULTING candidates.
        # This is safer for ID alignment.

        print("Retrieving Candidates...")
        candidates_df = retriever.retrieve_candidates(user_vectors)

        # Filter candidates to only include test customers
        candidates_df = candidates_df[
            candidates_df["customer_id"].isin(target_cust_indices)
        ]

        if candidates_df.empty:
            print("Warning: No candidates found for test customers.")
            # We return an empty DF with correct columns, pipeline will handle fallback
            cols = [
                "customer_id",
                "article_id",
                "score_short",
                "score_long",
                "score_vis",
                "score_hist",
            ]
            return pd.DataFrame(columns=cols)

        # 2. Features
        features_df = self._compute_features(
            candidates_df, full_history_df, articles_df, customers_df, prediction_date
        )

        # No labels for test

        print(f"Saving Ranker Test data: {features_df.shape}")
        features_df.to_parquet(Config.CACHE_RANKER_TEST, index=False)

        return features_df

    def _compute_features(
        self,
        candidates: pd.DataFrame,
        history_df: pd.DataFrame,
        articles_df: pd.DataFrame,
        customers_df: pd.DataFrame,
        ref_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Internal method to compute all features for a given set of candidates and history.
        """
        with Timer("Feature Engineering"):
            df = candidates.copy()

            # --- 1. Ranking Features ---
            # Compute rank within each score type (1 = highest score)
            # We use 'min' method to handle ties (e.g. 0 scores get same rank)
            # We only rank where score > 0 effectively, but ranking all is fine.
            score_cols = ["score_short", "score_long", "score_vis", "score_hist"]
            for col in score_cols:
                # Ascending=False because higher score is better
                df[f'rank_{col.split("_")[1]}'] = (
                    df.groupby("customer_id")[col]
                    .rank(method="min", ascending=False)
                    .astype("int16")
                )

            # --- 2. Item Metadata ---
            # Merge static article features
            # Columns to use
            art_cols = [
                "article_id",
                "product_type_no",
                "graphical_appearance_no",
                "colour_group_code",
                "perceived_colour_value_id",
                "department_no",
                "index_group_no",
                "section_no",
                "garment_group_no",
            ]
            df = df.merge(articles_df[art_cols], on="article_id", how="left")

            # --- 3. Customer Metadata ---
            cust_cols = [
                "customer_id",
                "age",
                "club_member_status",
                "fashion_news_frequency",
            ]
            # Ensure categorical columns are encoded if they aren't already (DataLoader maps IDs, but status might be object)
            # Assuming DataLoader or raw CSV has them. If object, we should label encode.
            # For this implementation, we assume they are numeric or we let LGBM handle it if passed as category.
            # We'll check types. If object, we convert to category codes.

            # Safe merge
            df = df.merge(customers_df[cust_cols], on="customer_id", how="left")

            # Handle Categorical conversions for object columns
            for col in ["club_member_status", "fashion_news_frequency"]:
                if df[col].dtype == "object":
                    df[col] = df[col].astype("category").cat.codes

            # --- 4. Dynamic Item Features (Popularity) ---
            # Popularity in last 7 days relative to ref_date
            start_7d = ref_date - timedelta(days=7)
            pop_7d = (
                history_df[history_df["t_dat"] >= start_7d]
                .groupby("article_id")
                .size()
                .reset_index(name="pop_7d")
            )

            # Popularity in last 28 days
            start_28d = ref_date - timedelta(days=28)
            pop_28d = (
                history_df[history_df["t_dat"] >= start_28d]
                .groupby("article_id")
                .size()
                .reset_index(name="pop_28d")
            )

            df = df.merge(pop_7d, on="article_id", how="left")
            df = df.merge(pop_28d, on="article_id", how="left")

            # Fill NaN popularity with 0
            df["pop_7d"] = df["pop_7d"].fillna(0).astype("int32")
            df["pop_28d"] = df["pop_28d"].fillna(0).astype("int32")

            # --- 5. Dynamic User Features (Activity) ---
            # Count total transactions per user in history
            user_activity = (
                history_df.groupby("customer_id")
                .size()
                .reset_index(name="user_activity_count")
            )
            df = df.merge(user_activity, on="customer_id", how="left")
            df["user_activity_count"] = (
                df["user_activity_count"].fillna(0).astype("int32")
            )

            # --- 6. Item Age ---
            # Days since first appearance in history
            # This is expensive to compute on full history every time.
            # We approximate using the min date in the current history slice.
            item_min_date = (
                history_df.groupby("article_id")["t_dat"]
                .min()
                .reset_index(name="first_sale_date")
            )
            df = df.merge(item_min_date, on="article_id", how="left")

            # Calculate days elapsed
            df["item_age_days"] = (ref_date - df["first_sale_date"]).dt.days
            df["item_age_days"] = df["item_age_days"].fillna(-1).astype("int16")
            df.drop(columns=["first_sale_date"], inplace=True)

            # --- Memory Optimization ---
            # Convert float64 to float32
            fcols = df.select_dtypes("float").columns
            df[fcols] = df[fcols].astype("float32")

            # Convert int64 to int32
            icols = df.select_dtypes("int64").columns
            df[icols] = df[icols].astype("int32")

            return df
