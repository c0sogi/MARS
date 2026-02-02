import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import gc
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from library import config, data_manager, visual_module


class FeatureGenerator:
    def __init__(self):
        # Load ID mappings
        self.cust_to_idx, self.idx_to_cust, self.art_to_idx, self.idx_to_art = (
            data_manager.get_id_mappings()
        )

        # Load and normalize embeddings for cosine similarity
        self.embeddings = visual_module.extract_embeddings(load_cached_data=True)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings_norm = self.embeddings / norms

        # Load static metadata
        self.articles_df = pd.read_csv(config.ARTICLES_CSV)
        self.customers_df = pd.read_csv(config.CUSTOMERS_CSV)

        # Preprocess static features
        self._preprocess_static_data()

    def _preprocess_static_data(self):
        """Preprocesses customer and article data (encoding, filling NaNs)."""
        # --- Customers ---
        # Fill Age
        self.customers_df["age"] = self.customers_df["age"].fillna(
            self.customers_df["age"].median()
        )

        # Encode Categoricals
        cat_cols_cust = ["club_member_status", "fashion_news_frequency"]
        for col in cat_cols_cust:
            self.customers_df[col] = pd.factorize(self.customers_df[col])[0]

        # Select relevant columns
        self.customers_df = self.customers_df[
            ["customer_id", "age", "club_member_status", "fashion_news_frequency"]
        ]

        # --- Articles ---
        # Select relevant columns
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
        self.articles_df = self.articles_df[art_cols].copy()

        # Encode Categoricals (most are already numeric IDs, but ensuring consistency)
        for col in art_cols:
            if col != "article_id":
                self.articles_df[col] = pd.factorize(self.articles_df[col])[0]

    def generate_features(
        self,
        candidates_df,
        history_df,
        target_df=None,
        cache_key="default",
        load_cached_data=True,
    ):
        """
        Generates ranking features for the provided candidates.

        Args:
            candidates_df (pd.DataFrame): User-Item pairs from retrieval.
            history_df (pd.DataFrame): Transaction history used for context (popularity, visual profile).
            target_df (pd.DataFrame, optional): Transactions in the target window (for labels).
            cache_key (str): Identifier for caching.
            load_cached_data (bool): Whether to load from disk.

        Returns:
            pd.DataFrame: Augmented DataFrame with features (and labels if target_df provided).
        """
        os.makedirs(config.WORKING_DIR, exist_ok=True)
        cache_path = config.WORKING_DIR / f"features_{cache_key}.parquet"

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading features from {cache_path}...")
            return pd.read_parquet(cache_path)

        print(
            f"Generating features for {len(candidates_df)} candidates (Key: {cache_key})..."
        )

        df = candidates_df.copy()

        # 1. Retrieval Rank Features
        # Rank scores within user groups to give the model relative ordering context
        df["rank_seq"] = df.groupby("customer_id")["score_seq"].rank(
            method="first", ascending=False
        )
        df["rank_vis"] = df.groupby("customer_id")["score_vis"].rank(
            method="first", ascending=False
        )
        df["rank_hist"] = df.groupby("customer_id")["score_hist"].rank(
            method="first", ascending=False
        )

        # 2. Global Popularity (Log-normalized)
        # Based on the provided history window
        pop_counts = history_df["article_id"].value_counts().reset_index()
        pop_counts.columns = ["article_id", "global_pop"]
        pop_counts["global_pop"] = np.log1p(pop_counts["global_pop"])

        df = df.merge(pop_counts, on="article_id", how="left")
        df["global_pop"] = df["global_pop"].fillna(0)

        # 3. Visual Consistency
        # Cosine similarity between candidate item and user's history centroid
        print("Computing visual consistency...")
        df = self._add_visual_consistency(df, history_df)

        # 4. Static Features
        df = df.merge(self.customers_df, on="customer_id", how="left")
        df = df.merge(self.articles_df, on="article_id", how="left")

        # 5. Generate Labels (if Training)
        if target_df is not None:
            print("Generating labels...")
            # Create a set of positive pairs
            target_df = target_df[["customer_id", "article_id"]].copy()
            target_df["label"] = 1
            # Drop duplicates in target to avoid exploding joins
            target_df = target_df.drop_duplicates()

            df = df.merge(target_df, on=["customer_id", "article_id"], how="left")
            df["label"] = df["label"].fillna(0).astype(int)

        # Save to cache
        df.to_parquet(cache_path, index=False)
        print(f"Features saved to {cache_path}")

        return df

    def _add_visual_consistency(self, candidates_df, history_df):
        """
        Computes the cosine similarity between candidate items and the centroid
        of the user's recent history. Optimized using matrix multiplication.
        """
        # Filter history to only relevant users to save memory
        relevant_users = candidates_df["customer_id"].unique()
        hist_subset = history_df[history_df["customer_id"].isin(relevant_users)].copy()

        if len(hist_subset) == 0:
            candidates_df["visual_consistency"] = 0.0
            return candidates_df

        # Map IDs to matrix indices
        hist_subset["user_idx"] = hist_subset["customer_id"].map(self.cust_to_idx)
        hist_subset["art_idx"] = hist_subset["article_id"].map(self.art_to_idx)
        hist_subset = hist_subset.dropna()

        # Build User-Item History Matrix (Binary Preference)
        # Shape: (N_users, N_items)
        rows = hist_subset["user_idx"].values.astype(int)
        cols = hist_subset["art_idx"].values.astype(int)
        data = np.ones(len(rows), dtype=np.float32)

        n_users = len(self.cust_to_idx)
        n_items = len(self.art_to_idx)

        # Sum embeddings of purchased items
        U_item_sparse = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

        # User Centroids: (N_users, 512) = (N_users, N_items) @ (N_items, 512)
        user_centroids = U_item_sparse.dot(self.embeddings_norm)

        # Normalize centroids to unit vectors
        centroid_norms = np.linalg.norm(user_centroids, axis=1, keepdims=True)
        centroid_norms[centroid_norms == 0] = 1.0
        user_centroids_norm = user_centroids / centroid_norms

        # Compute Dot Product for Candidates
        # We need to map candidates to the same indices
        temp_df = candidates_df[["customer_id", "article_id"]].copy()
        temp_df["user_idx"] = temp_df["customer_id"].map(self.cust_to_idx)
        temp_df["art_idx"] = temp_df["article_id"].map(self.art_to_idx)

        # Extract vectors for aligned rows
        u_indices = temp_df["user_idx"].values.astype(int)
        i_indices = temp_df["art_idx"].values.astype(int)

        # Gather vectors
        u_vecs = user_centroids_norm[u_indices]
        i_vecs = self.embeddings_norm[i_indices]

        # Dot product (row-wise sum of element-wise product)
        # Result is cosine similarity because both inputs are normalized
        consistency_scores = np.sum(u_vecs * i_vecs, axis=1)

        candidates_df["visual_consistency"] = consistency_scores
        return candidates_df


def train_ranker(train_df, val_df, params):
    """
    Trains a LightGBM ranker using the lambdarank objective.

    Args:
        train_df (pd.DataFrame): Training data with features and 'label'.
        val_df (pd.DataFrame): Validation data with features and 'label'.
        params (dict): LightGBM parameters.

    Returns:
        lgb.Booster: Trained model.
    """
    print("Preparing LightGBM datasets...")

    # Sort by customer_id for group creation
    train_df = train_df.sort_values("customer_id")
    val_df = val_df.sort_values("customer_id")

    # Define features (exclude ID columns and label)
    ignore_cols = ["customer_id", "article_id", "label", "t_dat"]
    features = [c for c in train_df.columns if c not in ignore_cols]

    print(f"Training with {len(features)} features: {features}")

    X_train = train_df[features]
    y_train = train_df["label"]
    # Create query groups (number of items per user)
    q_train = train_df.groupby("customer_id").size().values

    X_val = val_df[features]
    y_val = val_df["label"]
    q_val = val_df.groupby("customer_id").size().values

    # Create LGBM Datasets
    dtrain = lgb.Dataset(X_train, label=y_train, group=q_train)
    dval = lgb.Dataset(X_val, label=y_val, group=q_val, reference=dtrain)

    # Train
    print("Starting training...")
    callbacks = [
        lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS),
        lgb.log_evaluation(period=config.VERBOSE_EVAL),
    ]

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dval],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    # Save model text dump
    model.save_model(config.LGBM_MODEL_PATH)

    # Feature Importance
    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": model.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance", ascending=False)

    print("\nTop 10 Features:")
    print(importance.head(10))

    return model


def generate_submission(model, test_df, sample_submission_df):
    """
    Generates the final submission CSV.

    Args:
        model (lgb.Booster): Trained ranker.
        test_df (pd.DataFrame): Test candidates with features.
        sample_submission_df (pd.DataFrame): Template for submission.
    """
    print("Scoring test candidates...")

    # Prepare features
    ignore_cols = ["customer_id", "article_id", "label", "t_dat", "prediction", "score"]
    features = [c for c in test_df.columns if c not in ignore_cols]

    # Predict
    test_df["score"] = model.predict(test_df[features])

    print("Selecting Top-12 predictions...")

    # Sort by score descending
    test_df = test_df.sort_values(["customer_id", "score"], ascending=[True, False])

    # Group and collect top 12 article IDs
    # Convert article_id to string with leading zeros
    test_df["article_id_str"] = "0" + test_df["article_id"].astype(str)

    preds = (
        test_df.groupby("customer_id")["article_id_str"]
        .apply(lambda x: " ".join(x.head(12)))
        .reset_index()
    )
    preds.columns = ["customer_id", "prediction"]

    # Merge with sample submission to ensure all customers are present
    print("Formatting submission...")
    submission = sample_submission_df[["customer_id"]].merge(
        preds, on="customer_id", how="left"
    )

    # Fill missing predictions (if any) with empty string or fallback
    # Note: In a real scenario, we would fallback to global popularity.
    # Here, we assume retrieval covered active users.
    missing_count = submission["prediction"].isna().sum()
    if missing_count > 0:
        print(
            f"Warning: {missing_count} users have no predictions. Filling with empty strings."
        )
        submission["prediction"] = submission["prediction"].fillna("")

    # Save
    submission.to_csv(config.SUBMISSION_CSV, index=False)
    print(f"Submission saved to {config.SUBMISSION_CSV}")
