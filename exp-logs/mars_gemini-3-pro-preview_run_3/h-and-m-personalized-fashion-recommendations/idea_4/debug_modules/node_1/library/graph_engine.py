import os
import numpy as np
import pandas as pd
from scipy import sparse
from library.config import Config
from library.data_utils import load_metadata, load_articles, load_customers


class BehavioralGraphBuilder:
    """
    Constructs behavioral graphs and user history representations based on
    transaction data. Implements time-decayed weighting for both transitions
    and user profiles.
    """

    def __init__(self):
        self.decay_rate = Config.TIME_DECAY_RATE
        self.working_dir = Config.WORKING_DIR
        self.user_history_path = self.working_dir / "user_history.npz"

    def _calculate_time_weights(
        self, dates: pd.Series, reference_date: pd.Timestamp
    ) -> np.ndarray:
        """
        Calculates exponential time decay weights.
        Weight = exp(-decay_rate * days_diff)
        """
        # Ensure dates are datetime
        if not np.issubdtype(dates.dtype, np.datetime64):
            dates = pd.to_datetime(dates)

        days_diff = (reference_date - dates).dt.days
        # Clip negative days (if any data leak) to 0
        days_diff = days_diff.clip(lower=0)

        weights = np.exp(-self.decay_rate * days_diff)
        return weights.values.astype(np.float32)

    def build_transition_matrix(
        self, load_cached_data: bool = True
    ) -> sparse.csr_matrix:
        """
        Builds a time-decayed item-to-item transition matrix.
        Entry (i, j) represents the strength of transition from item i to item j.
        """
        os.makedirs(self.working_dir, exist_ok=True)

        if load_cached_data and Config.TRANSITION_MATRIX_PATH.exists():
            print(
                f"Loading cached Transition Matrix from {Config.TRANSITION_MATRIX_PATH}"
            )
            return sparse.load_npz(Config.TRANSITION_MATRIX_PATH)

        print("Building Transition Matrix...")

        # 1. Load Data
        train_df = load_metadata("train")
        articles_df, article_map = load_articles(load_cached_data=True)

        # 2. Preprocess
        train_df["t_dat"] = pd.to_datetime(train_df["t_dat"])
        train_df["article_idx"] = train_df["article_id"].map(article_map)

        # Drop transactions with unknown articles (if any)
        train_df = train_df.dropna(subset=["article_idx"])
        train_df["article_idx"] = train_df["article_idx"].astype(np.int32)

        # 3. Sort for sequential processing
        train_df = train_df.sort_values(["customer_id", "t_dat"])

        # 4. Identify Transitions
        # Shift to get previous item
        train_df["prev_article_idx"] = train_df["article_idx"].shift(1)
        train_df["prev_customer"] = train_df["customer_id"].shift(1)

        # Filter valid transitions (same customer)
        # We only keep rows where current customer == prev customer
        mask = train_df["customer_id"] == train_df["prev_customer"]
        transitions = train_df[mask].copy()

        # 5. Calculate Weights
        # We use the date of the *target* purchase (B in A->B) for recency
        max_date = train_df["t_dat"].max()
        weights = self._calculate_time_weights(transitions["t_dat"], max_date)

        # 6. Build Sparse Matrix
        # Rows: Source (prev_article), Cols: Target (article)
        n_articles = len(article_map)

        row_ind = transitions["prev_article_idx"].values.astype(np.int32)
        col_ind = transitions["article_idx"].values.astype(np.int32)

        # Aggregate duplicate transitions by summing weights
        transition_matrix = sparse.csr_matrix(
            (weights, (row_ind, col_ind)),
            shape=(n_articles, n_articles),
            dtype=np.float32,
        )

        # 7. Save
        print(f"Saving Transition Matrix to {Config.TRANSITION_MATRIX_PATH}")
        sparse.save_npz(Config.TRANSITION_MATRIX_PATH, transition_matrix)

        return transition_matrix

    def build_user_history(self, load_cached_data: bool = True) -> sparse.csr_matrix:
        """
        Builds a sparse user history matrix where rows are users and columns are items.
        Values are time-decayed weights indicating user preference/recency.
        """
        os.makedirs(self.working_dir, exist_ok=True)

        if load_cached_data and self.user_history_path.exists():
            print(f"Loading cached User History from {self.user_history_path}")
            return sparse.load_npz(self.user_history_path)

        print("Building User History Matrix...")

        # 1. Load Data
        train_df = load_metadata("train")
        articles_df, article_map = load_articles(load_cached_data=True)
        customers_df, customer_map = load_customers(load_cached_data=True)

        # 2. Map Indices
        train_df["t_dat"] = pd.to_datetime(train_df["t_dat"])

        # Map Articles
        train_df["article_idx"] = train_df["article_id"].map(article_map)

        # Map Customers
        # Note: train_df["customer_id"] are hashes. customer_map keys are hashes.
        train_df["customer_idx"] = train_df["customer_id"].map(customer_map)

        # Drop invalid mappings
        train_df = train_df.dropna(subset=["article_idx", "customer_idx"])

        # 3. Calculate Weights
        max_date = train_df["t_dat"].max()
        weights = self._calculate_time_weights(train_df["t_dat"], max_date)

        # 4. Build Sparse Matrix
        n_customers = len(customer_map)
        n_articles = len(article_map)

        row_ind = train_df["customer_idx"].values.astype(np.int32)
        col_ind = train_df["article_idx"].values.astype(np.int32)

        user_history = sparse.csr_matrix(
            (weights, (row_ind, col_ind)),
            shape=(n_customers, n_articles),
            dtype=np.float32,
        )

        # 5. Save
        print(f"Saving User History to {self.user_history_path}")
        sparse.save_npz(self.user_history_path, user_history)

        return user_history

    def build_global_popularity(self, load_cached_data: bool = True) -> pd.DataFrame:
        """
        Calculates global item popularity based on time-decayed purchase counts.
        Saves as a Parquet file for the ranker.
        """
        os.makedirs(self.working_dir, exist_ok=True)

        if load_cached_data and Config.GLOBAL_POPULARITY_PATH.exists():
            print(
                f"Loading cached Global Popularity from {Config.GLOBAL_POPULARITY_PATH}"
            )
            return pd.read_parquet(Config.GLOBAL_POPULARITY_PATH)

        print("Building Global Popularity...")

        # Reuse logic from user history to get weighted interactions
        # If user history is not cached, this will compute it.
        # If it is, it loads fast.
        user_history = self.build_user_history(load_cached_data=load_cached_data)

        # Sum columns (items) to get total weighted popularity
        # axis=0 sums over users
        popularity_scores = np.array(user_history.sum(axis=0)).flatten()

        # Load article map to get IDs back
        articles_df, article_map = load_articles(load_cached_data=True)

        # Create DataFrame
        # article_map is {id: idx}. We need {idx: id}
        idx_to_article = {v: k for k, v in article_map.items()}

        # Ensure we cover all articles from 0 to n_articles-1
        n_articles = len(article_map)
        article_ids = [idx_to_article[i] for i in range(n_articles)]

        pop_df = pd.DataFrame(
            {
                "article_id": article_ids,
                "article_idx": range(n_articles),
                "global_popularity": popularity_scores,
            }
        )

        # Normalize to 0-1 range for stability in ranker
        max_pop = pop_df["global_popularity"].max()
        if max_pop > 0:
            pop_df["global_popularity"] /= max_pop

        print(f"Saving Global Popularity to {Config.GLOBAL_POPULARITY_PATH}")
        pop_df.to_parquet(Config.GLOBAL_POPULARITY_PATH, index=False)

        return pop_df
