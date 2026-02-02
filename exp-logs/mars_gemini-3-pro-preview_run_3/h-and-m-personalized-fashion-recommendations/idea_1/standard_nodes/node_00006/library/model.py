import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from pathlib import Path
from library import config


class TimeAwareTransitionGraph:
    def __init__(self):
        self.global_popularity = []  # List of article_ids sorted by popularity
        self.transition_matrix = None  # scipy.sparse.csr_matrix
        self.idx_to_article = np.array([])  # Array mapping index -> article_id
        self.article_to_idx = {}  # Dict mapping article_id -> index

    def fit(self, transactions_df, load_cached_data=True):
        """
        Builds the transition graph and global popularity list.

        Args:
            transactions_df (pd.DataFrame): DataFrame with columns 'customer_id', 'article_id', 't_dat', 'weight'.
            load_cached_data (bool): Whether to attempt loading from cache.
        """
        # Define cache paths
        cache_dir = config.WORKING_DIR
        matrix_path = cache_dir / "transition_matrix.npz"
        pop_path = cache_dir / "global_popularity.parquet"
        idx_path = cache_dir / "article_index.npy"

        # 1. Try Loading from Cache
        if (
            load_cached_data
            and matrix_path.exists()
            and pop_path.exists()
            and idx_path.exists()
        ):
            print("Loading model components from cache...")
            self.transition_matrix = sp.load_npz(matrix_path)
            self.idx_to_article = np.load(idx_path, allow_pickle=True)
            # Reconstruct dict map
            # Using a simple dict comprehension is reasonably fast for 100k items
            self.article_to_idx = {art: i for i, art in enumerate(self.idx_to_article)}

            pop_df = pd.read_parquet(pop_path)
            self.global_popularity = pop_df["article_id"].tolist()
            return

        print("Fitting model from scratch...")
        os.makedirs(cache_dir, exist_ok=True)

        # 2. Compute Global Popularity
        # Sum weights per article to get general trends
        print("Computing global popularity...")
        pop_series = transactions_df.groupby("article_id")["weight"].sum()
        pop_df = pop_series.sort_values(ascending=False).reset_index()
        self.global_popularity = pop_df["article_id"].tolist()

        # Save Popularity
        pop_df.to_parquet(pop_path, index=False)

        # 3. Build Transition Matrix
        print("Building transition matrix...")
        # Ensure data is sorted by customer and time
        df = transactions_df.sort_values(["customer_id", "t_dat"])

        # Create Integer Mapping for Articles
        # Using pandas Categorical is efficient for this
        df["article_id"] = df["article_id"].astype("category")
        self.idx_to_article = df["article_id"].cat.categories.to_numpy()
        self.article_to_idx = {art: i for i, art in enumerate(self.idx_to_article)}

        # Save Index Mapping
        np.save(idx_path, self.idx_to_article)

        # Extract codes and weights
        codes = df["article_id"].cat.codes.values
        customer_ids = df["customer_id"].values
        weights = df["weight"].values

        # Identify transitions: (item_t -> item_t+1)
        # We only consider transitions within the same customer history
        # Create a mask where the next row belongs to the same customer
        mask = customer_ids[:-1] == customer_ids[1:]

        # Source nodes: item at t
        sources = codes[:-1][mask]
        # Target nodes: item at t+1
        targets = codes[1:][mask]
        # Weights: weight of item at t+1 (favoring recent target items)
        edge_weights = weights[1:][mask]

        # Construct Sparse Matrix
        # shape is (N, N) where N is number of unique articles
        n_articles = len(self.idx_to_article)

        # coo_matrix is fast to construct. Converting to csr sums duplicates efficiently.
        print("Constructing sparse matrix...")
        self.transition_matrix = sp.coo_matrix(
            (edge_weights, (sources, targets)), shape=(n_articles, n_articles)
        ).tocsr()

        # Normalize Matrix (L1 normalization per row)
        # This converts counts to probabilities: P(j|i)
        print("Normalizing transition matrix...")
        self.transition_matrix = normalize(self.transition_matrix, norm="l1", axis=1)

        # Save Matrix
        print(f"Saving transition matrix to {matrix_path}...")
        sp.save_npz(matrix_path, self.transition_matrix)

    def generate_predictions(self, user_history_df):
        """
        Generates predictions using sparse matrix multiplication.
        Scores = (User_History @ Transition_Matrix) + (Repurchase_Weight * User_History)

        Args:
            user_history_df (pd.DataFrame): DataFrame with 'customer_id', 'article_id', 'weight'.
                                            Can contain multiple rows per customer.

        Returns:
            pd.DataFrame: DataFrame with 'customer_id' and 'prediction' columns.
        """
        print("Generating predictions (Vectorized)...")

        # 1. Prepare User-Item Matrix (U)
        # Unique customers in the input order (for final mapping)
        unique_customers = user_history_df["customer_id"].unique()
        cust_to_idx = {cust: i for i, cust in enumerate(unique_customers)}

        # Map IDs to indices
        user_indices = user_history_df["customer_id"].map(cust_to_idx).values

        # Map article_ids to internal indices
        # Items not in training set will be -1
        idx_map = pd.Index(self.idx_to_article)
        article_indices = idx_map.get_indexer(user_history_df["article_id"].values)

        # Filter out unknown items
        valid_mask = article_indices != -1
        user_indices = user_indices[valid_mask]
        article_indices = article_indices[valid_mask]
        weights = user_history_df["weight"].values[valid_mask]

        n_users = len(unique_customers)
        n_items = len(self.idx_to_article)

        # Build Sparse Matrix U (Users x Items)
        U = sp.coo_matrix(
            (weights, (user_indices, article_indices)), shape=(n_users, n_items)
        ).tocsr()

        # 2. Compute Scores
        # Transition Scores: U @ T
        print("Computing transition scores...")
        scores = U.dot(self.transition_matrix)

        # Repurchase Scores: Add U itself (weighted)
        # Note: U contains history weights. We boost them.
        print("Adding repurchase scores...")
        scores = scores + (U * config.REPURCHASE_WEIGHT)

        # 3. Extract Top K
        print("Extracting top predictions...")
        global_top_k = self.global_popularity[: config.TOP_K]

        predictions = []

        # Access CSR internals for speed
        indptr = scores.indptr
        indices = scores.indices
        data = scores.data

        for i in range(n_users):
            preds = []

            start = indptr[i]
            end = indptr[i + 1]

            if end > start:
                row_indices = indices[start:end]
                row_data = data[start:end]

                # Get top K
                if len(row_data) > config.TOP_K:
                    # argpartition to get top K unsorted
                    top_k_arg = np.argpartition(row_data, -config.TOP_K)[
                        -config.TOP_K :
                    ]
                    # Sort top K
                    sorted_top_k_arg = top_k_arg[np.argsort(row_data[top_k_arg])[::-1]]
                    best_indices = row_indices[sorted_top_k_arg]
                else:
                    # Sort all
                    sorted_arg = np.argsort(row_data)[::-1]
                    best_indices = row_indices[sorted_arg]

                preds = self.idx_to_article[best_indices].tolist()

            # Fallback
            if len(preds) < config.TOP_K:
                for p in global_top_k:
                    if p not in preds:
                        preds.append(p)
                        if len(preds) == config.TOP_K:
                            break

            predictions.append(" ".join(map(str, preds)))

        return pd.DataFrame(
            {"customer_id": unique_customers, "prediction": predictions}
        )
