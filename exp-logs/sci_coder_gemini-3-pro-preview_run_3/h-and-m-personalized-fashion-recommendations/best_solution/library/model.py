import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
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

        # Save Matrix
        print(f"Saving transition matrix to {matrix_path}...")
        sp.save_npz(matrix_path, self.transition_matrix)

    def generate_predictions(self, target_customers, user_history_df):
        """
        Generates predictions using Vectorized Sparse Propagation (Cite solution_lesson_node_00005).

        S = U * T + alpha * U

        Args:
            target_customers (array-like): List of customer_ids to predict for.
            user_history_df (pd.DataFrame): DataFrame with 'customer_id', 'article_id', 'weight'.

        Returns:
            pd.DataFrame: DataFrame with 'customer_id' and 'prediction' columns.
        """
        print(f"Generating predictions for {len(target_customers)} users...")

        # Pre-compute global top K for fallback
        global_top_k = self.global_popularity[: config.TOP_K]

        # 1. Build Sparse User Matrix U (N_users x N_items)
        print("Building sparse user history matrix...")

        # Map customers to row indices
        cust_to_idx = {cust: i for i, cust in enumerate(target_customers)}

        # Filter history for target customers only
        history_subset = user_history_df[
            user_history_df["customer_id"].isin(cust_to_idx)
        ].copy()

        if len(history_subset) == 0:
            print(
                "Warning: No history found for target customers. Using global popularity."
            )
            default_pred = " ".join(map(str, global_top_k))
            return pd.DataFrame(
                {
                    "customer_id": target_customers,
                    "prediction": [default_pred] * len(target_customers),
                }
            )

        # Map IDs to indices
        history_subset["row_idx"] = history_subset["customer_id"].map(cust_to_idx)
        history_subset["col_idx"] = history_subset["article_id"].map(
            self.article_to_idx
        )

        # Drop invalid items (items not in training set)
        history_subset = history_subset.dropna(subset=["col_idx"])

        # Construct CSR Matrix
        rows = history_subset["row_idx"].values
        cols = history_subset["col_idx"].astype(int).values
        data = history_subset["weight"].values

        n_users = len(target_customers)
        n_items = len(self.idx_to_article)

        user_matrix = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

        # 2. Batch Inference (Cite solution_lesson_node_00007)
        predictions = []
        batch_size = config.BATCH_SIZE

        print(f"Processing in batches of {batch_size}...")

        for start_idx in range(0, n_users, batch_size):
            end_idx = min(start_idx + batch_size, n_users)

            # Slice user matrix
            U_batch = user_matrix[start_idx:end_idx]

            # Compute Scores: S = U * T + alpha * U
            # Cite solution_lesson_node_00005 (Vectorized Propagation)
            # Cite solution_lesson_node_00011 (Repurchase Weight)
            scores = U_batch.dot(self.transition_matrix)
            scores += config.REPURCHASE_WEIGHT * U_batch

            # Extract Top K (Cite solution_lesson_node_00007 - Avoid dense materialization)
            batch_preds = []

            for i in range(scores.shape[0]):
                row = scores[i]

                if row.nnz == 0:
                    batch_preds.append(" ".join(map(str, global_top_k)))
                    continue

                r_indices = row.indices
                r_data = row.data

                if len(r_data) > config.TOP_K:
                    top_k_idx = np.argpartition(r_data, -config.TOP_K)[-config.TOP_K :]
                    sorted_top_k_idx = top_k_idx[np.argsort(r_data[top_k_idx])[::-1]]
                    best_item_indices = r_indices[sorted_top_k_idx]
                else:
                    sorted_idx = np.argsort(r_data)[::-1]
                    best_item_indices = r_indices[sorted_idx]

                candidates = self.idx_to_article[best_item_indices].tolist()

                # Fill with popularity if needed
                if len(candidates) < config.TOP_K:
                    seen = set(candidates)
                    for pop_item in global_top_k:
                        if pop_item not in seen:
                            candidates.append(pop_item)
                            seen.add(pop_item)
                            if len(candidates) >= config.TOP_K:
                                break

                batch_preds.append(" ".join(map(str, candidates)))

            predictions.extend(batch_preds)

        return pd.DataFrame(
            {"customer_id": target_customers, "prediction": predictions}
        )
