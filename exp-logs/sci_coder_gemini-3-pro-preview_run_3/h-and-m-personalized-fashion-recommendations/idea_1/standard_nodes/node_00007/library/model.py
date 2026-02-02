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

    def generate_predictions(self, transactions_df, customer_ids_to_predict):
        """
        Generates predictions using Vectorized Sparse Propagation (Cite solution_lesson_node_00005).

        Args:
            transactions_df (pd.DataFrame): User history transactions.
            customer_ids_to_predict (array-like): List of customer_ids to predict for.

        Returns:
            pd.DataFrame: DataFrame with 'customer_id' and 'prediction' columns.
        """
        print("Generating predictions using Vectorized Sparse Propagation...")

        # 1. Setup
        customer_ids = np.array(customer_ids_to_predict)
        n_users = len(customer_ids)
        n_items = len(self.idx_to_article)

        # Map customers to 0..N indices for the sparse matrix
        cust_to_idx = {cid: i for i, cid in enumerate(customer_ids)}

        # 2. Build User-Item History Matrix (U)
        # Filter transactions to only relevant users
        # We need to map the customer_ids in transactions_df to our new 0..N indices
        print("Building user history matrix...")
        relevant_txns = transactions_df[
            transactions_df["customer_id"].isin(cust_to_idx)
        ].copy()

        # Map customers to row indices
        relevant_txns["user_idx"] = relevant_txns["customer_id"].map(cust_to_idx)

        # Map articles to col indices
        # Note: self.article_to_idx contains all items in the transition graph
        # Items not in the graph are ignored (as they have no transitions)
        relevant_txns["item_idx"] = relevant_txns["article_id"].map(self.article_to_idx)

        # Drop invalid items
        relevant_txns = relevant_txns.dropna(subset=["item_idx"])
        relevant_txns["item_idx"] = relevant_txns["item_idx"].astype(int)

        # Create Sparse Matrix U
        # Shape: (n_users_to_predict, n_items)
        # We sum weights if user bought item multiple times
        U = sp.csr_matrix(
            (
                relevant_txns["weight"].values,
                (relevant_txns["user_idx"].values, relevant_txns["item_idx"].values),
            ),
            shape=(n_users, n_items),
        )

        # 3. Generate Predictions in Batches
        predictions = []
        global_top_k = self.global_popularity[: config.TOP_K]

        # Pre-compute fallback string
        fallback_pred = " ".join(map(str, global_top_k))

        print(f"Processing {n_users} users in batches of {config.BATCH_SIZE}...")

        for start in range(0, n_users, config.BATCH_SIZE):
            end = min(start + config.BATCH_SIZE, n_users)

            # Slice U for this batch
            U_batch = U[start:end]

            # Compute Scores: S = U * T + alpha * U
            # U_batch: (B, I), T: (I, I) -> S: (B, I)
            # Cite solution_lesson_node_00005: Vectorized propagation
            scores = U_batch.dot(self.transition_matrix)

            # Add Repurchase signal
            if config.REPURCHASE_WEIGHT > 0:
                scores = scores + config.REPURCHASE_WEIGHT * U_batch

            # Convert to dense for sorting (assuming BATCH_SIZE is small enough)
            # scores is csr_matrix. toarray() returns dense.
            scores_dense = scores.toarray()

            # Extract Top K
            batch_preds = []
            for i in range(len(scores_dense)):
                row_scores = scores_dense[i]

                # Check if row is empty (Cold Start)
                if row_scores.sum() == 0:
                    batch_preds.append(fallback_pred)
                    continue

                # Get indices of top K
                # argpartition is faster than sort
                if len(row_scores) > config.TOP_K:
                    top_k_idx = np.argpartition(row_scores, -config.TOP_K)[
                        -config.TOP_K :
                    ]
                    # Sort these top K
                    top_k_idx = top_k_idx[np.argsort(row_scores[top_k_idx])[::-1]]
                else:
                    top_k_idx = np.argsort(row_scores)[::-1]

                # Map back to article_ids
                top_items = self.idx_to_article[top_k_idx]
                batch_preds.append(" ".join(map(str, top_items)))

            predictions.extend(batch_preds)

        return pd.DataFrame({"customer_id": customer_ids, "prediction": predictions})
