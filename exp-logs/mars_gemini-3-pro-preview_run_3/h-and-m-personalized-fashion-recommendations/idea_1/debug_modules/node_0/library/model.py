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

    def generate_predictions(self, user_history_df):
        """
        Generates predictions for users based on their last purchased item.

        Args:
            user_history_df (pd.DataFrame): DataFrame with 'customer_id' and 'article_id'
                                            representing the last purchase.

        Returns:
            pd.DataFrame: DataFrame with 'customer_id' and 'prediction' columns.
        """
        print("Generating predictions...")

        # Pre-compute global top K for fallback
        global_top_k = self.global_popularity[: config.TOP_K]

        # Prepare inputs
        customer_ids = user_history_df["customer_id"].values
        last_article_ids = user_history_df["article_id"].values

        # Map last articles to indices
        # pd.Index.get_indexer is very fast for bulk lookups
        # Returns -1 for items not found in the index
        idx_map = pd.Index(self.idx_to_article)
        source_indices = idx_map.get_indexer(last_article_ids)

        predictions = []

        # Iterate over each query
        # While a loop in python is generally slow, iterating 1.3M times with efficient
        # sparse row slicing is feasible within the time limit compared to full matrix ops.

        # Optimization: Access CSR internals directly to avoid overhead
        indptr = self.transition_matrix.indptr
        indices = self.transition_matrix.indices
        data = self.transition_matrix.data

        for src_idx in source_indices:
            preds = []

            # If the source item exists in our graph
            if src_idx != -1:
                # Get the row corresponding to this article
                start = indptr[src_idx]
                end = indptr[src_idx + 1]

                if end > start:
                    row_indices = indices[start:end]
                    row_data = data[start:end]

                    # Get top K neighbors
                    if len(row_data) > config.TOP_K:
                        # argpartition puts the top K elements at the end (unsorted)
                        top_k_arg = np.argpartition(row_data, -config.TOP_K)[
                            -config.TOP_K :
                        ]
                        # Sort these top K by weight descending
                        sorted_top_k_arg = top_k_arg[
                            np.argsort(row_data[top_k_arg])[::-1]
                        ]
                        best_indices = row_indices[sorted_top_k_arg]
                    else:
                        # If fewer than K, just sort all of them
                        sorted_arg = np.argsort(row_data)[::-1]
                        best_indices = row_indices[sorted_arg]

                    # Map back to article IDs
                    preds = self.idx_to_article[best_indices].tolist()

            # Fallback / Padding
            # Fill with global popularity if we don't have 12 predictions
            if len(preds) < config.TOP_K:
                for p in global_top_k:
                    if p not in preds:
                        preds.append(p)
                        if len(preds) == config.TOP_K:
                            break

            # Format as space-separated string
            predictions.append(" ".join(map(str, preds)))

        return pd.DataFrame({"customer_id": customer_ids, "prediction": predictions})
