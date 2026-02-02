import os
import gc
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from library.utils import Timer, memory_cleanup


class SparseMatrixOps:
    """
    A library of specialized sparse matrix operations for the Trend-Modulated Vectorized Cascade (TMVC)
    architecture. Handles the construction of time-decayed interaction matrices and efficient
    Item-Item similarity computation with IDF weighting and Top-K pruning.
    """

    def __init__(self, cache_dir="./working/idea_17"):
        """
        Initialize the SparseMatrixOps manager.

        Args:
            cache_dir (str): Directory to store cached mappings and matrices.
        """
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_mappings(self, articles_df, customers_df, load_cached=True):
        """
        Generates or loads global mappings for customers and articles.
        Ensures consistent matrix dimensions across different pipeline stages.

        Args:
            articles_df (pd.DataFrame): Master dataframe of all articles.
            customers_df (pd.DataFrame): Master dataframe of all customers.
            load_cached (bool): Whether to attempt loading from cache.

        Returns:
            tuple: (item_map, user_map, inverse_item_map, inverse_user_map)
                   item_map: dict(article_id -> int)
                   user_map: dict(customer_id -> int)
                   inverse_item_map: dict(int -> article_id)
                   inverse_user_map: dict(int -> customer_id)
        """
        item_map_path = os.path.join(self.cache_dir, "item_map.parquet")
        user_map_path = os.path.join(self.cache_dir, "user_map.parquet")

        if (
            load_cached
            and os.path.exists(item_map_path)
            and os.path.exists(user_map_path)
        ):
            print("[SparseMatrixOps] Loading cached mappings...")
            item_df = pd.read_parquet(item_map_path)
            user_df = pd.read_parquet(user_map_path)

            # Reconstruct dictionaries
            item_map = dict(zip(item_df["article_id"], item_df["item_idx"]))
            user_map = dict(zip(user_df["customer_id"], user_df["user_idx"]))

            # Reconstruct inverse maps
            inv_item_map = dict(zip(item_df["item_idx"], item_df["article_id"]))
            inv_user_map = dict(zip(user_df["user_idx"], user_df["customer_id"]))

            return item_map, user_map, inv_item_map, inv_user_map

        with Timer("Generating Mappings"):
            # Create Item Map
            unique_articles = articles_df["article_id"].unique()
            item_map = {art_id: i for i, art_id in enumerate(unique_articles)}

            # Create User Map
            unique_customers = customers_df["customer_id"].unique()
            user_map = {cust_id: i for i, cust_id in enumerate(unique_customers)}

            # Create DataFrames for caching
            item_df = pd.DataFrame(
                {
                    "article_id": list(item_map.keys()),
                    "item_idx": list(item_map.values()),
                }
            )
            user_df = pd.DataFrame(
                {
                    "customer_id": list(user_map.keys()),
                    "user_idx": list(user_map.values()),
                }
            )

            # Save to cache
            item_df.to_parquet(item_map_path, index=False)
            user_df.to_parquet(user_map_path, index=False)

            # Inverse maps
            inv_item_map = {v: k for k, v in item_map.items()}
            inv_user_map = {v: k for k, v in user_map.items()}

            print(f"Mapped {len(item_map)} items and {len(user_map)} users.")

        return item_map, user_map, inv_item_map, inv_user_map

    def build_decayed_interaction_matrix(
        self,
        transactions_df,
        item_map,
        user_map,
        decay_strategy="sqrt",
        load_cached=True,
        cache_name="interaction",
    ):
        """
        Constructs a sparse interaction matrix with continuous time-decay.

        Args:
            transactions_df (pd.DataFrame): Transaction history.
            item_map (dict): Mapping from article_id to index.
            user_map (dict): Mapping from customer_id to index.
            decay_strategy (str): 'sqrt' (1/sqrt(t)), 'linear' (1/t), 'none' (1.0).
            load_cached (bool): Whether to load from cache.
            cache_name (str): Identifier for the cache file.

        Returns:
            scipy.sparse.csr_matrix: The interaction matrix (Users x Items).
        """
        file_name = f"{cache_name}_{decay_strategy}.npz"
        cache_path = os.path.join(self.cache_dir, file_name)

        if load_cached and os.path.exists(cache_path):
            print(f"[SparseMatrixOps] Loading cached matrix: {file_name}")
            return sp.load_npz(cache_path)

        with Timer(f"Building Interaction Matrix ({decay_strategy})"):
            # Ensure date format
            if not np.issubdtype(transactions_df["t_dat"].dtype, np.datetime64):
                transactions_df["t_dat"] = pd.to_datetime(transactions_df["t_dat"])

            # Calculate days elapsed
            max_date = transactions_df["t_dat"].max()
            # (max_date - t_dat).dt.days gives 0 for the most recent day
            days_elapsed = (max_date - transactions_df["t_dat"]).dt.days.values.astype(
                np.float32
            )

            # Avoid division by zero by adding 1.0 (so most recent day is day 1 in denominator terms)
            # Strategy definitions:
            if decay_strategy == "sqrt":
                # 1 / sqrt(days + 1)
                weights = 1.0 / np.sqrt(days_elapsed + 1.0)
            elif decay_strategy == "linear":
                # 1 / (days + 1)
                weights = 1.0 / (days_elapsed + 1.0)
            elif decay_strategy == "none":
                weights = np.ones_like(days_elapsed)
            else:
                raise ValueError(f"Unknown decay strategy: {decay_strategy}")

            # Map IDs to indices
            # We filter out transactions for items/users not in the maps (though maps should be global)
            # Using map is faster than apply
            # Note: If transactions_df is huge, map can be slow.
            # We assume transactions_df is already filtered or maps cover everything.
            # To be safe and fast, we use pandas mapping.

            # We need to handle potential missing keys if the maps provided are subsets
            # However, standard usage implies global maps.

            # Create a copy to avoid modifying original
            df_temp = transactions_df[["customer_id", "article_id"]].copy()
            df_temp["weight"] = weights

            # Map to indices
            # Using a series map is efficient
            # We assume inputs are strings as per metadata

            # Optimization: Use factorize if maps were not provided, but here maps are strictly enforced.
            # We use a merge for safety and speed on large data vs .map()

            # Convert maps to DF for merging
            # (This overhead is negligible compared to safety)
            # Actually, .map is fine if we are sure. Let's use map but handle NaNs.

            df_temp["user_idx"] = df_temp["customer_id"].map(user_map)
            df_temp["item_idx"] = df_temp["article_id"].map(item_map)

            # Drop unmapped
            initial_len = len(df_temp)
            df_temp = df_temp.dropna(subset=["user_idx", "item_idx"])
            if len(df_temp) < initial_len:
                print(
                    f"Warning: Dropped {initial_len - len(df_temp)} transactions due to missing map keys."
                )

            # Cast to integers
            rows = df_temp["user_idx"].astype(np.int32).values
            cols = df_temp["item_idx"].astype(np.int32).values
            data = df_temp["weight"].astype(np.float32).values

            # Shape
            n_users = len(user_map)
            n_items = len(item_map)

            # Construct COO then CSR
            # Sum duplicate entries (multiple purchases of same item on different days add up)
            matrix = sp.coo_matrix((data, (rows, cols)), shape=(n_users, n_items))
            matrix = matrix.tocsr()

            # Save
            print(f"Saving matrix to {cache_path}")
            sp.save_npz(cache_path, matrix)

            # Cleanup
            del df_temp, rows, cols, data
            memory_cleanup()

        return matrix

    def compute_cosine_similarity(
        self, interaction_matrix, top_k=100, load_cached=True, cache_name="similarity"
    ):
        """
        Computes the Item-Item Cosine Similarity matrix with IDF weighting and Top-K pruning.

        Logic:
        1. Apply Item-IDF weighting to the interaction matrix columns.
        2. L2 Normalize the rows (Users).
        3. Compute S = X.T @ X.
        4. Prune to Top-K neighbors per item.

        Args:
            interaction_matrix (scipy.sparse.csr_matrix): Users x Items matrix.
            top_k (int): Number of neighbors to keep per item.
            load_cached (bool): Whether to load from cache.
            cache_name (str): Cache identifier.

        Returns:
            scipy.sparse.csr_matrix: Items x Items similarity matrix.
        """
        file_name = f"{cache_name}_top{top_k}.npz"
        cache_path = os.path.join(self.cache_dir, file_name)

        if load_cached and os.path.exists(cache_path):
            print(f"[SparseMatrixOps] Loading cached similarity matrix: {file_name}")
            return sp.load_npz(cache_path)

        with Timer("Computing Cosine Similarity"):
            X = interaction_matrix.astype(np.float32)
            n_users, n_items = X.shape

            # --- 1. IDF Weighting ---
            print("Applying IDF Weighting...")
            # Document frequency = number of users who bought the item
            # Use X.csc to iterate columns efficiently or just simple bincount on indices if CSR
            # X is CSR. X.indices gives column indices.
            col_counts = np.bincount(X.indices, minlength=n_items)

            # IDF = log(N_users / (1 + count))
            # We add 1 to count to avoid division by zero
            idf = np.log(n_users / (col_counts + 1.0))

            # Apply IDF to columns.
            # Multiply X by diagonal matrix of IDF
            # Efficient way: X @ diag(idf)
            # Or multiply data: X.data *= idf[X.indices]
            X.data *= idf[X.indices]

            # --- 2. Row Normalization ---
            print("Applying Row-wise L2 Normalization...")
            # Normalize user vectors to unit length
            X = normalize(X, norm="l2", axis=1)

            # --- 3. Matrix Multiplication ---
            print(f"Computing X.T @ X (Shape: {n_items}x{n_items})...")
            # This results in Cosine Similarity between Items based on User vectors
            S = X.T @ X

            # Free X to save memory
            del X
            memory_cleanup()

            # --- 4. Top-K Pruning ---
            print(f"Pruning to Top-{top_k} neighbors...")

            # Strategy:
            # Since n_items ~105k, the matrix S is 105k x 105k.
            # If we densify it, it takes ~40GB RAM (float32). We have 220GB.
            # Densifying allows extremely fast vectorized argpartition.

            # Check if density is too high for sparse but okay for dense
            # Actually, we convert to dense block by block or fully if memory allows.
            # Let's try full dense conversion given the hardware specs.

            try:
                S_dense = S.toarray()  # This might spike memory
                del S
                memory_cleanup()

                # Set diagonal to 0 (item is not similar to itself for recommendation purposes)
                np.fill_diagonal(S_dense, 0.0)

                # Argpartition to find top K
                # We want indices of top K elements
                # argpartition puts the kth element in sorted position, and all smaller before, larger after
                # We want largest.
                # -top_k

                # Initialize output arrays for sparse construction
                rows = []
                cols = []
                data = []

                # We can iterate rows or do full matrix ops.
                # Full matrix argpartition:
                # indices = np.argpartition(S_dense, -top_k, axis=1)[:, -top_k:]
                # This gives the indices of the top k elements for each row.

                top_k_indices = np.argpartition(S_dense, -top_k, axis=1)[:, -top_k:]

                # Create row indices
                row_indices = np.arange(n_items)[:, None]  # Column vector
                row_indices = np.broadcast_to(row_indices, top_k_indices.shape)

                # Extract values
                top_k_values = S_dense[row_indices, top_k_indices]

                # Filter out zero values (if an item has fewer than K neighbors)
                mask = top_k_values > 0

                final_rows = row_indices[mask]
                final_cols = top_k_indices[mask]
                final_data = top_k_values[mask]

                del S_dense, row_indices, top_k_indices, top_k_values
                memory_cleanup()

                # Reconstruct CSR
                S_pruned = sp.csr_matrix(
                    (final_data, (final_rows, final_cols)), shape=(n_items, n_items)
                )

            except MemoryError:
                print(
                    "MemoryError during dense pruning. Falling back to sparse row iteration (slower)."
                )
                # Fallback: iterate sparse rows
                # This is just a safety net, unlikely to be hit with 220GB RAM
                S_pruned = sp.lil_matrix(S.shape, dtype=np.float32)
                for i in range(S.shape[0]):
                    row = S.getrow(i)
                    if row.nnz > top_k:
                        # get indices of top k
                        ind = np.argpartition(row.data, -top_k)[-top_k:]
                        S_pruned[i, row.indices[ind]] = row.data[ind]
                    else:
                        S_pruned[i, row.indices] = row.data
                S_pruned = S_pruned.tocsr()

            print(f"Saving similarity matrix to {cache_path}")
            sp.save_npz(cache_path, S_pruned)

            return S_pruned
