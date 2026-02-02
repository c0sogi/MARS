import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import gc
from sklearn.preprocessing import normalize
from library.utils import Timer, reduce_mem_usage


class SparseMatrixBuilder:
    """
    Constructs the Decay-Weighted Interaction Matrix (X_decay) from transaction data.
    Handles ID mapping, aggregation, IDF weighting, and normalization.
    """

    def __init__(self, cache_dir="./working/idea_8"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_paths(self, suffix=""):
        """Returns file paths for cached artifacts."""
        return {
            "matrix": os.path.join(self.cache_dir, f"interaction_matrix{suffix}.npz"),
            "user_map": os.path.join(self.cache_dir, f"user_map{suffix}.parquet"),
            "item_map": os.path.join(self.cache_dir, f"item_map{suffix}.parquet"),
        }

    def build(self, train_df, test_customers, load_cached_data=True, suffix=""):
        """
        Builds the processed interaction matrix.

        Args:
            train_df (pd.DataFrame): Transaction data with 'customer_id', 'article_id', 'decay_weight'.
            test_customers (pd.DataFrame): DataFrame with 'customer_id' for submission.
            load_cached_data (bool): Whether to attempt loading from disk.
            suffix (str): Optional suffix for filenames (e.g. for different validation folds).

        Returns:
            X (csr_matrix): The normalized, IDF-weighted interaction matrix.
            user_map (pd.Series): Mapping from customer_id to integer index.
            item_map (pd.Series): Mapping from article_id to integer index.
        """
        paths = self._get_paths(suffix)

        # 1. Try Loading from Cache
        if load_cached_data:
            if all(os.path.exists(p) for p in paths.values()):
                print(
                    f"[SparseMatrixBuilder] Loading cached matrix and maps from {self.cache_dir}..."
                )
                with Timer("Load Cache"):
                    X = sp.load_npz(paths["matrix"])
                    user_map = pd.read_parquet(paths["user_map"])["user_idx"]
                    item_map = pd.read_parquet(paths["item_map"])["item_idx"]
                    return X, user_map, item_map
            else:
                print("[SparseMatrixBuilder] Cache missing. Rebuilding...")

        # 2. Build Maps
        with Timer("Build ID Maps"):
            # Users: Union of Train and Test to ensure all required users have an index
            unique_train_users = train_df["customer_id"].unique()
            unique_test_users = test_customers["customer_id"].unique()

            # Using numpy union1d for sorted unique elements
            all_users = np.union1d(unique_train_users, unique_test_users)

            # Items: Only those present in training data
            all_items = train_df["article_id"].unique()

            # Create Mappings (ID -> Index)
            user_map = pd.Series(
                data=np.arange(len(all_users), dtype=np.int32),
                index=all_users,
                name="user_idx",
            )

            item_map = pd.Series(
                data=np.arange(len(all_items), dtype=np.int32),
                index=all_items,
                name="item_idx",
            )

            print(f"  Unique Users: {len(user_map)}")
            print(f"  Unique Items: {len(item_map)}")

        # 3. Map Dataframes to Indices
        with Timer("Map Transactions"):
            # We map the IDs in train_df to the generated indices
            # Using map is faster than merge for single columns

            # Ensure we are working with the correct types for mapping
            if train_df["article_id"].dtype != item_map.index.dtype:
                train_df["article_id"] = train_df["article_id"].astype(
                    item_map.index.dtype
                )

            # Map users
            # Note: We use reindex or map. Since we built map from union, all train users are in map.
            user_indices = train_df["customer_id"].map(user_map).astype(np.int32)

            # Map items
            item_indices = train_df["article_id"].map(item_map).astype(np.int32)

            weights = train_df["decay_weight"].values.astype(np.float32)

            # Handle any potential mapping failures (should be none if logic is correct)
            valid_mask = (~user_indices.isna()) & (~item_indices.isna())
            if not valid_mask.all():
                print(
                    f"  Warning: Dropping {len(train_df) - valid_mask.sum()} rows due to mapping issues."
                )
                user_indices = user_indices[valid_mask]
                item_indices = item_indices[valid_mask]
                weights = weights[valid_mask]

        # 4. Construct Sparse Matrix
        with Timer("Construct CSR Matrix"):
            # Create COO matrix
            # Shape is (Total Users, Total Items)
            n_users = len(user_map)
            n_items = len(item_map)

            X = sp.coo_matrix(
                (weights, (user_indices, item_indices)),
                shape=(n_users, n_items),
                dtype=np.float32,
            )

            # Convert to CSR and sum duplicates
            # This efficiently handles multiple purchases of the same item by the same user
            # by summing their decay weights.
            X = X.tocsr()
            X.sum_duplicates()

            print(f"  Raw Matrix Shape: {X.shape}, NNZ: {X.nnz}")

        # 5. Apply IDF Weighting
        with Timer("Apply IDF"):
            # Document Frequency: Number of users who bought the item
            # In a binary matrix, this is column sum. In weighted, we count non-zeros.
            # Convert to CSC for efficient column operations if needed, but CSR is fine for simple counts

            # Count non-zeros per column
            col_nnz = np.diff(X.tocsc().indptr)

            # IDF formula: log(N / (df + 1)) + 1 (smooth IDF)
            N = n_users
            idf = np.log(N / (col_nnz + 1.0)) + 1.0
            idf = idf.astype(np.float32)

            # Create diagonal matrix
            IDF_mat = sp.diags(idf)

            # Apply weighting: X_weighted = X * IDF
            X = X.dot(IDF_mat)

        # 6. Normalize Rows
        with Timer("Normalize Rows"):
            # L2 normalization per user
            X = normalize(X, norm="l2", axis=1)

        # 7. Save to Cache
        with Timer("Save Artifacts"):
            sp.save_npz(paths["matrix"], X)

            # Save maps as Parquet (reset index to save the ID column)
            pd.DataFrame(user_map).reset_index().rename(
                columns={"index": "customer_id"}
            ).to_parquet(paths["user_map"])
            pd.DataFrame(item_map).reset_index().rename(
                columns={"index": "article_id"}
            ).to_parquet(paths["item_map"])

        return X, user_map, item_map
