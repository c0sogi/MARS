import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import os
import gc
import logging
from library.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Constructs and manages the sparse matrices for the Multi-Scale Stratified Graph Cascade.
    Handles ID mapping, Interaction Matrix construction, Similarity Matrix computation,
    Pruning, and Caching.
    """

    def __init__(self, config: Config):
        self.config = config
        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        self.n_users = 0
        self.n_items = 0

    def fit_mappings(self, customers_df: pd.DataFrame, articles_df: pd.DataFrame):
        """
        Create global mappings for users and items based on master metadata files.
        Ensures all potential users and items have assigned indices.
        """
        logger.info("Fitting ID mappings...")

        # Users
        unique_users = customers_df["customer_id"].unique()
        self.user_map = {uid: i for i, uid in enumerate(unique_users)}
        self.reverse_user_map = {i: uid for uid, i in self.user_map.items()}
        self.n_users = len(unique_users)

        # Items
        unique_items = articles_df["article_id"].unique()
        self.item_map = {iid: i for i, iid in enumerate(unique_items)}
        self.reverse_item_map = {i: iid for iid, i in self.item_map.items()}
        self.n_items = len(unique_items)

        logger.info(f"Mappings fitted. Users: {self.n_users}, Items: {self.n_items}")

    def save_mappings(self):
        """Save ID mappings to disk."""
        path = self.config.CACHE_MAPPINGS
        np.savez(
            path,
            user_keys=list(self.user_map.keys()),
            user_vals=list(self.user_map.values()),
            item_keys=list(self.item_map.keys()),
            item_vals=list(self.item_map.values()),
        )
        logger.info("Saved mappings to cache.")

    def load_mappings(self) -> bool:
        """Load ID mappings from disk if available."""
        path = self.config.CACHE_MAPPINGS
        if os.path.exists(path):
            logger.info("Loading mappings from cache...")
            data = np.load(path, allow_pickle=True)
            self.user_map = dict(zip(data["user_keys"], data["user_vals"]))
            self.item_map = dict(zip(data["item_keys"], data["item_vals"]))

            # Rebuild reverse mappings
            self.reverse_user_map = {v: k for k, v in self.user_map.items()}
            self.reverse_item_map = {v: k for k, v in self.item_map.items()}

            self.n_users = len(self.user_map)
            self.n_items = len(self.item_map)
            logger.info(
                f"Loaded mappings. Users: {self.n_users}, Items: {self.n_items}"
            )
            return True
        return False

    def _apply_idf(self, X: sp.csr_matrix) -> sp.csr_matrix:
        """
        Apply IDF weighting to the columns (items) of the interaction matrix.
        IDF_i = log(N_users / (1 + count_i))
        """
        logger.info("Applying IDF weighting...")
        N = X.shape[0]
        # Count non-zero entries per column (item popularity)
        item_counts = np.array(X.getnnz(axis=0)).astype(np.float32)

        # Compute IDF
        idf = np.log(N / (1 + item_counts))

        # Create diagonal matrix and apply to columns
        idf_diag = sp.diags(idf)
        X_idf = X.dot(idf_diag)

        return X_idf

    def _prune_similarity_matrix(self, S: sp.csr_matrix, k: int) -> sp.csr_matrix:
        """
        Prune similarity matrix to keep only top-K values per row.
        Uses block-wise densification to handle memory efficiently.
        """
        logger.info(f"Pruning similarity matrix to top-{k} neighbors...")

        rows = []
        cols = []
        data = []

        # Process in chunks to save memory
        # 1000 rows * 100k cols * 4 bytes approx 400MB per chunk
        chunk_size = 1000
        n_rows = S.shape[0]

        for i in range(0, n_rows, chunk_size):
            end = min(i + chunk_size, n_rows)
            # Convert chunk to dense for efficient top-k selection
            chunk = S[i:end].toarray()

            # Iterate through rows in the chunk
            for r_idx in range(chunk.shape[0]):
                row_vals = chunk[r_idx]

                # If row has more than k non-zeros, prune
                if np.count_nonzero(row_vals) > k:
                    # argpartition puts the top k elements at the end
                    top_k_idx = np.argpartition(row_vals, -k)[-k:]

                    # Retrieve values and filter zeros (just in case)
                    vals = row_vals[top_k_idx]
                    mask = vals > 0

                    valid_indices = top_k_idx[mask]
                    valid_vals = vals[mask]

                    rows.extend([i + r_idx] * len(valid_indices))
                    cols.extend(valid_indices)
                    data.extend(valid_vals)
                else:
                    # Keep all non-zero elements
                    nz_idx = np.nonzero(row_vals)[0]
                    rows.extend([i + r_idx] * len(nz_idx))
                    cols.extend(nz_idx)
                    data.extend(row_vals[nz_idx])

            if (i // chunk_size) % 10 == 0:
                gc.collect()

        # Rebuild CSR matrix
        S_pruned = sp.csr_matrix((data, (rows, cols)), shape=S.shape, dtype=np.float32)
        return S_pruned

    def build_interaction_matrix(
        self, df: pd.DataFrame, weight_col: str
    ) -> sp.csr_matrix:
        """
        Builds the User-Item interaction matrix from a DataFrame.
        Maps IDs to indices and aggregates weights.
        """
        logger.info(f"Building interaction matrix using {weight_col}...")

        # Filter to only rows with known users and items
        # We perform an inner join with the mapping dataframes
        user_map_df = pd.DataFrame(
            list(self.user_map.items()), columns=["customer_id", "uid"]
        )
        item_map_df = pd.DataFrame(
            list(self.item_map.items()), columns=["article_id", "iid"]
        )

        # Select only necessary columns to save memory
        temp_df = df[["customer_id", "article_id", weight_col]].copy()

        # Map IDs
        temp_df = temp_df.merge(user_map_df, on="customer_id", how="inner")
        temp_df = temp_df.merge(item_map_df, on="article_id", how="inner")

        # Aggregate weights for duplicate (user, item) pairs (Summation)
        grouped = temp_df.groupby(["uid", "iid"])[weight_col].sum().reset_index()

        # Construct CSR Matrix
        X = sp.csr_matrix(
            (
                grouped[weight_col].values,
                (grouped["uid"].values, grouped["iid"].values),
            ),
            shape=(self.n_users, self.n_items),
            dtype=np.float32,
        )

        del temp_df, grouped, user_map_df, item_map_df
        gc.collect()

        return X

    def build_similarity_matrix(self, X: sp.csr_matrix, prune_k: int) -> sp.csr_matrix:
        """
        Builds Item-Item similarity matrix S = X^T X.
        Applies IDF weighting, L2 Row Normalization, and Top-K Pruning.
        """
        logger.info("Building similarity matrix...")

        # 1. Apply IDF
        X_weighted = self._apply_idf(X)

        # 2. L2 Normalize Rows (Users)
        # Normalizes user vectors so high-activity users don't dominate
        X_norm = normalize(X_weighted, norm="l2", axis=1)

        # 3. Compute S = X^T @ X
        # Result is (Items x Items)
        logger.info("Computing dot product X.T @ X...")
        S = X_norm.T @ X_norm

        # 4. Prune to keep matrix sparse and inference fast
        S = self._prune_similarity_matrix(S, prune_k)

        return S

    def apply_inventory_mask(
        self, S: sp.csr_matrix, active_items: np.ndarray
    ) -> sp.csr_matrix:
        """
        Zeros out columns in S that correspond to items NOT in the active inventory.
        Used for the 'Slow' graph to prevent recommending obsolete items.
        """
        logger.info("Applying active inventory mask...")

        # Identify indices of active items
        active_indices = [
            self.item_map[iid] for iid in active_items if iid in self.item_map
        ]

        # Create a diagonal mask matrix (1 for active, 0 for inactive)
        mask = np.zeros(self.n_items, dtype=np.float32)
        mask[active_indices] = 1.0
        mask_diag = sp.diags(mask)

        # Apply mask: S * Mask (Scales columns)
        S_masked = S.dot(mask_diag)

        return S_masked

    def save_artifacts(self, name: str, X: sp.csr_matrix, S: sp.csr_matrix):
        """Save interaction and similarity matrices to cache."""
        path_X = os.path.join(self.config.WORKING_DIR, f"X_{name}.npz")
        path_S = os.path.join(self.config.WORKING_DIR, f"S_{name}.npz")
        sp.save_npz(path_X, X)
        sp.save_npz(path_S, S)
        logger.info(f"Saved artifacts for {name}")

    def load_artifacts(self, name: str):
        """Load interaction and similarity matrices from cache."""
        path_X = os.path.join(self.config.WORKING_DIR, f"X_{name}.npz")
        path_S = os.path.join(self.config.WORKING_DIR, f"S_{name}.npz")

        if os.path.exists(path_X) and os.path.exists(path_S):
            logger.info(f"Loading artifacts for {name} from cache...")
            X = sp.load_npz(path_X)
            S = sp.load_npz(path_S)
            return X, S
        return None, None

    def run(
        self,
        train_df: pd.DataFrame,
        customers_df: pd.DataFrame,
        articles_df: pd.DataFrame,
        active_items: np.ndarray,
        load_cached: bool = True,
    ):
        """
        Main execution method to build (or load) all graphs.

        Args:
            train_df: DataFrame containing training transactions.
            customers_df: Master customers DataFrame.
            articles_df: Master articles DataFrame.
            active_items: Array of article_ids considered active (for Slow graph masking).
            load_cached: Whether to attempt loading from cache.

        Returns:
            X_fast, S_fast, X_slow, S_slow
        """
        # 1. Mappings
        if load_cached and self.load_mappings():
            pass
        else:
            self.fit_mappings(customers_df, articles_df)
            self.save_mappings()

        # 2. Fast Graph
        X_fast, S_fast = None, None
        if load_cached:
            X_fast, S_fast = self.load_artifacts("fast")

        if X_fast is None:
            logger.info("Constructing Fast Graph...")
            X_fast = self.build_interaction_matrix(train_df, "weight_fast")
            S_fast = self.build_similarity_matrix(X_fast, self.config.TOP_K_NEIGHBORS)
            self.save_artifacts("fast", X_fast, S_fast)

        # 3. Slow Graph
        X_slow, S_slow = None, None
        if load_cached:
            X_slow, S_slow = self.load_artifacts("slow")

        if X_slow is None:
            logger.info("Constructing Slow Graph...")
            X_slow = self.build_interaction_matrix(train_df, "weight_slow")
            S_slow = self.build_similarity_matrix(X_slow, self.config.TOP_K_NEIGHBORS)

            # Apply Inventory Mask to Slow Graph
            S_slow = self.apply_inventory_mask(S_slow, active_items)

            self.save_artifacts("slow", X_slow, S_slow)

        return X_fast, S_fast, X_slow, S_slow
