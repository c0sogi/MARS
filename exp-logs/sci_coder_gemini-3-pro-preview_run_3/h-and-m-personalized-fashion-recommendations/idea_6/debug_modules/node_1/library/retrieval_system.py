import pandas as pd
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from library.config import Config
from library.utils import Timer, normalize_matrix


class SparseRetriever:
    """
    Stage 1: Multi-Temporal Vectorized Retrieval.

    Responsibilities:
    1. Construct User Vectors from transaction history with time decay.
    2. Load pre-computed sparse graphs (Short-Term, Long-Term, Visual).
    3. Perform sparse matrix multiplication (User x Graph) to generate scores.
    4. Merge results from multiple graphs to produce a candidate dataset.
    """

    def __init__(self):
        # Load entity maps to determine matrix dimensions
        self.article_map = np.load(Config.CACHE_ARTICLE_MAP, allow_pickle=True)
        self.customer_map = np.load(Config.CACHE_CUSTOMER_MAP, allow_pickle=True)
        self.n_articles = len(self.article_map)
        self.n_customers = len(self.customer_map)

    def compute_user_vectors(
        self, history_df: pd.DataFrame, target_date: pd.Timestamp
    ) -> sp.csr_matrix:
        """
        Constructs the User Representation Vector U based on history.
        Applies power-law time decay: w = 1 / (1 + days_elapsed).

        Args:
            history_df: DataFrame containing ['customer_id', 'article_id', 't_dat'].
                        IDs must be mapped integers.
            target_date: The reference date for calculating decay (usually start of prediction window).

        Returns:
            sp.csr_matrix: Shape (n_customers, n_articles), L1 normalized per row.
        """
        with Timer("Compute User Vectors"):
            if history_df.empty:
                return sp.csr_matrix(
                    (self.n_customers, self.n_articles), dtype=np.float32
                )

            # Ensure datetime
            if not np.issubdtype(history_df["t_dat"].dtype, np.datetime64):
                history_df = history_df.copy()
                history_df["t_dat"] = pd.to_datetime(history_df["t_dat"])

            # Calculate decay weights
            # days_elapsed = (target_date - t_dat).days
            # We use vectorized operations
            delta_days = (target_date - history_df["t_dat"]).dt.days.values.astype(
                np.float32
            )

            # Clip negative days (if history leaks into future slightly) to 0
            delta_days = np.maximum(delta_days, 0)

            weights = 1.0 / (1.0 + delta_days)

            # Construct Sparse Matrix
            row = history_df["customer_id"].values
            col = history_df["article_id"].values

            # Aggregate weights for duplicate (user, item) pairs automatically
            user_matrix = sp.coo_matrix(
                (weights, (row, col)), shape=(self.n_customers, self.n_articles)
            ).tocsr()

            # Normalize to create a probability-like distribution of user interests
            user_matrix = normalize_matrix(user_matrix, axis=1, norm="l1")

            return user_matrix

    def retrieve_candidates(self, user_vectors: sp.csr_matrix) -> pd.DataFrame:
        """
        Generates candidates by propagating user vectors through the multi-view graphs.

        Args:
            user_vectors: Sparse matrix (U) of user history.

        Returns:
            pd.DataFrame: Merged candidates with columns:
                          ['customer_id', 'article_id', 'score_short', 'score_long', 'score_vis', 'score_hist']
        """
        # 1. Load Graphs
        print("Loading sparse graphs for retrieval...")
        try:
            T_short = sp.load_npz(Config.CACHE_GRAPH_SHORT)
            T_long = sp.load_npz(Config.CACHE_GRAPH_LONG)
            T_vis = sp.load_npz(Config.CACHE_GRAPH_VISUAL)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Sparse graphs not found. Run GraphEngine.build_graphs() first."
            )

        # 2. Compute Scores (Matrix Multiplication)
        # Result shape: (n_customers, n_articles)
        with Timer("Propagation: Short-Term"):
            S_short = user_vectors.dot(T_short)

        with Timer("Propagation: Long-Term"):
            S_long = user_vectors.dot(T_long)

        with Timer("Propagation: Visual"):
            S_vis = user_vectors.dot(T_vis)

        # Repurchase scores are just the user history itself
        S_hist = user_vectors

        # 3. Extract Top-K Candidates from each source
        # We extract to DataFrames to facilitate merging
        k = Config.RETRIEVAL_TOP_K

        df_short = self._extract_top_k(S_short, k, "score_short")
        df_long = self._extract_top_k(S_long, k, "score_long")
        df_vis = self._extract_top_k(S_vis, k, "score_vis")
        df_hist = self._extract_top_k(S_hist, k, "score_hist")

        # 4. Merge Candidates
        with Timer("Merging Candidates"):
            # We use outer joins to keep the union of candidates
            # Start with Short as base
            merged_df = df_short

            # Merge Long
            merged_df = merged_df.merge(
                df_long, on=["customer_id", "article_id"], how="outer"
            )

            # Merge Visual
            merged_df = merged_df.merge(
                df_vis, on=["customer_id", "article_id"], how="outer"
            )

            # Merge History (Repurchase)
            merged_df = merged_df.merge(
                df_hist, on=["customer_id", "article_id"], how="outer"
            )

            # Fill NaNs with 0 (since sparse implies 0 score)
            score_cols = ["score_short", "score_long", "score_vis", "score_hist"]
            merged_df[score_cols] = merged_df[score_cols].fillna(0.0)

            # Downcast IDs to save memory
            merged_df["customer_id"] = merged_df["customer_id"].astype("int32")
            merged_df["article_id"] = merged_df["article_id"].astype("int32")
            for col in score_cols:
                merged_df[col] = merged_df[col].astype("float32")

        print(
            f"Generated {len(merged_df)} unique candidate pairs for {merged_df['customer_id'].nunique()} customers."
        )
        return merged_df

    def _extract_top_k(
        self, matrix: sp.csr_matrix, k: int, score_col: str
    ) -> pd.DataFrame:
        """
        Efficiently extracts the top K non-zero elements per row from a CSR matrix.
        Returns a DataFrame with ['customer_id', 'article_id', score_col].
        """
        # Prepare lists for bulk DataFrame creation
        # We iterate rows, but use slicing on the internal arrays for speed

        # Filter out empty rows to speed up iteration
        # (Though we need to keep track of correct customer_id index)

        n_rows = matrix.shape[0]

        # Arrays to hold results
        # We estimate size to pre-allocate or just use lists (lists are fast enough for appending blocks)
        all_cust_ids = []
        all_art_ids = []
        all_scores = []

        # Access CSR internals
        indptr = matrix.indptr
        indices = matrix.indices
        data = matrix.data

        # We can iterate only over rows that have data
        non_empty_rows = np.where(np.diff(indptr) > 0)[0]

        # Chunking for progress bar
        chunk_size = 10000

        for i in tqdm(
            range(0, len(non_empty_rows), chunk_size), desc=f"Extracting {score_col}"
        ):
            chunk_rows = non_empty_rows[i : i + chunk_size]

            for row_idx in chunk_rows:
                start = indptr[row_idx]
                end = indptr[row_idx + 1]

                row_indices = indices[start:end]
                row_data = data[start:end]

                if len(row_data) == 0:
                    continue

                # Get Top K
                if len(row_data) > k:
                    # argpartition is O(n) vs sort O(n log n)
                    # We want indices of the largest k elements
                    top_k_idx = np.argpartition(row_data, -k)[-k:]

                    selected_indices = row_indices[top_k_idx]
                    selected_scores = row_data[top_k_idx]
                else:
                    selected_indices = row_indices
                    selected_scores = row_data

                # Append
                # Repeat customer ID
                all_cust_ids.append(
                    np.full(len(selected_indices), row_idx, dtype=np.int32)
                )
                all_art_ids.append(selected_indices)
                all_scores.append(selected_scores)

        if not all_cust_ids:
            return pd.DataFrame(columns=["customer_id", "article_id", score_col])

        # Concatenate
        flat_cust = np.concatenate(all_cust_ids)
        flat_art = np.concatenate(all_art_ids)
        flat_score = np.concatenate(all_scores)

        df = pd.DataFrame(
            {"customer_id": flat_cust, "article_id": flat_art, score_col: flat_score}
        )

        return df
