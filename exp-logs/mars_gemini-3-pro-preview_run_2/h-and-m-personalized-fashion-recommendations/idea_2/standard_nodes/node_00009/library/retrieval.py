import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from library.config import (
    WORKING_DIR,
    CACHE_COOCCURRENCE,
    CACHE_GLOBAL_POPULARITY,
    CACHE_USER_MAP,
    CACHE_ITEM_MAP,
    CACHE_CANDIDATES_TRAIN,
    SEED,
    DATE_COL,
    USER_ID_COL,
    ITEM_ID_COL,
    N_CPUS,
)

# Set random seed
np.random.seed(SEED)


class CooccurrenceRecommender:
    """
    Stage 1: Candidate Retrieval using Time-Weighted Item-Item Co-occurrence.
    """

    def __init__(self):
        self.user_map = None  # pd.DataFrame: customer_id -> user_idx
        self.item_map = None  # pd.DataFrame: article_id -> item_idx
        self.user_item_matrix = None  # sparse matrix: History of users in training
        self.cooccurrence_matrix = None  # sparse matrix: Item-Item relationships
        self.global_popularity = None  # list: Top items for cold start
        self.user_id_to_idx = {}
        self.item_id_to_idx = {}
        self.idx_to_item_id = {}

    def fit(self, transactions_df, load_cached_data=True):
        """
        Builds the co-occurrence matrix from transaction history.

        Parameters
        ----------
        transactions_df : pd.DataFrame
            The training transactions.
        load_cached_data : bool
            Whether to attempt loading pre-computed matrices from disk.
        """
        # Ensure working directory exists
        os.makedirs(WORKING_DIR, exist_ok=True)

        # Paths for internal artifacts not explicitly defined in config but needed
        matrix_path = CACHE_COOCCURRENCE
        user_matrix_path = WORKING_DIR / "user_item_matrix.npz"

        # 1. Try Loading from Cache
        if load_cached_data:
            if (
                matrix_path.exists()
                and CACHE_USER_MAP.exists()
                and CACHE_ITEM_MAP.exists()
                and CACHE_GLOBAL_POPULARITY.exists()
                and user_matrix_path.exists()
            ):

                print("Loading retrieval model from cache...")
                self.cooccurrence_matrix = sp.load_npz(matrix_path)
                self.user_item_matrix = sp.load_npz(user_matrix_path)
                self.user_map = pd.read_parquet(CACHE_USER_MAP)
                self.item_map = pd.read_parquet(CACHE_ITEM_MAP)
                self.global_popularity = np.load(
                    CACHE_GLOBAL_POPULARITY, allow_pickle=True
                ).tolist()

                # Rebuild dictionaries for fast lookup
                self.user_id_to_idx = dict(
                    zip(self.user_map[USER_ID_COL], self.user_map["user_idx"])
                )
                self.item_id_to_idx = dict(
                    zip(self.item_map[ITEM_ID_COL], self.item_map["item_idx"])
                )
                self.idx_to_item_id = dict(
                    zip(self.item_map["item_idx"], self.item_map[ITEM_ID_COL])
                )

                print("Model loaded successfully.")
                return
            else:
                print("Cached retrieval model not found or incomplete. Recomputing...")

        # 2. Preprocessing
        print("Preprocessing transactions for retrieval...")
        df = transactions_df.copy()

        # Ensure date is datetime
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        max_date = df[DATE_COL].max()

        # Create Mappings
        print("Creating ID mappings...")
        unique_users = df[USER_ID_COL].unique()
        unique_items = df[ITEM_ID_COL].unique()

        self.user_map = pd.DataFrame({USER_ID_COL: unique_users})
        self.user_map["user_idx"] = np.arange(len(unique_users))

        self.item_map = pd.DataFrame({ITEM_ID_COL: unique_items})
        self.item_map["item_idx"] = np.arange(len(unique_items))

        # Create fast lookup dicts
        self.user_id_to_idx = dict(
            zip(self.user_map[USER_ID_COL], self.user_map["user_idx"])
        )
        self.item_id_to_idx = dict(
            zip(self.item_map[ITEM_ID_COL], self.item_map["item_idx"])
        )
        self.idx_to_item_id = dict(
            zip(self.item_map["item_idx"], self.item_map[ITEM_ID_COL])
        )

        # Map IDs in dataframe
        # Using map is faster than merge for single column
        df["user_idx"] = df[USER_ID_COL].map(self.user_id_to_idx)
        df["item_idx"] = df[ITEM_ID_COL].map(self.item_id_to_idx)

        # 3. Calculate Time Weights
        # Weight = 1 / (days_elapsed + 1)
        # High decay recency weighting
        print("Calculating time decay weights...")
        df["days_diff"] = (max_date - df[DATE_COL]).dt.days
        # Cite solution_lesson_node_00006: Aggressive time-decay (power 2.5) outperforms linear decay
        df["weight"] = 1.0 / ((df["days_diff"] + 1.0) ** 2.5)

        # 4. Build User-Item Matrix (History)
        # Rows: Users, Cols: Items, Values: Sum of weights (if multiple purchases)
        # We use sum to capture frequency + recency
        print("Building User-Item sparse matrix...")
        n_users = len(unique_users)
        n_items = len(unique_items)

        # Group by user and item to sum weights
        grouped = df.groupby(["user_idx", "item_idx"])["weight"].sum().reset_index()

        self.user_item_matrix = sp.csr_matrix(
            (
                grouped["weight"].values,
                (grouped["user_idx"].values, grouped["item_idx"].values),
            ),
            shape=(n_users, n_items),
        )

        # 5. Build Co-occurrence Matrix
        # C = U.T * U
        # This gives unnormalized co-occurrence scores
        print("Computing Co-occurrence matrix (A^T * A)...")
        # Note: This can be memory intensive. With 100k items, it's manageable on 220GB RAM.
        self.cooccurrence_matrix = self.user_item_matrix.T.dot(self.user_item_matrix)

        # Set diagonal to zero (we don't recommend item i because user bought item i,
        # though for consumables this might be valid, for fashion usually we want 'complete the look')
        # However, re-purchasing basics is common. Let's keep diagonal but maybe dampen it?
        # The prompt implies standard co-occurrence. Let's strictly follow "bought together".
        # Usually "bought together" implies i != j.
        # But for simplicity and to allow re-purchase prediction, we leave it.

        # 6. Compute Global Popularity (Cold Start)
        # Top items from the last 7 days
        print("Computing Global Popularity...")
        last_week_start = max_date - pd.Timedelta(days=7)
        recent_df = df[df[DATE_COL] > last_week_start]

        # Count frequency (simple count, not weighted, for popularity)
        pop_counts = recent_df[ITEM_ID_COL].value_counts()
        self.global_popularity = pop_counts.head(12).index.tolist()

        # 7. Save to Cache
        print("Saving artifacts to cache...")
        sp.save_npz(matrix_path, self.cooccurrence_matrix)
        sp.save_npz(user_matrix_path, self.user_item_matrix)
        self.user_map.to_parquet(CACHE_USER_MAP, index=False)
        self.item_map.to_parquet(CACHE_ITEM_MAP, index=False)
        np.save(CACHE_GLOBAL_POPULARITY, np.array(self.global_popularity))

        print("Fit complete.")

    def generate_candidates(self, customer_ids, k=100):
        """
        Generates candidate items for a list of customers.

        Parameters
        ----------
        customer_ids : list or np.array
            List of customer_ids to generate candidates for.
        k : int
            Number of candidates to retrieve per user.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns [customer_id, article_id, cooccurrence_score]
        """
        print(f"Generating candidates for {len(customer_ids)} customers...")

        # Identify Warm vs Cold users
        # Warm: Users present in self.user_map
        # Cold: Users not seen in training

        # Convert input to Series for mapping
        cust_series = pd.Series(customer_ids)
        user_indices = cust_series.map(self.user_id_to_idx)

        # Split indices
        warm_mask = user_indices.notna()
        warm_user_idxs = user_indices[warm_mask].astype(int).values
        warm_customer_ids = cust_series[warm_mask].values

        cold_customer_ids = cust_series[~warm_mask].values

        results = []

        # --- Process Warm Users ---
        if len(warm_user_idxs) > 0:
            print(f"Processing {len(warm_user_idxs)} warm users...")

            # Process in batches to avoid OOM with dense result matrices
            batch_size = 1000
            num_batches = int(np.ceil(len(warm_user_idxs) / batch_size))

            for i in range(num_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, len(warm_user_idxs))

                batch_idxs = warm_user_idxs[start:end]
                batch_cust_ids = warm_customer_ids[start:end]

                # Get history vectors: (Batch, Items)
                # Slicing CSR by rows is efficient
                user_history = self.user_item_matrix[batch_idxs]

                # Compute Scores: (Batch, Items) x (Items, Items) -> (Batch, Items)
                # Result is sparse
                scores = user_history.dot(self.cooccurrence_matrix)

                # Convert to dense for sorting (Top-K)
                # Note: With 100k items, dense row is 400KB. Batch 1000 is 400MB. Safe.
                scores_dense = scores.toarray()

                # Get Top K
                # argpartition is faster than sort
                # We want indices of top K
                # If n_items < k, take all
                n_items = scores_dense.shape[1]
                curr_k = min(k, n_items)

                # argpartition puts top k at the end
                top_k_idx = np.argpartition(scores_dense, -curr_k, axis=1)[:, -curr_k:]

                # Sort the top k (argpartition doesn't sort)
                # We need to extract values to sort
                rows = np.arange(len(batch_idxs))[:, None]
                top_k_scores = scores_dense[rows, top_k_idx]

                # Sort descending
                sort_order = np.argsort(-top_k_scores, axis=1)
                sorted_indices = top_k_idx[rows, sort_order]
                sorted_scores = top_k_scores[rows, sort_order]

                # Flatten and store
                # We need to map item_idx back to article_id
                for r in range(len(batch_idxs)):
                    c_id = batch_cust_ids[r]
                    idxs = sorted_indices[r]
                    scs = sorted_scores[r]

                    # Filter out zero scores if we want?
                    # The ranker might learn from them, but usually we only want retrieved items.
                    # If score is 0, it means no co-occurrence.
                    mask = scs > 0
                    valid_idxs = idxs[mask]
                    valid_scores = scs[mask]

                    # Map back
                    valid_article_ids = [self.idx_to_item_id[idx] for idx in valid_idxs]

                    # Append to results
                    # Structure: customer_id, article_id, score
                    for aid, s in zip(valid_article_ids, valid_scores):
                        results.append((c_id, aid, float(s)))

                    # If fewer than K items found (due to sparsity), fill with global pop
                    if len(valid_article_ids) < k:
                        needed = k - len(valid_article_ids)
                        # Add global pop (excluding already added)
                        # Simple heuristic: just add top global items until full
                        added = 0
                        for gp_item in self.global_popularity:
                            if gp_item not in valid_article_ids:
                                # Assign a small score for global pop or 0?
                                # Let's assign 0 or min_score/2 to indicate low relevance but high pop
                                results.append((c_id, gp_item, 0.0))
                                added += 1
                                if added >= needed:
                                    break

        # --- Process Cold Users ---
        if len(cold_customer_ids) > 0:
            print(
                f"Processing {len(cold_customer_ids)} cold users (Global Popularity fallback)..."
            )
            # Assign global popularity to all cold users
            # Score 0.0
            for c_id in cold_customer_ids:
                # Just take top K global items
                # If global pop list is smaller than K, we just take what we have
                # (Global pop is usually small, e.g. 12, but we can extend it if needed.
                # Here we assume global_popularity list is sufficient or we just provide what we have)
                # The prompt says "Augment... with Global Popularity".
                # We will provide the global popularity list.
                for item in self.global_popularity[:k]:
                    results.append((c_id, item, 0.0))

        # Create DataFrame
        print("Constructing candidates DataFrame...")
        candidates_df = pd.DataFrame(
            results, columns=["customer_id", "article_id", "cooccurrence_score"]
        )

        return candidates_df
