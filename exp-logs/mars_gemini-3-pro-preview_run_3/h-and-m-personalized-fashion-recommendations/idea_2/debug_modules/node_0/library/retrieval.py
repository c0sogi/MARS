import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import gc
from tqdm import tqdm
import library.config as config
import library.data_loader as data_loader


class SparseGraphRetriever:
    def __init__(
        self,
        decay_rate=config.DECAY_RATE,
        history_weight=config.HISTORY_WEIGHT,
        top_k=config.TOP_K_CANDIDATES,
    ):
        self.decay_rate = decay_rate
        self.history_weight = history_weight
        self.top_k = top_k
        self.article_map = None  # article_id -> int index
        self.reverse_article_map = None  # int index -> article_id
        self.transition_matrix = None  # Sparse matrix T
        self.n_articles = 0

    def _load_article_map(self, load_cached_data=True):
        """
        Ensures article mapping exists. Loads from cache or creates it.
        """
        if self.article_map is not None:
            return

        cache_path = config.CACHE_ARTICLE_MAP

        if load_cached_data and cache_path.exists():
            print(f"Loading article map from {cache_path}")
            self.article_map = np.load(cache_path, allow_pickle=True).item()
        else:
            print("Creating article map from articles.csv...")
            # Load all articles to ensure fixed dimension
            articles_df = pd.read_csv(config.ARTICLES_CSV)
            unique_articles = articles_df[config.ITEM_COL].unique()
            self.article_map = {aid: i for i, aid in enumerate(unique_articles)}

            # Save cache
            os.makedirs(config.WORKING_DIR, exist_ok=True)
            np.save(cache_path, self.article_map)

        self.reverse_article_map = {v: k for k, v in self.article_map.items()}
        self.n_articles = len(self.article_map)
        print(f"Total articles in map: {self.n_articles}")

    def fit(self, df_train, load_cached_data=True):
        """
        Builds the transition matrix T from training data.

        Args:
            df_train (pd.DataFrame): Transaction history.
            load_cached_data (bool): Whether to load T from disk if available.
        """
        self._load_article_map(load_cached_data=load_cached_data)

        cache_path = config.CACHE_TRANSITION_MATRIX

        if load_cached_data and cache_path.exists():
            print(f"Loading transition matrix from {cache_path}")
            self.transition_matrix = sp.load_npz(cache_path)
            return

        print("Building transition matrix (this may take a while)...")

        # Ensure data is sorted
        df = df_train.sort_values([config.USER_COL, config.DATE_COL]).copy()

        # Map article_ids to indices
        # Filter out articles not in our map (if any)
        df = df[df[config.ITEM_COL].isin(self.article_map)]
        df["aid_idx"] = df[config.ITEM_COL].map(self.article_map).astype(np.int32)

        # Create shifted columns to identify transitions
        # We want transition: Item(t) -> Item(t+1)
        df["next_aid_idx"] = df["aid_idx"].shift(-1)
        df["next_user"] = df[config.USER_COL].shift(-1)
        df["next_date"] = df[config.DATE_COL].shift(-1)

        # Filter valid transitions (same user)
        # The last row of a user will have next_user != user (or NaN), so it gets filtered
        valid_mask = df[config.USER_COL] == df["next_user"]
        transitions = df[valid_mask].copy()

        # Calculate weight based on time decay
        # weight = 1 / (1 + decay * days_diff)
        transitions["days_diff"] = (
            transitions["next_date"] - transitions[config.DATE_COL]
        ).dt.days
        transitions["weight"] = 1.0 / (1.0 + self.decay_rate * transitions["days_diff"])

        # Aggregate weights for duplicate transitions (i -> j)
        # Group by (aid_idx, next_aid_idx) and sum weights
        print("Aggregating transition weights...")
        trans_grouped = (
            transitions.groupby(["aid_idx", "next_aid_idx"])["weight"]
            .sum()
            .reset_index()
        )

        # Build Sparse Matrix
        print("Constructing CSR matrix...")
        row = trans_grouped["aid_idx"].values
        col = trans_grouped["next_aid_idx"].values.astype(np.int32)  # ensure int index
        data = trans_grouped["weight"].values

        self.transition_matrix = sp.csr_matrix(
            (data, (row, col)),
            shape=(self.n_articles, self.n_articles),
            dtype=np.float32,
        )

        # Save to cache
        print(f"Saving transition matrix to {cache_path}")
        sp.save_npz(cache_path, self.transition_matrix)

        # Cleanup
        del df, transitions, trans_grouped
        gc.collect()

    def query(self, df_history, target_users):
        """
        Generates candidates for a list of target users.

        Args:
            df_history (pd.DataFrame): Full history used to build User vectors.
            target_users (array-like): List of customer_ids to predict for.

        Returns:
            pd.DataFrame: Candidates in long format [customer_id, article_id, score, rank]
        """
        if self.transition_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")

        print(f"Generating candidates for {len(target_users)} users...")

        # 1. Build User-Item History Matrix (U) for target users
        # Map target users to temporary index 0..M-1
        user_map = {uid: i for i, uid in enumerate(target_users)}

        # Filter history for these users
        relevant_history = df_history[df_history[config.USER_COL].isin(user_map)].copy()

        # Map IDs
        relevant_history = relevant_history[
            relevant_history[config.ITEM_COL].isin(self.article_map)
        ]
        relevant_history["user_idx"] = relevant_history[config.USER_COL].map(user_map)
        relevant_history["aid_idx"] = relevant_history[config.ITEM_COL].map(
            self.article_map
        )

        # Deduplicate (User, Item) pairs for binary history vector
        # Or we can use count. Let's use binary presence for U as per standard graph prop.
        # However, to support 'history_weight', we might want recent items to have initial weight.
        # For simplicity and robustness: Binary U.
        relevant_history = relevant_history.drop_duplicates(
            subset=["user_idx", "aid_idx"]
        )

        # Create Sparse U (Rows: Target Users, Cols: Articles)
        row_u = relevant_history["user_idx"].values
        col_u = relevant_history["aid_idx"].values
        data_u = np.ones(len(row_u), dtype=np.float32)

        U = sp.csr_matrix(
            (data_u, (row_u, col_u)),
            shape=(len(target_users), self.n_articles),
            dtype=np.float32,
        )

        # 2. Propagation: S = U * T
        print("Performing sparse propagation...")
        S = U.dot(self.transition_matrix)

        # 3. Add History Bias: S += alpha * U
        # Note: U is binary. If we want to boost repurchases, we add alpha * U.
        if self.history_weight > 0:
            print("Adding history bias...")
            S = S + (U * self.history_weight)

        # 4. Extract Top-K
        # S is a CSR matrix. We need to extract top-K indices per row efficiently.
        print("Extracting top-K candidates...")

        # Result containers
        res_users = []
        res_items = []
        res_scores = []
        res_ranks = []

        # Iterate over rows
        # Using tqdm for progress tracking
        indptr = S.indptr
        indices = S.indices
        data = S.data

        for i in tqdm(range(len(target_users)), mininterval=5.0):
            # Get row slice
            start = indptr[i]
            end = indptr[i + 1]

            if start == end:
                # No candidates (empty history or no transitions)
                continue

            row_indices = indices[start:end]
            row_data = data[start:end]

            # If we have fewer than K items, take all
            k = min(len(row_data), self.top_k)

            # Get top K
            if len(row_data) <= self.top_k:
                top_k_idx = np.argsort(row_data)[::-1]
            else:
                # argpartition puts top k at the end, unsorted
                unsorted_top_k = np.argpartition(row_data, -k)[-k:]
                # sort them
                sorted_top_k = unsorted_top_k[
                    np.argsort(row_data[unsorted_top_k])[::-1]
                ]
                top_k_idx = sorted_top_k

            # Map back to real IDs
            best_indices = row_indices[top_k_idx]
            best_scores = row_data[top_k_idx]

            # Store
            current_user_id = target_users[i]

            # Append to lists (batching is faster than row-by-row append to DF)
            # We repeat user_id k times
            res_users.extend([current_user_id] * len(best_indices))

            # Convert article indices back to article_ids
            # We can do this in bulk later, but here we need to store indices first
            res_items.extend(best_indices)
            res_scores.extend(best_scores)
            res_ranks.extend(range(1, len(best_indices) + 1))

        # 5. Construct DataFrame
        print("Constructing candidate DataFrame...")
        candidates_df = pd.DataFrame(
            {
                config.USER_COL: res_users,
                "aid_idx": res_items,
                "retrieval_score": res_scores,
                "rank": res_ranks,
            }
        )

        # Map aid_idx back to article_id
        # It's faster to map the whole column once
        # Create a vector for reverse mapping
        # reverse_map_vec = np.empty(self.n_articles, dtype=object) # article_id is int64 usually
        # But here article_id in config is likely int64.

        # Let's use map with the dict
        candidates_df[config.ITEM_COL] = candidates_df["aid_idx"].map(
            self.reverse_article_map
        )

        # Drop internal index
        candidates_df.drop(columns=["aid_idx"], inplace=True)

        return candidates_df

    def get_global_popularity_fallback(self, df_train, n=12):
        """
        Returns top n popular items from training data.
        Useful for cold-start users.
        """
        pop = df_train[config.ITEM_COL].value_counts().head(n).index.tolist()
        return " ".join(map(str, pop))
