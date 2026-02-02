import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import gc
from sklearn.preprocessing import minmax_scale
from library import config, data_factory, graph_engine


class TGSCRecommender:
    """
    Temporal-Graph Stratified Cascade (TGSC) Recommender.
    Implements a 3-stage stratified inference model:
    1. Habit (Repurchase) > 2000
    2. Collaborative Filtering (Time-Embedded) [100, 1000]
    3. Global Trend [0, 10]
    """

    def __init__(self, user_map, item_map, similarity_matrix):
        self.user_map = user_map
        self.item_map = item_map
        self.similarity_matrix = similarity_matrix

        # Reverse map for decoding predictions
        self.id_to_article = {v: k for k, v in item_map.items()}

        self.n_users = len(user_map)
        self.n_items = len(item_map)

        # Placeholders
        self.global_trends = None
        self.history_matrix = (
            None  # CSR Matrix: Rows=Users, Cols=Items, Data=MinDaysElapsed+1
        )

    def fit(self, train_df, load_cached_data=True):
        """
        Prepares global trends and user history index.
        """
        print("Fitting TGSC Recommender...")

        # 1. Compute Global Trends
        self._compute_global_trends(train_df, load_cached_data)

        # 2. Prepare User History Index
        self._prepare_history_index(train_df, load_cached_data)

    def _compute_global_trends(self, df, load_cached_data):
        cache_path = config.CACHE_GLOBAL_TRENDS

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading global trends from {cache_path}...")
            self.global_trends = np.load(cache_path)
            return

        print("Computing global trends (Time-Decayed Sales Velocity)...")
        # Map article_ids to indices
        # We assume df has 'article_id' as original strings, need to map to item_idx
        # However, for efficiency, we filter to items in our map
        valid_items = df[df["article_id"].isin(self.item_map)]

        # Create a temporary dataframe for calculation
        tmp = valid_items[["article_id", "days_elapsed"]].copy()
        tmp["item_idx"] = tmp["article_id"].map(self.item_map)

        # Weight = exp(-lambda * days)
        # We use a slightly more aggressive decay for trends to capture "Right Now" popularity
        decay = config.DECAY_RATE
        tmp["weight"] = np.exp(-decay * tmp["days_elapsed"])

        # Aggregate
        trend_scores = np.zeros(self.n_items, dtype=np.float32)
        agg = tmp.groupby("item_idx")["weight"].sum()
        trend_scores[agg.index] = agg.values

        # Scale to [0, 10]
        # Handle case where all zero
        if trend_scores.max() > 0:
            trend_scores = minmax_scale(trend_scores, feature_range=(0, 10))

        self.global_trends = trend_scores.astype(np.float32)

        print(f"Saving global trends to {cache_path}...")
        np.save(cache_path, self.global_trends)

    def _prepare_history_index(self, df, load_cached_data):
        """
        Builds a CSR matrix where A[u, i] = min_days_elapsed + 1.
        The +1 allows us to store 0-day latency as 1, preserving sparsity (0 = no history).
        """
        cache_path = config.CACHE_USER_HISTORY

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading user history index from {cache_path}...")
            # We save as parquet for portability, but load into sparse
            hist_df = pd.read_parquet(cache_path)
            row = hist_df["user_idx"].values
            col = hist_df["item_idx"].values
            data = hist_df["days_shifted"].values
            self.history_matrix = sp.csr_matrix(
                (data, (row, col)), shape=(self.n_users, self.n_items), dtype=np.float32
            )
            return

        print("Building user history index...")
        # Filter for known users/items
        valid_mask = (df["customer_id"].isin(self.user_map)) & (
            df["article_id"].isin(self.item_map)
        )
        tmp = df.loc[valid_mask, ["customer_id", "article_id", "days_elapsed"]].copy()

        tmp["user_idx"] = tmp["customer_id"].map(self.user_map)
        tmp["item_idx"] = tmp["article_id"].map(self.item_map)

        # Keep most recent purchase (min days_elapsed) per user-item pair
        # This is critical for the "Habit" score
        grp = tmp.groupby(["user_idx", "item_idx"])["days_elapsed"].min().reset_index()

        # Shift days by +1 so 0 (today) becomes 1, and 0 in matrix means "no interaction"
        grp["days_shifted"] = grp["days_elapsed"] + 1.0

        # Create CSR
        row = grp["user_idx"].values
        col = grp["item_idx"].values
        data = grp["days_shifted"].values.astype(np.float32)

        self.history_matrix = sp.csr_matrix(
            (data, (row, col)), shape=(self.n_users, self.n_items), dtype=np.float32
        )

        print(f"Saving user history index to {cache_path}...")
        grp.to_parquet(cache_path, index=False)

    def predict(self, customer_ids_to_predict):
        """
        Generates predictions for the given list of customer_ids.
        Returns a DataFrame with columns ['customer_id', 'prediction'].
        """
        print(f"Generating predictions for {len(customer_ids_to_predict)} users...")

        preds = []
        batch_size = config.BATCH_SIZE
        n_users = len(customer_ids_to_predict)

        # Process in batches
        for start_idx in range(0, n_users, batch_size):
            end_idx = min(start_idx + batch_size, n_users)
            batch_ids = customer_ids_to_predict[start_idx:end_idx]
            current_batch_size = len(batch_ids)

            # Initialize scores with Global Trend (Stratum 3)
            # Shape: (Batch, Items). We tile the global trend vector.
            # Memory: 10k * 100k * 4 bytes ~= 4GB. Acceptable.
            batch_scores = np.tile(self.global_trends, (current_batch_size, 1))
            batch_scores += config.OFFSET_TREND

            # Identify known users
            known_mask = np.array([uid in self.user_map for uid in batch_ids])
            known_indices = np.where(known_mask)[0]

            if len(known_indices) > 0:
                # Get user indices for the known users in this batch
                known_uids = [batch_ids[i] for i in known_indices]
                user_indices = np.array([self.user_map[uid] for uid in known_uids])

                # Retrieve History
                # Slice the history matrix
                u_hist = self.history_matrix[user_indices]

                # --- Stratum 1: Habit (Repurchase) ---
                # Score = Offset + 1 / days_elapsed
                # u_hist contains (days + 1). So we want 1 / (data - 1 + 1). No, just 1/data is fine approx.
                # Exact prompt requirement: 1 / days_elapsed.
                # Our data is days + 1. So days = data - 1.
                # To avoid div by zero if days=0, we use 1 / (days + 1) = 1 / data.
                # This decays: Day 0 -> 1.0, Day 1 -> 0.5, etc.
                habit_scores = u_hist.copy()
                habit_scores.data = config.OFFSET_HABIT + (1.0 / habit_scores.data)

                # Add to dense batch scores
                # We only update the rows corresponding to known users
                # .toarray() converts sparse slice to dense
                batch_scores[known_indices] += habit_scores.toarray()

                # --- Stratum 2: CF (Time-Embedded) ---
                # Query Vector: exp(-lambda * days)
                # u_hist data is (days + 1).
                query_vec = u_hist.copy()
                # Recover days: data - 1
                days = query_vec.data - 1.0
                query_vec.data = np.exp(-config.DECAY_RATE * days)

                # Matrix Multiplication: U_query @ S
                cf_raw = query_vec.dot(self.similarity_matrix)

                # Row-wise Normalization & Scaling [100, 1000]
                # Convert to dense for normalization
                cf_dense = cf_raw.toarray()

                # Compute row maxs for normalization
                row_maxs = cf_dense.max(axis=1, keepdims=True)
                row_maxs[row_maxs == 0] = 1.0  # Avoid div by zero

                cf_norm = cf_dense / row_maxs

                # Scale
                cf_final = config.OFFSET_CF + (cf_norm * (1000.0 - 100.0))

                # Add to batch scores
                batch_scores[known_indices] += cf_final

            # --- Retrieval ---
            # Select top K
            k = config.TOP_K_PREDICTIONS
            # argpartition puts the k-th largest element at index -k
            # and all larger elements to the right
            top_k_idx = np.argpartition(batch_scores, -k, axis=1)[:, -k:]

            # Sort the top k (argpartition doesn't guarantee order)
            # We take the scores of the top k indices
            rows = np.arange(current_batch_size)[:, None]
            top_k_scores = batch_scores[rows, top_k_idx]

            # Sort indices based on scores (descending)
            sort_order = np.argsort(-top_k_scores, axis=1)
            sorted_top_k_idx = top_k_idx[rows, sort_order]

            # Map back to strings
            for i in range(current_batch_size):
                item_indices = sorted_top_k_idx[i]
                art_ids = [self.id_to_article[idx] for idx in item_indices]
                preds.append(" ".join(map(str, art_ids)))

            # Explicit GC
            del batch_scores

            if (start_idx // batch_size) % 5 == 0:
                print(f"Processed {end_idx}/{n_users} users...")
                gc.collect()

        return pd.DataFrame(
            {"customer_id": customer_ids_to_predict, "prediction": preds}
        )


def generate_submission():
    """
    Main entry point to generate the submission file.
    """
    print("Starting submission generation...")

    # 1. Load Data
    # We combine Train and Val to get the maximum history for the final model
    print("Loading datasets...")
    df_train = data_factory.load_and_preprocess(config.TRAIN_META_PATH)
    df_val = data_factory.load_and_preprocess(config.VAL_META_PATH)
    df_full = pd.concat([df_train, df_val], ignore_index=True)

    # Load Test Users
    df_test = pd.read_csv(config.TEST_META_PATH)
    test_users = df_test["customer_id"].tolist()

    # 2. Build/Load Graph Artifacts
    user_map, item_map = graph_engine.get_mappings(df_full)

    # Interaction Matrix (needed for Similarity)
    interaction_matrix = graph_engine.build_decayed_interaction_matrix(
        df_full, user_map, item_map
    )

    # Similarity Matrix
    similarity_matrix = graph_engine.compute_similarity_matrix(interaction_matrix)

    # 3. Initialize and Fit Recommender
    recommender = TGSCRecommender(user_map, item_map, similarity_matrix)
    recommender.fit(df_full)

    # 4. Generate Predictions
    submission_df = recommender.predict(test_users)

    # 5. Save
    out_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    print(f"Saving submission to {out_path}...")
    submission_df.to_csv(out_path, index=False)
    print("Done.")
