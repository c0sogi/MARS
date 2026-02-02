import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import gc
from sklearn.preprocessing import normalize, minmax_scale
from library.config import Config
from library.data_utils import load_processed_data, get_temporal_view
from library.id_mapper import IdMapper
from library.sparse_engine import SparseEngine


class IGDCRecommender:
    """
    Inventory-Gated Dual-Window Cascade (IGDC) Recommender.

    Implements a three-stage stratified retrieval system:
    1. Habitual Repurchase (Priors Layer)
    2. Inventory-Gated CF (Discovery Layer)
    3. Global Trend (Fallback Layer)
    """

    def __init__(self):
        self.mapper = IdMapper()
        self.engine = SparseEngine()

        # Placeholders for matrices/vectors
        self.S_long = None  # Item-Item Similarity Matrix
        self.M_active = None  # Inventory Mask
        self.R_trend = None  # Global Trend Vector
        self.U_intent = None  # User Intent Matrix (Sparse)
        self.U_habit = None  # User Habit Matrix (Sparse, Weighted)

        # Dimensions
        self.n_users = 0
        self.n_items = 0

    def fit(self, load_cached_data=True):
        """
        Prepares all necessary matrices and vectors for prediction.
        """
        print("Initializing IGDC Recommender...")

        # 1. Load Data & Maps
        # This handles caching of the raw processed data
        transactions, user_map, item_map = load_processed_data(
            load_cached_data=load_cached_data
        )

        # Initialize Mapper
        self.mapper.user_map = user_map
        self.mapper.item_map = item_map
        self.mapper.fit(load_cached_data=True)  # Use the cache we just verified/created

        self.n_users = len(user_map)
        self.n_items = len(item_map)

        print(f"Total Users: {self.n_users}, Total Items: {self.n_items}")

        # 2. Create Temporal Views
        print("Creating temporal views...")
        # Structure View: T-16 weeks to T-1 week
        df_structure = get_temporal_view(
            transactions, Config.WINDOW_STRUCTURE_DAYS, Config.REFERENCE_DATE
        )

        # Intent View: T-2 weeks to T-1 week
        df_intent = get_temporal_view(
            transactions, Config.WINDOW_INTENT_DAYS, Config.REFERENCE_DATE
        )

        # Habit View: T-4 weeks to T
        df_habit = get_temporal_view(
            transactions, Config.WINDOW_HABIT_DAYS, Config.REFERENCE_DATE
        )

        # Inventory/Trend View: T-1 week to T
        df_inventory = get_temporal_view(
            transactions, Config.WINDOW_INVENTORY_DAYS, Config.REFERENCE_DATE
        )

        # Clean up raw transactions to free memory
        del transactions
        gc.collect()

        # 3. Build Global Matrices (Structure, Mask, Trend)
        print("Building global matrices...")

        # S_long: Structure Learning
        self.S_long = self.engine.get_similarity_matrix(
            df_structure, self.n_users, self.n_items, load_cached_data=load_cached_data
        )

        # M_active: Inventory Feasibility
        self.M_active = self.engine.get_inventory_mask(
            df_inventory, self.n_items, load_cached_data=load_cached_data
        )

        # R_trend: Global Fallback
        self.R_trend = self.engine.get_trend_vector(
            df_inventory, self.n_items, load_cached_data=load_cached_data
        )

        # Normalize R_trend to [0, SCORE_TREND_MAX]
        max_trend = self.R_trend.max()
        if max_trend > 0:
            self.R_trend = (self.R_trend / max_trend) * Config.SCORE_TREND_MAX

        # 4. Build User-Specific Matrices (Intent, Habit)
        # We build these as global sparse matrices (N_users x N_items) to allow fast slicing later.
        # Note: We do NOT cache these user matrices to disk in this specific implementation
        # because they are fast to build and large to save, but we could if needed.

        print("Building user intent matrix...")
        # Intent: Simple counts or binary. SparseEngine uses counts.
        self.U_intent = self.engine.build_interaction_matrix(
            df_intent, self.n_users, self.n_items
        )
        # L2 normalize intent rows so dot product is cosine-like
        self.U_intent = normalize(self.U_intent, norm="l2", axis=1)

        print("Building user habit matrix (with decay)...")
        self.U_habit = self._build_weighted_habit_matrix(df_habit)

        print("Fit complete.")

    def _build_weighted_habit_matrix(self, df):
        """
        Constructs the Habit matrix with time decay.
        Weight = 1 / (days_elapsed + 1)
        """
        if df.empty:
            return sp.csr_matrix((self.n_users, self.n_items), dtype=np.float32)

        # Calculate days elapsed
        ref_date = pd.to_datetime(Config.REFERENCE_DATE)
        df["days_elapsed"] = (ref_date - df["t_dat"]).dt.days

        # Calculate weight
        # Ensure no division by zero or negative days (though filtering handles this)
        df["weight"] = 1.0 / (df["days_elapsed"] + 1.0)

        # Aggregate weights for duplicate (user, item) pairs
        # Groupby is safer than implicit summing in coo_matrix construction for control,
        # but coo_matrix sums duplicates by default which is exactly what we want.

        rows = df["user_idx"].values
        cols = df["item_idx"].values
        data = df["weight"].values.astype(np.float32)

        mat = sp.csr_matrix(
            (data, (rows, cols)), shape=(self.n_users, self.n_items), dtype=np.float32
        )
        return mat

    def predict(self, batch_size=5000):
        """
        Generates predictions for all users in the sample submission file.
        Writes directly to the submission file to save memory.
        """
        print("Starting prediction pipeline...")

        # Load sample submission to get target customers
        sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION_CSV)
        target_customers = sub_df["customer_id"].unique()

        # Map to indices
        target_user_idxs = self.mapper.transform(target_customers, "user")

        # Prepare output file
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        out_file = Config.SUBMISSION_FILE

        print(f"Writing predictions to {out_file}...")

        with open(out_file, "w") as f:
            f.write("customer_id,prediction\n")

            # Iterate in batches
            total_users = len(target_user_idxs)
            for start_idx in range(0, total_users, batch_size):
                end_idx = min(start_idx + batch_size, total_users)
                batch_user_idxs = target_user_idxs[start_idx:end_idx]
                batch_customer_ids = target_customers[start_idx:end_idx]

                # --- 1. Stratum 3: Trend (Fallback) ---
                # Base score is the trend vector broadcasted to the batch
                # Shape: (batch_size, n_items)
                # We start with this array.
                batch_scores = np.tile(self.R_trend, (len(batch_user_idxs), 1))

                # --- 2. Stratum 2: Inventory-Gated CF (Discovery) ---
                # Slice Intent Matrix
                U_batch_intent = self.U_intent[batch_user_idxs]

                # Compute CF: (Batch x Items) = (Batch x Items) * (Items x Items)
                # S_long is sparse, U_batch_intent is sparse. Result is dense-ish.
                # We use sparse dot product.
                if U_batch_intent.nnz > 0:
                    cf_scores = U_batch_intent.dot(self.S_long)

                    # Convert to dense if sparse
                    if sp.issparse(cf_scores):
                        cf_scores = cf_scores.toarray()

                    # Apply Inventory Mask
                    # Element-wise multiplication
                    cf_scores = cf_scores * self.M_active

                    # Scale to [100, 1000]
                    # Since U is normalized and S is normalized, max dot product is ~1.0.
                    # We simply scale.
                    cf_scores = (
                        cf_scores * (Config.SCORE_CF_MAX - Config.SCORE_CF_MIN)
                        + Config.SCORE_CF_MIN
                    )

                    # Add to base scores
                    batch_scores += cf_scores

                # --- 3. Stratum 1: Habitual Repurchase (Priors) ---
                # Slice Habit Matrix
                U_batch_habit = self.U_habit[batch_user_idxs]

                if U_batch_habit.nnz > 0:
                    habit_scores = U_batch_habit.toarray()

                    # Apply Offset (> 2000)
                    # We only shift non-zero entries.
                    # Mask for non-zero
                    nonzero_mask = habit_scores > 0
                    habit_scores[nonzero_mask] += Config.SCORE_HABIT_OFFSET

                    # Add to total
                    batch_scores += habit_scores

                # --- 4. Ranking & formatting ---
                # Select Top-K
                # argpartition is faster than sort for top-k
                k = Config.TOP_K_PREDICTIONS

                # We need indices of top k elements.
                # argpartition puts the top k elements at the end (unsorted)
                top_k_part = np.argpartition(batch_scores, -k, axis=1)[:, -k:]

                # To get them sorted, we gather the values and sort
                # Or just sort the top_k indices based on values
                rows = np.arange(len(batch_scores))[:, None]
                top_k_values = batch_scores[rows, top_k_part]

                # Sort descending
                sorter = np.argsort(top_k_values, axis=1)[:, ::-1]
                top_k_indices = top_k_part[rows, sorter]

                # Map back to strings
                # Flatten to vectorize the mapping
                flat_indices = top_k_indices.flatten()
                flat_article_ids = self.mapper.inverse_transform(flat_indices, "item")

                # Reshape back
                batch_article_ids = flat_article_ids.reshape(len(batch_user_idxs), k)

                # Write to file
                for cust_id, preds in zip(batch_customer_ids, batch_article_ids):
                    pred_str = " ".join(map(str, preds))
                    f.write(f"{cust_id},{pred_str}\n")

                if start_idx % 50000 == 0:
                    print(f"Processed {end_idx}/{total_users} users...")

        print("Prediction complete. Submission saved.")

    def run(self):
        """
        Executes the full pipeline.
        """
        self.fit()
        self.predict()
