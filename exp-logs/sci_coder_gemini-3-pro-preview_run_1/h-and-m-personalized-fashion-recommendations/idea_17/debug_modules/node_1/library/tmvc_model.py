import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
import gc
from library.utils import Timer, memory_cleanup
from library.sparse_ops import SparseMatrixOps


class TMVCRecommender:
    """
    Trend-Modulated Vectorized Cascade (TMVC) Recommender.
    Implements a three-stage stratified retrieval system:
    1. Habitual Repurchase (Priors) -> Score > 2000
    2. Trend-Modulated CF (Discovery) -> Score [100, 1000]
    3. Global Trend (Fallback) -> Score [0, 10]
    """

    def __init__(self, cache_dir="./working/idea_17"):
        self.cache_dir = cache_dir
        self.ops = SparseMatrixOps(cache_dir=cache_dir)
        self.S_decay = None
        self.V_trend = None
        self.item_map = None
        self.user_map = None
        self.inv_item_map = None
        self.item_lookup = None

    def fit(
        self,
        df_structure,
        df_velocity,
        df_full_history,
        articles_df,
        customers_df,
        load_cached=True,
    ):
        """
        Constructs the necessary matrices and vectors for the TMVC architecture.

        Args:
            df_structure: DataFrame for Structure Window (e.g., 16 weeks).
            df_velocity: DataFrame for Velocity Window (e.g., 1 week).
            df_full_history: DataFrame for User History (Full).
            articles_df: Master articles dataframe.
            customers_df: Master customers dataframe.
            load_cached: Whether to use cached artifacts.
        """
        with Timer("TMVC Fitting Phase"):
            # 1. Generate Mappings
            (
                self.item_map,
                self.user_map,
                self.inv_item_map,
                _,
            ) = self.ops.get_mappings(articles_df, customers_df, load_cached)

            # 2. Build Structure Matrix (S_decay)
            # Use sqrt decay for structure learning as per design
            X_structure = self.ops.build_decayed_interaction_matrix(
                df_structure,
                self.item_map,
                self.user_map,
                decay_strategy="sqrt",
                load_cached=load_cached,
                cache_name="structure",
            )

            self.S_decay = self.ops.compute_cosine_similarity(
                X_structure, top_k=100, load_cached=load_cached, cache_name="S_decay"
            )

            del X_structure
            memory_cleanup()

            # 3. Build Velocity Vector (V_trend)
            v_path = os.path.join(self.cache_dir, "V_trend.npy")
            if load_cached and os.path.exists(v_path):
                print(f"Loading cached Velocity Vector from {v_path}")
                self.V_trend = np.load(v_path)
            else:
                print("Computing Velocity Vector...")
                # Count sales per article in velocity window
                counts = df_velocity["article_id"].value_counts()

                # Initialize dense vector
                n_items = len(self.item_map)
                v_vec = np.zeros(n_items, dtype=np.float32)

                # Map counts to indices
                # Iterate is fast enough for 100k items
                for art_id, count in counts.items():
                    if art_id in self.item_map:
                        idx = self.item_map[art_id]
                        v_vec[idx] = count

                # Log transform: log(1 + count)
                self.V_trend = np.log1p(v_vec)
                np.save(v_path, self.V_trend)

            # 4. Pre-build Full History Matrix (U_intent / U_habit)
            # This is used in predict, but we trigger build here to ensure cache exists
            # Use linear decay (1/t) for aggressive history weighting
            self.ops.build_decayed_interaction_matrix(
                df_full_history,
                self.item_map,
                self.user_map,
                decay_strategy="linear",
                load_cached=load_cached,
                cache_name="history_full",
            )

            # Prepare Item Lookup Array for fast inference mapping
            max_idx = max(self.inv_item_map.keys())
            self.item_lookup = np.empty(max_idx + 1, dtype=object)
            for k_idx, v_art in self.inv_item_map.items():
                self.item_lookup[k_idx] = v_art

    def predict(self, test_df, df_full_history, batch_size=5000):
        """
        Generates predictions for the customers in test_df.

        Args:
            test_df: DataFrame containing 'customer_id' column.
            df_full_history: Used to load the history matrix (via cache).
            batch_size: Number of users to process at once.

        Returns:
            pd.DataFrame: Submission dataframe with 'customer_id' and 'prediction'.
        """
        if self.S_decay is None or self.V_trend is None:
            raise ValueError("Model must be fitted before prediction.")

        # --- 1. Global Trend Preparation ---
        # Scale V_trend to [0, 10] for Stratum 3
        max_v = self.V_trend.max()
        if max_v > 0:
            global_trend_score = (self.V_trend / max_v) * 10.0
        else:
            global_trend_score = self.V_trend

        # Pre-calculate top global items for cold-start users
        top_global_indices = np.argsort(-global_trend_score)[:12]
        top_global_items = self.item_lookup[top_global_indices]
        top_global_str = " ".join(top_global_items)

        # --- 2. User Preparation ---
        # Map test customers to indices
        # We need to handle users who might not be in the map (cold start)
        test_df = test_df.copy()
        test_df["user_idx"] = test_df["customer_id"].map(self.user_map)

        # Split into known and unknown
        known_mask = test_df["user_idx"].notna()
        known_users_df = test_df[known_mask]
        unknown_users_df = test_df[~known_mask]

        # Load Full History Matrix
        # This returns the matrix for ALL users in training
        X_history = self.ops.build_decayed_interaction_matrix(
            df_full_history,
            self.item_map,
            self.user_map,
            decay_strategy="linear",
            load_cached=True,
            cache_name="history_full",
        )

        # --- 3. Batch Inference for Known Users ---
        predictions = []
        customer_ids = []

        # Get indices as integers
        target_indices = known_users_df["user_idx"].astype(int).values
        target_cust_ids = known_users_df["customer_id"].values
        n_targets = len(target_indices)

        print(
            f"Predicting for {n_targets} known users and {len(unknown_users_df)} cold-start users..."
        )

        with Timer("Batch Inference"):
            for start_idx in range(0, n_targets, batch_size):
                end_idx = min(start_idx + batch_size, n_targets)

                # Current batch indices (global user indices)
                batch_u_indices = target_indices[start_idx:end_idx]
                batch_cust_ids = target_cust_ids[start_idx:end_idx]
                current_batch_size = len(batch_u_indices)

                # Slice history matrix for this batch
                # Shape: (Batch_Size, N_Items)
                U_batch = X_history[batch_u_indices]

                # --- Stratum 1: Habit (Priors) ---
                # Score > 2000.
                # U_batch has weights in (0, 1].
                # We shift them by 2000.
                # We keep this sparse for efficient addition later.
                R_habit = U_batch.copy()
                R_habit.data += 2000.0

                # --- Stratum 2: CF (Discovery) ---
                # R_cf = (U_batch @ S_decay) * V_trend
                # Result is dense (Batch_Size, N_Items)
                R_cf = U_batch @ self.S_decay

                # Convert to dense array
                R_cf = R_cf.toarray()

                # Modulate with Velocity (Element-wise multiplication)
                # Broadcast V_trend (N_Items,) across batch
                R_cf *= self.V_trend[None, :]

                # Normalize to [100, 1000]
                # Find max per row
                row_max = R_cf.max(axis=1, keepdims=True)
                row_max[row_max == 0] = 1.0  # Prevent division by zero

                # Scale: 100 + (val / max) * 900
                R_cf = (R_cf / row_max) * 900.0 + 100.0

                # --- Stratum 3: Global Trend ---
                # Add global score to all items
                # R_total = R_cf + Global
                R_total = R_cf + global_trend_score[None, :]

                # --- Aggregation ---
                # Add Habit scores (Sparse + Dense)
                # We iterate the sparse matrix to add values to the dense matrix
                coo_habit = R_habit.tocoo()
                # coo_habit.row is relative to the batch (0..batch_size-1)
                R_total[coo_habit.row, coo_habit.col] += coo_habit.data

                # --- Retrieval ---
                # Find top 12
                k = 12
                # argpartition puts top k at the end
                top_k_idx = np.argpartition(R_total, -k, axis=1)[:, -k:]

                # Sort the top k
                row_idx = np.arange(current_batch_size)[:, None]
                top_k_vals = R_total[row_idx, top_k_idx]

                # Argsort returns indices that sort the array
                # We want descending order
                sort_ord = np.argsort(-top_k_vals, axis=1)
                final_idx = top_k_idx[row_idx, sort_ord]

                # Map to Article IDs
                batch_preds = self.item_lookup[final_idx]

                # Format strings
                batch_pred_strings = [" ".join(row) for row in batch_preds]

                predictions.extend(batch_pred_strings)
                customer_ids.extend(batch_cust_ids)

                # Cleanup memory
                del R_total, R_cf, R_habit, U_batch
                # memory_cleanup() # Optional: calling too often slows down loop

        # --- 4. Handle Cold Start Users ---
        if len(unknown_users_df) > 0:
            unknown_ids = unknown_users_df["customer_id"].values
            unknown_preds = [top_global_str] * len(unknown_ids)

            predictions.extend(unknown_preds)
            customer_ids.extend(unknown_ids)

        # --- 5. Finalize Submission ---
        sub_df = pd.DataFrame({"customer_id": customer_ids, "prediction": predictions})

        return sub_df
