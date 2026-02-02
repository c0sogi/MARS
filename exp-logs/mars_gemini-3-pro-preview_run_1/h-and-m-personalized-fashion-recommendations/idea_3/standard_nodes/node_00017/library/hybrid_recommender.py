import os
import gc
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

from library.config import Config
from library.data_utils import (
    get_global_mapper,
    load_and_filter_data,
    build_user_history_vectors,
)
from library.visual_features import compute_visual_similarity_matrix
from library.collaborative_filtering import (
    compute_behavioral_similarity_matrix,
    calculate_global_trend,
)


class HybridRecommender:
    """
    Implements the Hybrid Multi-View Similarity Ensemble.
    Aggregates signals from Repurchase (Habit), Behavioral Co-occurrence,
    Visual Similarity, and Global Trends using a weighted linear combination.
    """

    def __init__(self, load_cached_data=True, train_weeks=Config.TRAIN_WEEKS):
        """
        Initializes the recommender by loading/computing all necessary matrices.

        Args:
            load_cached_data (bool): If True, attempts to load matrices from disk cache.
            train_weeks (int): Number of weeks of transaction history to use.
        """
        print("Initializing HybridRecommender...")

        # 1. Setup Mapper (Global Universe of Users/Items)
        self.mapper = get_global_mapper()

        # 2. Load Training Data
        # We load this once and pass it to the component builders.
        # This ensures all components use the exact same data split/window.
        print(f"Loading training data (last {train_weeks} weeks)...")
        self.train_df = load_and_filter_data(Config.PATH_TRAIN, weeks=train_weeks)

        # 3. Build/Load Ensemble Components

        # View 1: Habit (Repurchase)
        # Sparse Matrix: (N_Users, N_Items)
        self.U_hist = build_user_history_vectors(
            self.train_df, self.mapper, load_cached_data=load_cached_data
        )

        # View 2: Behavioral Similarity (Co-occurrence)
        # Sparse Matrix: (N_Items, N_Items)
        self.S_behavior = compute_behavioral_similarity_matrix(
            self.train_df, self.mapper, load_cached_data=load_cached_data
        )

        # View 3: Visual Similarity (Content)
        # Sparse Matrix: (N_Items, N_Items)
        # Only compute if weight > 0 to save resources (Cite solution_lesson_node_00016)
        if Config.WEIGHT_GAMMA > 0:
            self.S_visual = compute_visual_similarity_matrix(
                self.mapper, load_cached_data=load_cached_data
            )
        else:
            self.S_visual = None

        # View 4: Global Trend (Popularity)
        # Dense Vector: (N_Items,)
        self.V_trend = calculate_global_trend(
            self.train_df, self.mapper, load_cached_data=load_cached_data
        )

        # 4. Set Ensemble Weights
        self.w_rep = Config.WEIGHT_ALPHA  # Alpha: Repurchase
        self.w_beh = Config.WEIGHT_BETA  # Beta: Behavioral
        self.w_vis = Config.WEIGHT_GAMMA  # Gamma: Visual
        self.w_trend = Config.WEIGHT_DELTA  # Delta: Trend

        print(
            f"Ensemble Weights initialized: "
            f"Repurchase={self.w_rep}, Behavioral={self.w_beh}, "
            f"Visual={self.w_vis}, Trend={self.w_trend}"
        )

        # 5. Cleanup
        # We no longer need the raw dataframe, as matrices are built.
        del self.train_df
        gc.collect()

    def predict_scores(self, user_indices):
        """
        Computes the dense score matrix for a batch of users.

        Formula:
        Score = w_trend * V_trend
              + w_rep * U_hist
              + w_beh * (U_hist @ S_beh)
              + w_vis * (U_hist @ S_vis)

        Args:
            user_indices (np.array): Array of valid user indices (0..N-1).

        Returns:
            np.array: Dense score matrix of shape (Batch_Size, N_Items).
        """
        batch_size = len(user_indices)

        # Extract User History for this batch (Batch, N_Items) - Sparse
        U_batch = self.U_hist[user_indices]

        # Initialize dense scores with Global Trend
        # Broadcast V_trend (N_Items,) to (Batch, N_Items)
        # We use np.tile or broadcasting. Broadcasting is implicit in addition usually,
        # but for initialization we need the shape.
        # V_trend is (N_Items,), we want (Batch, N_Items).
        scores = np.tile(self.V_trend, (batch_size, 1)) * self.w_trend

        # Add Repurchase Signal (Habit)
        if self.w_rep > 0:
            # U_batch is sparse. We multiply by weight and densify.
            # Note: Densifying a very sparse matrix is fast if batch size is reasonable.
            scores += U_batch.multiply(self.w_rep).toarray()

        # Add Behavioral Signal (Collaborative Filtering)
        if self.w_beh > 0:
            # Sparse @ Sparse -> Sparse
            R_beh = U_batch.dot(self.S_behavior)
            scores += R_beh.multiply(self.w_beh).toarray()

        # Add Visual Signal (Content-Based)
        if self.w_vis > 0:
            # Sparse @ Sparse -> Sparse
            R_vis = U_batch.dot(self.S_visual)
            scores += R_vis.multiply(self.w_vis).toarray()

        return scores

    def generate_submission(self, output_path=Config.PATH_SUBMISSION, batch_size=2000):
        """
        Generates predictions for all users in the sample submission file.
        Writes the result to a CSV file.

        Args:
            output_path (str): Path to save the submission CSV.
            batch_size (int): Number of users to process at once.
        """
        print(f"Generating submission to {output_path}...")

        # 1. Load Sample Submission to get exact customer list
        sample_sub = pd.read_csv(Config.PATH_SAMPLE_SUBMISSION)
        customer_ids = sample_sub["customer_id"].values
        num_users = len(customer_ids)

        print(f"Total customers to predict: {num_users}")

        # 2. Map Customer IDs to Indices
        # map_users returns -1 for unknown users
        all_user_indices = self.mapper.map_users(pd.Series(customer_ids))

        # 3. Prepare Output
        predictions = []

        # 4. Batch Inference
        for start_idx in tqdm(range(0, num_users, batch_size), desc="Predicting"):
            end_idx = min(start_idx + batch_size, num_users)

            # Get indices for current batch
            batch_u_inds = all_user_indices[start_idx:end_idx]
            current_batch_size = len(batch_u_inds)

            # Identify valid users (those who exist in our universe)
            # Users with index -1 are unknown (Cold Start)
            valid_mask = batch_u_inds >= 0

            # Initialize batch scores container
            batch_scores = np.zeros(
                (current_batch_size, self.mapper.get_num_items()), dtype=np.float32
            )

            # Compute scores for valid users
            if valid_mask.any():
                valid_indices = batch_u_inds[valid_mask]
                batch_scores[valid_mask] = self.predict_scores(valid_indices)

            # Handle completely unknown users (index -1)
            # They get pure Global Trend
            if (~valid_mask).any():
                trend_scores = self.V_trend * self.w_trend
                batch_scores[~valid_mask] = trend_scores

            # 5. Retrieve Top-12 Items
            n_preds = Config.TOP_N_PREDICTIONS

            # np.argpartition puts the top K items at the end of the array (unsorted)
            # We select the indices of these top K items
            # axis=1 performs this row-wise
            top_k_indices = np.argpartition(batch_scores, -n_preds, axis=1)[
                :, -n_preds:
            ]

            # Sort the top K items by score descending
            # We fetch the actual scores for these indices to sort them
            rows = np.arange(current_batch_size)[:, None]
            top_k_scores = batch_scores[rows, top_k_indices]

            # argsort gives indices relative to the top_k subarray
            sort_inds = np.argsort(-top_k_scores, axis=1)

            # Map back to global item indices
            final_indices = top_k_indices[rows, sort_inds]

            # 6. Convert Indices to Article IDs (Strings)
            # Flatten to map efficiently
            flat_indices = final_indices.flatten()
            flat_article_ids = self.mapper.get_items_from_indices(flat_indices)

            # Reshape back to (Batch, 12)
            batch_article_ids = flat_article_ids.reshape(current_batch_size, n_preds)

            # Format as strings
            for row in batch_article_ids:
                # Format: "0123456789 0987654321 ..."
                # article_ids are int32, need zfill(10)
                pred_str = " ".join([f"{aid:010d}" for aid in row])
                predictions.append(pred_str)

        # 7. Save Submission
        submission_df = pd.DataFrame(
            {"customer_id": customer_ids, "prediction": predictions}
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved successfully to {output_path}")
