import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import gc

from library import settings
from library.data_manager import TransactionLoader
from library.graph_model import InteractionGraph
from library.predictor import StratifiedRecommender
from library.metrics import apk


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # Initialize
    set_seed(settings.RANDOM_SEED)
    print("Starting TWIG-SR Pipeline Execution...")

    # =========================================================================
    # Phase 1: Validation and Failure Analysis
    # =========================================================================
    print("\n=== Phase 1: Validation & Failure Analysis ===")

    # 1. Load Data (Validation Split)
    # This splits data into Train (T-10w to T-1w) and Val (T)
    loader = TransactionLoader()
    train_df, val_df, user_map, item_map = loader.get_data(
        validation=True, load_cached_data=True
    )

    n_users = len(user_map)
    n_items = len(item_map)

    # 2. Build Interaction Graph
    # Constructs X (User-Item) and S (Item-Item) matrices
    graph = InteractionGraph(n_users, n_items)
    graph.build(train_df, load_cached_data=True)
    X, S = graph.get_matrices()

    # 3. Prepare Auxiliary Signals (Habit & Trends)
    # We use the logic from StratifiedRecommender but run it manually to inspect results
    temp_recommender = StratifiedRecommender()
    global_trends = temp_recommender._compute_global_trends(
        train_df, n_items, load_cached_data=True
    )
    H = temp_recommender._build_habit_matrix(train_df, n_users, n_items)

    # 4. Prepare Ground Truth for Evaluation
    # Group validation items by user_idx: {user_idx: [item_id1, item_id2, ...]}
    val_ground_truth = val_df.groupby("user_idx")["item_idx"].apply(list).to_dict()
    target_users = np.array(list(val_ground_truth.keys()))

    print(f"Validating on {len(target_users)} users...")

    # 5. Inference Loop (Manual Execution for Analysis)
    batch_size = 2000
    ap_scores = []
    user_indices = []

    # Hyperparameters from settings
    offset_cf_min = settings.CF_OFFSET_MIN
    offset_cf_max = settings.CF_OFFSET_MAX
    top_k = 12

    total_users = len(target_users)

    # Process in batches
    for start_idx in range(0, total_users, batch_size):
        end_idx = min(start_idx + batch_size, total_users)
        batch_users = target_users[start_idx:end_idx]
        current_batch_size = len(batch_users)

        # --- Stratum 3: Global Trends (Base Score) ---
        # Shape: (batch_size, n_items)
        scores = np.tile(global_trends, (current_batch_size, 1))

        # --- Stratum 2: Collaborative Filtering ---
        # R_cf = X[batch] @ S
        X_batch = X[batch_users]
        R_cf_sparse = X_batch.dot(S)
        R_cf_dense = R_cf_sparse.toarray()

        # Normalize and Scale to [100, 1000]
        max_vals = R_cf_dense.max(axis=1, keepdims=True)
        max_vals[max_vals == 0] = 1.0
        R_cf_norm = R_cf_dense / max_vals

        mask_cf = R_cf_dense > 0
        scores[mask_cf] = offset_cf_min + (
            R_cf_norm[mask_cf] * (offset_cf_max - offset_cf_min)
        )

        # --- Stratum 1: Habitual Repurchase ---
        # Score > 2000
        H_batch = H[batch_users]
        H_dense = H_batch.toarray()
        mask_habit = H_dense > 0
        scores[mask_habit] = H_dense[mask_habit]

        # --- Retrieval ---
        # Get indices of top 12 scores
        top_k_indices = np.argpartition(scores, -top_k, axis=1)[:, -top_k:]

        # Sort the top 12 indices by score descending
        rows = np.arange(current_batch_size)[:, None]
        top_k_values = scores[rows, top_k_indices]
        sorted_local_indices = np.argsort(-top_k_values, axis=1)
        final_indices = top_k_indices[rows, sorted_local_indices]

        # --- Evaluation ---
        for i, u_idx in enumerate(batch_users):
            preds = final_indices[i]
            actuals = val_ground_truth[u_idx]
            # Compute Average Precision (AP) for this user
            score = apk(actuals, preds, k=12)
            ap_scores.append(score)
            user_indices.append(u_idx)

    # 6. Compute and Print Metrics
    final_map = np.mean(ap_scores)
    print(f"Final Validation Metric: {final_map:.10f}")

    # 7. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate user history length from training data
    user_hist_len = train_df.groupby("user_idx").size()

    # Create Analysis DataFrame
    analysis_df = pd.DataFrame({"user_idx": user_indices, "ap": ap_scores})

    # Map history length (fill 0 for cold-start users in val)
    analysis_df["hist_len"] = analysis_df["user_idx"].map(user_hist_len).fillna(0)

    # Calculate Error Magnitude (1.0 - AP)
    analysis_df["error"] = 1.0 - analysis_df["ap"]

    # Compute Correlation
    corr = analysis_df["error"].corr(analysis_df["hist_len"])
    print(f"Correlation (Error vs History Length): {corr:.10f}")

    # Cleanup Memory
    del (
        X,
        S,
        H,
        global_trends,
        scores,
        R_cf_dense,
        H_dense,
        train_df,
        val_df,
        analysis_df,
    )
    gc.collect()

    # =========================================================================
    # Phase 2: Submission
    # =========================================================================
    threshold = 0.0265060791

    if final_map > threshold:
        print(f"\n=== Phase 2: Submission (Score {final_map:.5f} > {threshold}) ===")
        print("Retraining on full dataset and generating submission...")

        # Instantiate the Recommender to handle the full pipeline
        # validation=False triggers full data load (T-10w to T) and submission generation
        submitter = StratifiedRecommender()
        submitter.run(validation=False)

    else:
        print(
            f"\nScore {final_map:.5f} did not meet threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
