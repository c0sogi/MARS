import pandas as pd
import numpy as np
import os
import shutil
import sys

# Import provided libraries
from library import config
from library import data_utils
from library import sparse_engine
from library import stratified_inference
from library import evaluation


def main():
    print("Starting Demo Script...")

    # 1. Setup & Configuration Overrides
    # We use a temporary cache directory for the demo to ensure clean execution
    DEMO_CACHE_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_CACHE_DIR):
        shutil.rmtree(DEMO_CACHE_DIR)
    os.makedirs(DEMO_CACHE_DIR)

    # Override config paths to use the demo cache
    config.CACHE_DIR = DEMO_CACHE_DIR

    # Set seeds for reproducibility
    np.random.seed(config.RANDOM_SEED)

    # 2. Data Loading & Preprocessing
    print("\n[Step 1] Loading and Splitting Data...")

    # Load transactions (forcing reload to ensure logic execution)
    # This loads the full train.csv.
    full_df = data_utils.load_transactions(config.TRAIN_PATH, load_cached_data=False)

    # Optimization: Filter to last 4 weeks to speed up matrix construction for the demo
    # The full pipeline typically uses 20 weeks.
    print("Filtering to last 4 weeks for demo speed...")
    train_window_df = data_utils.filter_date_window(full_df, weeks=4)

    # Split into Train and Validation (Last 1 week for validation)
    train_df, val_df = data_utils.get_time_split(train_window_df, val_weeks=1)

    # Load metadata for mapping generation
    customers_df = pd.read_csv(config.CUSTOMERS_PATH)
    articles_df = data_utils.load_articles()

    # Generate Mappings
    # We pass customers_df to ensure all users (including cold-start) are mapped
    user_to_idx, idx_to_user, item_to_idx, idx_to_item = data_utils.generate_mappings(
        train_df, customers_df, articles_df
    )

    # Validation 1: Check Mappings
    n_users = len(user_to_idx)
    n_items = len(item_to_idx)
    print(f"Mappings created: {n_users} users, {n_items} items.")
    assert n_users >= len(customers_df), "User mapping should include all customers"
    assert n_items >= len(articles_df), "Item mapping should include all articles"

    # 3. Sparse Matrix Construction
    print("\n[Step 2] Building Interaction and Similarity Matrices...")

    # Build Interaction Matrix (User x Item)
    interaction_matrix = sparse_engine.build_decayed_interaction_matrix(
        train_df, user_to_idx, item_to_idx, load_cached_data=False
    )

    # Validation 2: Interaction Matrix
    assert interaction_matrix.shape == (
        n_users,
        n_items,
    ), "Interaction matrix shape mismatch"
    # Check that rows are L2 normalized (magnitude approx 1.0 for active users)
    active_user_indices = np.diff(interaction_matrix.indptr).nonzero()[0]
    if len(active_user_indices) > 0:
        sample_idx = active_user_indices[0]
        norm = np.linalg.norm(interaction_matrix[sample_idx].data)
        assert np.isclose(
            norm, 1.0, atol=1e-5
        ), f"Rows should be L2 normalized. Got {norm}"

    # Compute Similarity Matrix (Item x Item)
    # Reducing top_k to 50 for speed in demo
    similarity_matrix = sparse_engine.compute_similarity_matrix(
        interaction_matrix, top_k=50, load_cached_data=False
    )

    # Validation 3: Similarity Matrix
    assert similarity_matrix.shape == (
        n_items,
        n_items,
    ), "Similarity matrix shape mismatch"

    # 4. Stratified Inference
    print("\n[Step 3] Initializing and Fitting Recommender...")

    recommender = stratified_inference.StratifiedRecommender()

    # Fit the model
    recommender.fit(
        train_df,
        interaction_matrix,
        similarity_matrix,
        user_to_idx,
        idx_to_user,
        item_to_idx,
        idx_to_item,
        load_cached_data=False,
    )

    # Validation 4: Internal Model State
    assert recommender.habit_matrix is not None, "Habit matrix not built"
    assert recommender.global_trend is not None, "Global trend not built"
    assert recommender.habit_matrix.shape == (
        n_users,
        n_items,
    ), "Habit matrix dimensions wrong"

    # 5. Prediction
    print("\n[Step 4] Generating Predictions...")

    # We predict for validation users to calculate MAP@12
    # To save time, we sample 5000 validation users
    val_customers = val_df["customer_id"].unique()
    sample_size = min(5000, len(val_customers))
    val_customers_sample = np.random.choice(val_customers, sample_size, replace=False)

    print(f"Predicting for {len(val_customers_sample)} validation users...")
    predictions_df = recommender.predict(val_customers_sample, batch_size=1000)

    # Validation 5: Prediction Output
    assert "customer_id" in predictions_df.columns
    assert "prediction" in predictions_df.columns
    assert len(predictions_df) == len(val_customers_sample)
    # Check format of prediction string (space separated article IDs)
    sample_pred = predictions_df.iloc[0]["prediction"]
    assert isinstance(sample_pred, str)
    # Should have up to 12 items
    assert len(sample_pred.split()) <= 12

    # 6. Evaluation
    print("\n[Step 5] Evaluating MAP@12...")

    # Filter validation dataframe to match the sampled customers
    val_df_subset = val_df[val_df["customer_id"].isin(val_customers_sample)]

    score = evaluation.calculate_map12(val_df_subset, predictions_df, k=12)

    # Validation 6: Score Sanity
    assert 0.0 <= score <= 1.0, f"MAP score {score} out of range [0, 1]"

    print(f"\nDemo Completed Successfully. MAP@12 on sample: {score:.6f}")


if __name__ == "__main__":
    main()
