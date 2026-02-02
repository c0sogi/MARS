import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
import shutil
from library.config import Config
from library.data_utils import (
    get_global_mapper,
    load_and_filter_data,
    build_user_history_vectors,
)
from library.visual_features import (
    extract_embeddings,
    compute_visual_similarity_matrix,
)
from library.collaborative_filtering import (
    compute_behavioral_similarity_matrix,
    calculate_global_trend,
)
from library.hybrid_recommender import HybridRecommender


def run_demo():
    print("=== Starting Library Usage Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed and Isolation
    # ---------------------------------------------------------
    print("\n[1] Configuring environment...")

    # Use a separate working directory for this demo
    DEMO_DIR = "./working/demo_script_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config attributes
    Config.WORKING_DIR = DEMO_DIR
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 10000  # Small sample for speed
    Config.TRAIN_WEEKS = 1  # Only 1 week of history
    Config.BATCH_SIZE = 1024  # High batch size for speed

    # Update cache paths to point to the new demo directory
    Config.CACHE_IMAGE_EMBEDDINGS = os.path.join(DEMO_DIR, "image_embeddings.parquet")
    Config.CACHE_SIM_VISUAL = os.path.join(DEMO_DIR, "similarity_visual.npz")
    Config.CACHE_SIM_BEHAVIOR = os.path.join(DEMO_DIR, "similarity_behavior.npz")
    Config.CACHE_USER_HISTORY = os.path.join(DEMO_DIR, "user_history.parquet")
    Config.CACHE_GLOBAL_TRENDS = os.path.join(DEMO_DIR, "global_trends.parquet")

    # Create a small sample submission file to speed up inference
    print("Creating small sample submission file...")
    full_sample_sub = pd.read_csv(Config.PATH_SAMPLE_SUBMISSION)
    small_sample_sub_path = os.path.join(DEMO_DIR, "small_sample_submission.csv")

    # Take top 100 users
    subset_users = full_sample_sub.head(100).copy()
    subset_users.to_csv(small_sample_sub_path, index=False)
    Config.PATH_SAMPLE_SUBMISSION = small_sample_sub_path
    Config.PATH_SUBMISSION = os.path.join(DEMO_DIR, "submission_demo.csv")

    # ---------------------------------------------------------
    # 2. Data Utils Demonstration
    # ---------------------------------------------------------
    print("\n[2] Testing Data Utils...")

    # Test IndexMapper
    mapper = get_global_mapper()
    n_users = mapper.get_num_users()
    n_items = mapper.get_num_items()
    print(f"Mapper stats: {n_users} users, {n_items} items")

    assert n_users > 0, "Mapper found no users"
    assert n_items > 0, "Mapper found no items"

    # Test Load and Filter
    df_train = load_and_filter_data(
        Config.PATH_TRAIN, weeks=Config.TRAIN_WEEKS, debug=Config.DEBUG
    )
    print(f"Loaded train data shape: {df_train.shape}")

    assert not df_train.empty, "Training dataframe is empty"
    assert "t_dat" in df_train.columns
    assert "customer_id" in df_train.columns

    # Test User History Construction
    # This will compute and save to Config.CACHE_USER_HISTORY
    U_hist = build_user_history_vectors(df_train, mapper, load_cached_data=False)
    print(f"User History Matrix: {U_hist.shape}, NNZ: {U_hist.nnz}")

    assert U_hist.shape == (n_users, n_items)
    assert sp.issparse(U_hist)

    # ---------------------------------------------------------
    # 3. Visual Features Demonstration (Mocked)
    # ---------------------------------------------------------
    print("\n[3] Testing Visual Features...")

    # Optimization: Instead of running ResNet50 on 100k items (which takes time),
    # we generate random low-dim embeddings and save them to the cache location.
    # This verifies the pipeline logic without the compute cost.
    print("Generating mock image embeddings for speed...")
    mock_dim = 64
    mock_embeddings = np.random.randn(n_items, mock_dim).astype(np.float32)

    df_emb = pd.DataFrame(
        {"item_idx": np.arange(n_items), "embedding": list(mock_embeddings)}
    )
    df_emb.to_parquet(Config.CACHE_IMAGE_EMBEDDINGS, index=False)

    # Now call extract_embeddings with load_cached_data=True
    # It should pick up our mock file
    loaded_embeddings = extract_embeddings(mapper, load_cached_data=True)
    assert loaded_embeddings.shape == (n_items, mock_dim)

    # Test Visual Similarity Computation
    # This will use the loaded (mock) embeddings to compute similarity
    S_vis = compute_visual_similarity_matrix(mapper, load_cached_data=False)
    print(f"Visual Similarity Matrix: {S_vis.shape}, NNZ: {S_vis.nnz}")

    assert S_vis.shape == (n_items, n_items)
    assert sp.issparse(S_vis)

    # ---------------------------------------------------------
    # 4. Collaborative Filtering Demonstration
    # ---------------------------------------------------------
    print("\n[4] Testing Collaborative Filtering...")

    # Test Behavioral Similarity
    S_beh = compute_behavioral_similarity_matrix(
        df_train, mapper, load_cached_data=False
    )
    print(f"Behavioral Similarity Matrix: {S_beh.shape}, NNZ: {S_beh.nnz}")

    assert S_beh.shape == (n_items, n_items)

    # Test Global Trend
    V_trend = calculate_global_trend(df_train, mapper, load_cached_data=False)
    print(f"Global Trend Vector: {V_trend.shape}")

    assert V_trend.shape == (n_items,)
    assert V_trend.max() <= 1.0 + 1e-6

    # ---------------------------------------------------------
    # 5. Hybrid Recommender Integration
    # ---------------------------------------------------------
    print("\n[5] Testing Hybrid Recommender...")

    # Instantiate Recommender
    # It will reload the caches we just created/verified
    recommender = HybridRecommender(
        load_cached_data=True, train_weeks=Config.TRAIN_WEEKS
    )

    # Test Single Batch Prediction
    # Get indices for the first 10 users in our subset
    test_user_ids = subset_users["customer_id"].iloc[:10]
    test_user_indices = mapper.map_users(test_user_ids)

    scores = recommender.predict_scores(test_user_indices)
    print(f"Prediction scores shape: {scores.shape}")

    assert scores.shape == (10, n_items)
    assert not np.isnan(scores).any(), "Scores contain NaNs"

    # Test Full Submission Generation
    # This uses the small_sample_submission.csv we created earlier
    recommender.generate_submission(output_path=Config.PATH_SUBMISSION, batch_size=50)

    # Validate Output
    assert os.path.exists(Config.PATH_SUBMISSION)
    df_sub = pd.read_csv(Config.PATH_SUBMISSION)

    print(f"Submission generated with shape: {df_sub.shape}")
    assert len(df_sub) == 100, "Submission should have 100 rows (matching our subset)"
    assert "customer_id" in df_sub.columns
    assert "prediction" in df_sub.columns

    # Check prediction format
    sample_pred = df_sub.iloc[0]["prediction"]
    assert isinstance(sample_pred, str)
    items = sample_pred.split()
    assert len(items) <= 12, "Should predict max 12 items"
    # Check item format (10 digits)
    assert len(items[0]) == 10, "Item ID should be 10 characters long"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Ensure reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    run_demo()
