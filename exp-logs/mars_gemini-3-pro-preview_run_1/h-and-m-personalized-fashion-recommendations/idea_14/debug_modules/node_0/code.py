import os
import shutil
import numpy as np
import pandas as pd
import scipy.sparse as sp
from library.config import Config
from library.data_utils import load_processed_data, get_temporal_view
from library.id_mapper import IdMapper
from library.sparse_engine import SparseEngine
from library.igdc_model import IGDCRecommender
from library.metrics import calculate_map12


def run_demo():
    print(">>> 1. Configuring Environment")

    # Set seeds for reproducibility
    np.random.seed(42)

    # --- Configuration Override for Fast Demo ---
    # We override the Config class attributes to run in a separate demo directory
    # and use smaller parameters for speed.

    demo_base_dir = "./working/demo_execution"
    if os.path.exists(demo_base_dir):
        shutil.rmtree(demo_base_dir)
    os.makedirs(demo_base_dir, exist_ok=True)

    # 1. Path Overrides
    Config.WORKING_DIR = demo_base_dir
    Config.SUBMISSION_DIR = os.path.join(demo_base_dir, "submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # We must update these dependent paths because they were evaluated at import time
    Config.CACHE_SIMILARITY_MATRIX = os.path.join(
        Config.WORKING_DIR, "similarity_matrix.npz"
    )
    Config.CACHE_USER_HISTORY = os.path.join(Config.WORKING_DIR, "user_history.parquet")
    Config.CACHE_ITEM_MAP = os.path.join(Config.WORKING_DIR, "item_map.parquet")
    Config.CACHE_USER_MAP = os.path.join(Config.WORKING_DIR, "user_map.parquet")
    Config.CACHE_INVENTORY_MASK = os.path.join(Config.WORKING_DIR, "inventory_mask.npy")
    Config.CACHE_GLOBAL_TREND = os.path.join(Config.WORKING_DIR, "global_trends.npy")
    Config.CACHE_HABIT_MATRIX = os.path.join(Config.WORKING_DIR, "habit_matrix.npz")

    # 2. Hyperparameter Overrides (Optimization for Speed)
    # Reducing window sizes reduces the number of transactions processed
    Config.WINDOW_STRUCTURE_DAYS = 7  # Use only 1 week for structure learning
    Config.WINDOW_INTENT_DAYS = 3  # Use 3 days for user intent
    Config.WINDOW_HABIT_DAYS = 14  # Use 2 weeks for habit
    Config.WINDOW_INVENTORY_DAYS = 3  # Use 3 days for trend/inventory

    Config.TOP_K_NEIGHBORS = 10  # Prune similarity matrix aggressively

    # Ensure directories exist
    Config.setup()

    print(f"  Working Directory: {Config.WORKING_DIR}")
    print(f"  Submission File: {Config.SUBMISSION_FILE}")

    # -------------------------------------------------------------------------
    # Data Loading & Processing Demo
    # -------------------------------------------------------------------------
    print("\n>>> 2. Demonstrating Data Loading (data_utils.py)")

    # Force load from scratch (load_cached_data=False) to demonstrate processing logic
    # This reads metadata/train.csv and metadata/val.csv, creates maps, and saves to demo_base_dir
    print("  Processing raw data...")
    transactions, user_map, item_map = load_processed_data(load_cached_data=False)

    # Validations
    print("  Validating loaded data structures...")
    assert isinstance(transactions, pd.DataFrame), "Transactions should be a DataFrame"
    assert not transactions.empty, "Transactions DataFrame is empty"
    assert "user_idx" in transactions.columns, "user_idx missing from transactions"
    assert "item_idx" in transactions.columns, "item_idx missing from transactions"
    assert len(user_map) > 0, "User map is empty"
    assert len(item_map) > 0, "Item map is empty"

    # Demonstrate Temporal View
    print("  Demonstrating Temporal View extraction...")
    ref_date = "2020-09-22"
    lookback_days = 3
    view = get_temporal_view(transactions, lookback_days, ref_date)

    # Check date range logic
    min_date = pd.to_datetime(ref_date) - pd.Timedelta(days=lookback_days)
    max_date = pd.to_datetime(ref_date)

    assert not view.empty, "Temporal view is empty"
    assert view["t_dat"].min() > min_date, "Temporal view contains old data"
    assert view["t_dat"].max() <= max_date, "Temporal view contains future data"
    print(f"  Temporal view extracted: {len(view)} rows within {lookback_days} days.")

    # -------------------------------------------------------------------------
    # ID Mapping Demo
    # -------------------------------------------------------------------------
    print("\n>>> 3. Demonstrating ID Mapping (id_mapper.py)")

    mapper = IdMapper()
    # Fit using the cache we just created in step 2
    mapper.fit(load_cached_data=True)

    # Test transformation consistency
    sample_user_id = user_map.iloc[0]["customer_id"]

    # Transform: ID -> Index
    user_idx = mapper.transform(sample_user_id, "user")
    assert isinstance(user_idx, (int, np.integer)), "Transform should return integer"

    # Inverse: Index -> ID
    recovered_id = mapper.inverse_transform(user_idx, "user")
    assert recovered_id == sample_user_id, "Inverse transform failed to recover ID"

    print(f"  Mapping check passed: {sample_user_id} -> {user_idx} -> {recovered_id}")

    # -------------------------------------------------------------------------
    # Sparse Engine Demo
    # -------------------------------------------------------------------------
    print("\n>>> 4. Demonstrating Sparse Engine (sparse_engine.py)")

    engine = SparseEngine()

    n_users = mapper.get_user_count()
    n_items = mapper.get_item_count()

    # Build a small interaction matrix from the temporal view
    print("  Building interaction matrix...")
    interaction_mat = engine.build_interaction_matrix(view, n_users, n_items)

    assert sp.issparse(interaction_mat), "Matrix should be sparse"
    assert interaction_mat.shape == (
        n_users,
        n_items,
    ), f"Incorrect shape: {interaction_mat.shape}"

    # Compute similarity (using the small view for speed)
    print("  Computing similarity matrix...")
    sim_mat = engine.compute_similarity(interaction_mat, top_k=5)

    assert sp.issparse(sim_mat), "Similarity matrix should be sparse"
    assert sim_mat.shape == (n_items, n_items), "Similarity matrix has wrong shape"
    print(f"  Similarity Matrix stats: Shape={sim_mat.shape}, NNZ={sim_mat.nnz}")

    # -------------------------------------------------------------------------
    # Full Model Pipeline Demo
    # -------------------------------------------------------------------------
    print("\n>>> 5. Demonstrating IGDC Recommender Pipeline (igdc_model.py)")

    model = IGDCRecommender()

    # Fit the model
    # This will use the cached data and the overridden Config parameters
    print("  Fitting model (this may take a moment)...")
    model.fit(load_cached_data=True)

    # Verify internal state
    assert model.S_long is not None, "Similarity matrix S_long not initialized"
    assert model.M_active is not None, "Inventory mask M_active not initialized"
    assert model.R_trend is not None, "Trend vector R_trend not initialized"

    # Predict
    # We use a large batch size to process the loop efficiently
    print("  Generating predictions...")
    model.predict(batch_size=50000)

    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created"
    print(f"  Submission generated at {Config.SUBMISSION_FILE}")

    # -------------------------------------------------------------------------
    # Evaluation Demo
    # -------------------------------------------------------------------------
    print("\n>>> 6. Demonstrating Metrics (metrics.py)")

    # Load the generated submission
    preds_df = pd.read_csv(Config.SUBMISSION_FILE, dtype={"prediction": str})

    # Calculate MAP@12
    # This compares the predictions against the validation set (metadata/val.csv).
    # Note: Since load_processed_data combines Train+Val by default for the 'processed' view,
    # and the model uses 'transactions' (Train+Val) to learn structure, this score
    # represents a training/leakage score, but it validates the code functionality.
    print("  Calculating MAP@12...")
    score = calculate_map12(preds_df, load_cached_data=False)

    print(f"  Final MAP@12 Score: {score:.6f}")
    assert 0.0 <= score <= 1.0, "MAP@12 score out of range"

    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
