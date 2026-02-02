import pandas as pd
import numpy as np
import os
import shutil
import scipy.sparse as sp
import random
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":
    set_seed(42)

    # --- 1. Monkeypatch Configuration for Demo ---
    # We redirect paths to a working directory to use a small subset of data
    from library.config import Config

    DEMO_BASE = "./working/demo_run"
    # Clean up previous run if exists
    if os.path.exists(DEMO_BASE):
        shutil.rmtree(DEMO_BASE)
    os.makedirs(DEMO_BASE, exist_ok=True)

    # Patch Config paths
    Config.WORKING_DIR = DEMO_BASE
    Config.CACHE_DIR = os.path.join(DEMO_BASE, "cache")
    Config.METADATA_DIR = os.path.join(DEMO_BASE, "metadata")
    Config.INPUT_DIR = os.path.join(DEMO_BASE, "input")
    Config.SUBMISSION_DIR = os.path.join(DEMO_BASE, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Patch Config parameters for small data
    Config.MIN_ITEM_PURCHASES = 1  # Reduce threshold for small sample

    # Ensure directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.METADATA_DIR, exist_ok=True)
    os.makedirs(Config.INPUT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Configuration patched. Working directory: {DEMO_BASE}")

    # --- 2. Create Sample Data ---
    print("Creating sample data for demonstration...")

    ORIG_METADATA_DIR = "./metadata"
    ORIG_INPUT_DIR = "./input"

    # Load small subsets of the provided metadata
    # Train: 10,000 rows
    df_train_sample = pd.read_csv(
        os.path.join(ORIG_METADATA_DIR, "train.csv"), nrows=10000
    )
    # Val: 2,000 rows
    df_val_sample = pd.read_csv(os.path.join(ORIG_METADATA_DIR, "val.csv"), nrows=2000)
    # Test: 500 rows
    df_test_sample = pd.read_csv(os.path.join(ORIG_METADATA_DIR, "test.csv"), nrows=500)

    # Save to patched metadata directory
    df_train_sample.to_csv(os.path.join(Config.METADATA_DIR, "train.csv"), index=False)
    df_val_sample.to_csv(os.path.join(Config.METADATA_DIR, "val.csv"), index=False)
    df_test_sample.to_csv(os.path.join(Config.METADATA_DIR, "test.csv"), index=False)

    # Create a subset of customers.csv
    # We read the first chunk of the real customers file
    df_customers_orig = pd.read_csv(
        os.path.join(ORIG_INPUT_DIR, "customers.csv"), nrows=5000
    )
    df_customers_orig.to_csv(
        os.path.join(Config.INPUT_DIR, "customers.csv"), index=False
    )

    print("Sample data created successfully.")

    # --- 3. Import Library Modules ---
    from library.data_processor import (
        load_and_filter_data,
        create_mappings,
        process_customer_cohorts,
    )
    from library.matrix_factory import MatrixFactory
    from library.trend_analyzer import TrendAnalyzer
    from library.inference_engine import StratifiedRecommender
    from library.utils import calculate_map12, format_submission

    # --- 4. Demonstrate Data Processor ---
    print("\n=== Demonstrating Data Processor ===")

    # A. Load and Filter
    print("Running load_and_filter_data...")
    train_df, val_df, test_df = load_and_filter_data(load_cached_data=False)

    assert not train_df.empty, "Train DF should not be empty"
    assert not val_df.empty, "Val DF should not be empty"
    assert not test_df.empty, "Test DF should not be empty"
    print(
        f"Data Loaded. Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}"
    )

    # B. Create Mappings
    print("Running create_mappings...")
    user_to_idx, idx_to_user, item_to_idx, idx_to_item = create_mappings(
        train_df, val_df, test_df, load_cached_data=False
    )

    n_users = len(user_to_idx)
    n_items = len(item_to_idx)
    print(f"Mappings created. Users: {n_users}, Items: {n_items}")
    assert n_users > 0
    assert n_items > 0

    # C. Process Cohorts
    print("Running process_customer_cohorts...")
    user_cohort_map = process_customer_cohorts(user_to_idx, load_cached_data=False)

    assert (
        len(user_cohort_map) == n_users
    ), "Cohort map length must match number of users"
    print(f"Cohort map generated. Shape: {user_cohort_map.shape}")

    # --- 5. Demonstrate Matrix Factory ---
    print("\n=== Demonstrating Matrix Factory ===")

    # A. User History Matrix
    print("Building User History Matrix...")
    U = MatrixFactory.build_user_history_matrix(
        train_df, user_to_idx, item_to_idx, load_cached_data=False
    )
    assert U.shape == (n_users, n_items)
    print(f"User History Matrix (U) built. Shape: {U.shape}, NNZ: {U.nnz}")

    # B. Symmetric Similarity
    print("Building Symmetric Similarity Matrix...")
    S_sym = MatrixFactory.build_symmetric_similarity(
        train_df, user_to_idx, item_to_idx, load_cached_data=False
    )
    assert S_sym.shape == (n_items, n_items)
    print(f"Symmetric Similarity (S_sym) built. Shape: {S_sym.shape}")

    # C. Transition Matrix
    print("Building Transition Matrix...")
    S_fwd = MatrixFactory.build_transition_matrix(
        train_df, user_to_idx, item_to_idx, load_cached_data=False
    )
    assert S_fwd.shape == (n_items, n_items)
    print(f"Transition Matrix (S_fwd) built. Shape: {S_fwd.shape}")

    # D. Hybrid Matrix
    print("Creating Hybrid Matrix...")
    S_hybrid = MatrixFactory.get_hybrid_matrix(S_sym, S_fwd)
    assert S_hybrid.shape == (n_items, n_items)
    print(f"Hybrid Matrix created.")

    # --- 6. Demonstrate Trend Analyzer ---
    print("\n=== Demonstrating Trend Analyzer ===")

    # A. Global Trends
    print("Computing Global Trends...")
    global_trends = TrendAnalyzer.compute_global_trends(
        train_df, item_to_idx, load_cached_data=False
    )
    assert global_trends.shape == (n_items,)
    print(f"Global trends computed. Max value: {global_trends.max():.4f}")

    # B. Cohort Trends
    print("Computing Cohort Trends...")
    cohort_trends = TrendAnalyzer.compute_cohort_trends(
        train_df, user_cohort_map, user_to_idx, item_to_idx, load_cached_data=False
    )
    assert isinstance(cohort_trends, dict)
    print(f"Cohort trends computed for {len(cohort_trends)} cohorts.")

    # --- 7. Demonstrate Inference Engine ---
    print("\n=== Demonstrating Inference Engine ===")

    # Instantiate Recommender
    recommender = StratifiedRecommender(
        user_history_matrix=U,
        hybrid_matrix=S_hybrid,
        cohort_trends=cohort_trends,
        global_trends=global_trends,
        user_cohort_map=user_cohort_map,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        idx_to_item=idx_to_item,
    )

    # Predict for Validation Users
    print("Predicting for validation users (subset)...")
    val_customers = val_df["customer_id"].unique()[:20]
    preds_matrix = recommender.predict(val_customers)

    assert preds_matrix.shape == (len(val_customers), Config.TOP_K)
    print(f"Predictions generated. Shape: {preds_matrix.shape}")

    # Convert to dict for MAP calculation
    val_preds_dict = {}
    for i, cid in enumerate(val_customers):
        pred_items = [idx_to_item[idx] for idx in preds_matrix[i]]
        val_preds_dict[cid] = pred_items

    # --- 8. Demonstrate Utils ---
    print("\n=== Demonstrating Utils ===")

    # Calculate MAP@12
    print("Calculating MAP@12...")
    # Filter val_df to only include the customers we predicted for
    val_df_subset = val_df[val_df["customer_id"].isin(val_customers)]
    score = calculate_map12(val_df_subset, val_preds_dict)
    print(f"MAP@12 Score: {score}")

    # Format Submission
    print("Formatting submission...")
    test_customers = test_df["customer_id"].unique()[:20]
    test_preds = recommender.predict(test_customers)
    format_submission(test_preds, test_customers, idx_to_item)

    # Verify submission file exists
    assert os.path.exists(Config.SUBMISSION_PATH)
    print(f"Submission file verified at {Config.SUBMISSION_PATH}")

    print("\nAll demonstrations completed successfully.")
