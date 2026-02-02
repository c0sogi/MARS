import os
import shutil
import numpy as np
import pandas as pd
import torch
import sys
import warnings

# Import library modules
from library import config
from library import data_loader
from library import feature_pipelines
from library import trainers
from library import neural_arch


def run_demo():
    print("=== Starting Library Usage Demo ===")

    # ---------------------------------------------------------
    # 1. Runtime Configuration Overrides for Speed & Isolation
    # ---------------------------------------------------------
    print("\n[1] Overriding Configuration for Demo Speed...")

    # Use a temporary cache directory for this demo run to ensure we test generation logic
    demo_cache_dir = "./working/demo_run_cache/"
    if os.path.exists(demo_cache_dir):
        shutil.rmtree(demo_cache_dir)
    config.CACHE_DIR = demo_cache_dir

    # Reduce Random Forest complexity
    config.RF_PARAMS = {
        "n_estimators": 5,
        "max_depth": 5,
        "random_state": 42,
        "n_jobs": 1,  # Avoid overhead for small demo
    }

    # Reduce TF-IDF features
    config.TFIDF_PARAMS["max_features"] = 50

    # Reduce MLP Training complexity
    config.MLP_PARAMS["epochs"] = 1
    config.MLP_PARAMS["hidden_dim"] = 16
    config.MLP_PARAMS["batch_size"] = 32
    config.MLP_PARAMS["patience"] = 1

    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    print("Configuration updated.")

    # ---------------------------------------------------------
    # 2. Data Loader Demonstration
    # ---------------------------------------------------------
    print("\n[2] Testing Data Loader...")

    # Force reload from source (ignore cache) to test processing logic
    train_df, val_df, test_df = data_loader.load_and_clean_data(load_cached_data=False)

    # Validation
    print("Validating Data Loader outputs...")
    assert not train_df.empty, "Train DataFrame is empty."
    assert not val_df.empty, "Val DataFrame is empty."
    assert not test_df.empty, "Test DataFrame is empty."

    # Check for Feature Engineering
    expected_ratio_col = "requester_upvote_ratio_at_request"
    assert (
        expected_ratio_col in train_df.columns
    ), f"Engineered feature {expected_ratio_col} missing."

    # Check for Leakage Removal
    # Columns ending in '_at_retrieval' should be gone
    leakage_cols = [c for c in train_df.columns if "_at_retrieval" in c]
    assert len(leakage_cols) == 0, f"Leakage columns detected: {leakage_cols}"

    print(f"Data Loaded Successfully. Train shape: {train_df.shape}")

    # ---------------------------------------------------------
    # 3. Stream A (Lexical-Tabular) Pipeline & Training
    # ---------------------------------------------------------
    print("\n[3] Testing Stream A (Random Forest) Pipeline...")

    pipeline_a = feature_pipelines.StreamA_Pipeline()

    # Run pipeline
    (X_train_a, y_train_a), (X_val_a, y_val_a), (X_test_a, ids_test_a) = pipeline_a.run(
        load_cached_data=False
    )

    # Validate shapes
    assert X_train_a.shape[0] == len(train_df), "Stream A X_train row count mismatch."
    assert (
        X_train_a.shape[1] > 50
    ), "Stream A feature count too low (should be TFIDF + Meta)."

    print("Training Random Forest...")
    rf_model = trainers.train_random_forest((X_train_a, y_train_a), (X_val_a, y_val_a))

    # Validate Prediction
    sample_pred = rf_model.predict_proba(X_val_a[0:1])
    assert sample_pred.shape == (1, 2), "Random Forest prediction shape incorrect."
    print("Stream A execution successful.")

    # ---------------------------------------------------------
    # 4. Stream B (Semantic-Tabular) Pipeline & Training
    # ---------------------------------------------------------
    print("\n[4] Testing Stream B (Dual-Branch MLP) Pipeline...")

    pipeline_b = feature_pipelines.StreamB_Pipeline()

    # Run pipeline
    # Note: This uses SentenceTransformer, which might download the model if not cached.
    # The environment should have internet or cache.
    train_data_b, val_data_b, test_data_b = pipeline_b.run(load_cached_data=False)

    X_sem_train, X_meta_train, y_train_b = train_data_b

    # Validate shapes
    # Embedding dim is usually 384 for all-MiniLM-L6-v2
    assert (
        X_sem_train.shape[1] == 384
    ), f"Unexpected embedding dimension: {X_sem_train.shape[1]}"
    assert X_meta_train.shape[0] == len(train_df), "Stream B X_meta row count mismatch."

    print("Training Dual-Branch MLP...")
    # Force CPU for demo stability if GPU is busy/OOM, though code handles cuda
    # We'll let the library decide, but we reduced batch/epochs in config.
    mlp_trainer = trainers.train_dual_branch_mlp(train_data_b, val_data_b)

    # Validate Prediction
    print("Generating predictions with MLP...")
    X_sem_test, X_meta_test, _ = test_data_b
    mlp_preds = mlp_trainer.predict((X_sem_test, X_meta_test, None))

    assert len(mlp_preds) == len(test_df), "MLP prediction count mismatch."
    assert 0.0 <= mlp_preds.min() <= 1.0, "MLP predictions out of probability range."
    assert 0.0 <= mlp_preds.max() <= 1.0, "MLP predictions out of probability range."

    print("Stream B execution successful.")

    # ---------------------------------------------------------
    # 5. Cleanup
    # ---------------------------------------------------------
    print("\n[5] Cleaning up...")
    if os.path.exists(demo_cache_dir):
        shutil.rmtree(demo_cache_dir)
        print(f"Removed temporary cache: {demo_cache_dir}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Suppress specific warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    try:
        run_demo()
    except AssertionError as e:
        print(f"\n!!! Validation Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n!!! An error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
