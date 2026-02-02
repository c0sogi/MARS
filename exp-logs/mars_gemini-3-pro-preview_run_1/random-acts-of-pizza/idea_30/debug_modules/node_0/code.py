import os
import shutil
import pandas as pd
import numpy as np
import torch
from library.config import Config
from library.utils import set_seed, ensure_dir
from library.features import FeatureEngineer
from library.model_rf import RFPredictor
from library.model_mlp import MLPTrainer


def run_demo():
    print("Starting Hybrid Ensemble Solution Demo...")

    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Define a temporary directory for this demo run
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    ensure_dir(demo_dir)

    # Override Config for speed and to use the demo directory
    print("Overriding Config settings for fast execution...")
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce Model Complexity for Demo
    Config.RF_N_ESTIMATORS = 10  # Reduced from 500
    Config.RF_N_JOBS = 1  # Avoid overhead for small data
    Config.MLP_EPOCHS = 2  # Reduced from 50
    Config.MLP_BATCH_SIZE = 8  # Small batch for small data
    Config.TFIDF_VOCAB_SIZE = 50  # Reduced vocab
    Config.TOP_K_SUBREDDITS = 5  # Reduced top-k

    # Cache file paths update
    Config.CACHE_RF_TRAIN = os.path.join(demo_dir, "rf_data_train.parquet")
    Config.CACHE_RF_VAL = os.path.join(demo_dir, "rf_data_val.parquet")
    Config.CACHE_RF_TEST = os.path.join(demo_dir, "rf_data_test.parquet")
    Config.CACHE_MLP_TRAIN = os.path.join(demo_dir, "nn_data_train.npz")
    Config.CACHE_MLP_VAL = os.path.join(demo_dir, "nn_data_val.npz")
    Config.CACHE_MLP_TEST = os.path.join(demo_dir, "nn_data_test.npz")

    # Set seed for reproducibility
    set_seed(Config.RANDOM_STATE)

    # =========================================================================
    # 2. Create Data Subsets
    # =========================================================================
    print("\nCreating data subsets for demonstration...")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Create subsets (e.g., 50 samples for train, 20 for val/test)
    subset_train = orig_train.head(50).copy()
    subset_val = orig_val.head(20).copy()
    subset_test = orig_test.head(20).copy()

    # Save subsets to demo directory
    demo_meta_dir = os.path.join(demo_dir, "metadata")
    ensure_dir(os.path.join(demo_meta_dir, "placeholder"))

    demo_train_path = os.path.join(demo_meta_dir, "train.csv")
    demo_val_path = os.path.join(demo_meta_dir, "val.csv")
    demo_test_path = os.path.join(demo_meta_dir, "test.csv")

    subset_train.to_csv(demo_train_path, index=False)
    subset_val.to_csv(demo_val_path, index=False)
    subset_test.to_csv(demo_test_path, index=False)

    # Point Config to these new files
    Config.TRAIN_PATH = demo_train_path
    Config.VAL_PATH = demo_val_path
    Config.TEST_PATH = demo_test_path

    print(f"Subset Train Shape: {subset_train.shape}")
    print(f"Subset Val Shape: {subset_val.shape}")

    # =========================================================================
    # 3. Feature Engineering
    # =========================================================================
    print("\nInitializing Feature Engineer...")
    fe = FeatureEngineer()

    # --- Stream A: Random Forest Features ---
    print("Generating Stream A (RF) features...")
    # Force re-generation by setting load_cached_data=False
    X_train_rf, X_val_rf, X_test_rf = fe.process_stream_a(load_cached_data=False)

    # Assertions for Stream A
    assert X_train_rf.shape[0] == 50, "RF Train rows mismatch"
    assert X_val_rf.shape[0] == 20, "RF Val rows mismatch"
    assert (
        "requester_received_pizza" in X_train_rf.columns
    ), "Target missing in RF Train"
    # Check if TF-IDF columns exist (simple check for columns starting with word chars)
    assert X_train_rf.shape[1] > 10, "RF features seem too few"
    print("Stream A features verified.")

    # --- Stream B: MLP Features ---
    print("Generating Stream B (MLP) features...")
    train_mlp, val_mlp, test_mlp = fe.process_stream_b(load_cached_data=False)

    # Assertions for Stream B
    assert "meta" in train_mlp, "MLP Train missing 'meta' key"
    assert "title_emb" in train_mlp, "MLP Train missing 'title_emb' key"
    assert train_mlp["meta"].shape[0] == 50, "MLP Train meta rows mismatch"
    assert (
        train_mlp["title_emb"].shape[1] == Config.MLP_EMBEDDING_DIM
    ), "Embedding dim mismatch"
    print("Stream B features verified.")

    # =========================================================================
    # 4. Model A: Random Forest
    # =========================================================================
    print("\n--- Running Random Forest Workflow ---")
    rf_model = RFPredictor()

    # Train
    rf_model.train(X_train_rf)

    # Evaluate
    rf_auc = rf_model.evaluate(X_val_rf)
    assert 0.0 <= rf_auc <= 1.0, "RF AUC out of bounds"

    # Predict on Test
    rf_preds_test = rf_model.predict_proba(X_test_rf)
    assert len(rf_preds_test) == 20, "RF Test prediction count mismatch"

    # Save/Load Check
    rf_path = os.path.join(demo_dir, "rf_model.pkl")
    rf_model.save(rf_path)
    loaded_rf = RFPredictor.load(rf_path)
    assert loaded_rf is not None, "Failed to load RF model"
    print("RF Workflow successful.")

    # =========================================================================
    # 5. Model B: MLP
    # =========================================================================
    print("\n--- Running MLP Workflow ---")
    mlp_trainer = MLPTrainer()

    # Train
    # Note: This will run for Config.MLP_EPOCHS (set to 2 above)
    best_auc = mlp_trainer.train(train_mlp, val_mlp)
    print(f"MLP Training Best AUC: {best_auc}")

    # Predict on Test
    mlp_preds_test = mlp_trainer.predict_proba(test_mlp)
    assert len(mlp_preds_test) == 20, "MLP Test prediction count mismatch"
    assert (
        mlp_preds_test.dtype == np.float32 or mlp_preds_test.dtype == np.float64
    ), "MLP preds not float"
    print("MLP Workflow successful.")

    # =========================================================================
    # 6. Ensemble & Submission
    # =========================================================================
    print("\n--- Creating Ensemble Submission ---")

    # Weighted Average
    final_preds = (Config.WEIGHT_RF * rf_preds_test) + (
        Config.WEIGHT_MLP * mlp_preds_test
    )

    # Create Submission DataFrame
    submission = pd.DataFrame(
        {
            "request_id": subset_test["request_id"],
            "requester_received_pizza": final_preds,
        }
    )

    print("Submission Head:")
    print(submission.head())

    ensure_dir(Config.SUBMISSION_FILE)
    submission.to_csv(Config.SUBMISSION_FILE, index=False)
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not created"

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
