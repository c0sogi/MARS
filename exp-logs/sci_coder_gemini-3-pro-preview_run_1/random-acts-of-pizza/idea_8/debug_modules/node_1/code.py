import os
import pandas as pd
import numpy as np
import shutil
import warnings

# Import library components
from library.config import Config
from library.utils import set_seed, load_data, get_common_columns, save_submission
from library.feature_engineering import FeaturePipeline
from library.trainers import train_random_forest, train_neural_net, predict_ensemble

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Library Usage Demo ===")

    # 1. Setup Environment and Data Subsets for Speed
    # ------------------------------------------------
    set_seed(42)
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    print("Creating data subsets for rapid execution...")
    # Load a small fraction of the actual data to ensure valid schema
    train_subset = pd.read_csv(Config.TRAIN_DATA_PATH).head(50)
    val_subset = pd.read_csv(Config.VAL_DATA_PATH).head(20)
    test_subset = pd.read_csv(Config.TEST_DATA_PATH).head(20)

    # Save subsets
    train_path = os.path.join(demo_dir, "train_subset.csv")
    val_path = os.path.join(demo_dir, "val_subset.csv")
    test_path = os.path.join(demo_dir, "test_subset.csv")

    train_subset.to_csv(train_path, index=False)
    val_subset.to_csv(val_path, index=False)
    test_subset.to_csv(test_path, index=False)

    # 2. Configure Config Class for Demo
    # ------------------------------------------------
    # We modify the Config singleton directly to affect library modules
    print("Overriding configuration for demonstration...")
    Config.TRAIN_DATA_PATH = train_path
    Config.VAL_DATA_PATH = val_path
    Config.TEST_DATA_PATH = test_path
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Reduce computational load
    Config.RF_N_ESTIMATORS = 10
    Config.MLP_EPOCHS = 2
    Config.MLP_BATCH_SIZE = 8
    Config.MLP_HIDDEN_DIM = 16
    Config.TFIDF_MAX_FEATURES = 100

    # 3. Test Utilities
    # ------------------------------------------------
    print("\n[Testing Library: utils.py]")
    loaded_df = load_data(train_path)
    assert len(loaded_df) == 50, "load_data failed to load correct number of rows"
    print(" - load_data: OK")

    common_cols = get_common_columns(train_subset, test_subset)
    assert (
        "request_text_edit_aware" in common_cols
    ), "get_common_columns failed to find expected column"
    print(" - get_common_columns: OK")

    # 4. Feature Engineering
    # ------------------------------------------------
    print("\n[Testing Library: feature_engineering.py]")
    pipeline = FeaturePipeline()

    # Force processing from scratch (load_cached_data=False)
    # This uses SBERT, which might take a moment to initialize
    print(" - Running feature processing (this includes SBERT encoding)...")
    data = pipeline.process_data(load_cached_data=False)

    # Verify Data Structure
    assert (
        "rf" in data and "mlp" in data
    ), "Processed data dictionary missing required keys"
    assert (
        data["rf"]["X_train"].shape[0] == 50
    ), "RF training data has incorrect row count"
    assert (
        data["mlp"]["X_train_text"].shape[0] == 50
    ), "MLP text data has incorrect row count"
    assert data["y_train"].shape[0] == 50, "Target data has incorrect row count"
    print(" - Feature processing: OK")

    # 5. Model Training: Random Forest
    # ------------------------------------------------
    print("\n[Testing Library: trainers.py - Random Forest]")
    rf_val_preds, rf_test_preds, rf_model = train_random_forest(
        data["rf"]["X_train"],
        data["y_train"],
        data["rf"]["X_val"],
        data["y_val"],
        data["rf"]["X_test"],
    )

    assert len(rf_val_preds) == 20, "RF val predictions shape mismatch"
    assert len(rf_test_preds) == 20, "RF test predictions shape mismatch"
    print(" - Random Forest training and inference: OK")

    # 6. Model Training: Gated Fusion MLP
    # ------------------------------------------------
    print("\n[Testing Library: trainers.py - Gated Fusion MLP]")
    # This implicitly tests library.architectures.GatedFusionMLP
    mlp_val_preds, mlp_test_preds, mlp_model = train_neural_net(
        data["mlp"], data["y_train"], data["y_val"]
    )

    assert len(mlp_val_preds) == 20, "MLP val predictions shape mismatch"
    assert len(mlp_test_preds) == 20, "MLP test predictions shape mismatch"
    print(" - MLP training and inference: OK")

    # 7. Ensembling and Submission
    # ------------------------------------------------
    print("\n[Testing Ensemble and Submission]")
    final_preds = predict_ensemble(rf_test_preds, mlp_test_preds, weights=(0.6, 0.4))
    assert len(final_preds) == 20, "Ensemble predictions shape mismatch"

    save_submission(data["test_ids"], final_preds)
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    # Check submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert sub_df.shape == (20, 2), "Submission file has incorrect shape"
    assert "request_id" in sub_df.columns, "Submission missing ID column"
    assert (
        "requester_received_pizza" in sub_df.columns
    ), "Submission missing target column"
    print(" - Submission generation: OK")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
