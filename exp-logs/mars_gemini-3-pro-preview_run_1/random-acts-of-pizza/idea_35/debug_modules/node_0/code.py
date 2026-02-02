import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library
from library.config import Config
from library.utils import set_seed, ensure_dir
from library.data_loader import load_dataset
from library.feature_engineering import FeaturePipeline
from library.model_architecture import DualQueryMLP, PizzaDataset
from library.model_training import train_rf, predict_rf, train_mlp, predict_mlp

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Library Demo ===\n")

    # 1. Setup and Configuration
    # We override some Config values to ensure the demo runs quickly and uses a temporary directory
    print("1. Setting up configuration...")
    Config.CACHE_DIR = "./working/demo_cache"
    Config.SUBMISSION_DIR = "./working/demo_output"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Ensure reproducibility
    set_seed(42)
    ensure_dir(Config.CACHE_DIR)
    ensure_dir(Config.SUBMISSION_DIR)
    print("   Configuration set. Cache dir:", Config.CACHE_DIR)

    # 2. Data Loading Verification
    print("\n2. Verifying Data Loader...")
    df_train = load_dataset("train", load_cached_data=False)
    df_val = load_dataset("val", load_cached_data=False)

    # Basic assertions
    assert isinstance(df_train, pd.DataFrame), "Train data should be a DataFrame"
    assert not df_train.empty, "Train data should not be empty"
    assert (
        "requester_received_pizza" in df_train.columns
    ), "Target column missing in train"
    assert "request_text_edit_aware" in df_train.columns, "Text column missing"
    # Verify list parsing happened (check type of first element in subreddits if not empty)
    if not df_train["requester_subreddits_at_request"].empty:
        first_val = df_train["requester_subreddits_at_request"].iloc[0]
        assert isinstance(
            first_val, list
        ), f"List column parsing failed, got {type(first_val)}"

    print(f"   Train shape: {df_train.shape}")
    print(f"   Val shape: {df_val.shape}")
    print("   Data Loader verification passed.")

    # 3. Model Architecture Verification (Unit Test style)
    print("\n3. Verifying DualQueryMLP Architecture...")
    # Create dummy inputs
    B, D, L, M = 4, 32, 10, 5  # Batch=4, EmbDim=32, HistoryLen=10, MetaDim=5
    dummy_title = torch.randn(B, D)
    dummy_body = torch.randn(B, D)
    dummy_history = torch.randn(B, L, D)
    dummy_meta = torch.randn(B, M)

    # Instantiate model
    model = DualQueryMLP(emb_dim=D, meta_dim=M, hidden_dim=16, dropout=0.1)

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(dummy_title, dummy_body, dummy_history, dummy_meta)

    assert logits.shape == (B,), f"Expected output shape ({B},), got {logits.shape}"
    print("   DualQueryMLP forward pass successful.")

    # 4. Feature Engineering Pipeline
    print("\n4. Running Feature Engineering Pipeline...")
    # We run the pipeline. Note: This processes the full dataset (2k samples),
    # which is fast enough (approx 1-2 mins) given the pre-installed packages.
    pipeline = FeaturePipeline()

    # Force re-computation to demonstrate the logic, ignoring any existing cache
    if os.path.exists(pipeline.cache_file):
        os.remove(pipeline.cache_file)

    data_dict = pipeline.run(load_cached_data=False)

    # Verify dictionary structure
    assert "rf" in data_dict
    assert "mlp" in data_dict
    assert "train" in data_dict["mlp"]

    # Unpack RF data for verification
    X_rf_train, y_train, X_rf_val, y_val, X_rf_test, test_ids = data_dict["rf"]
    assert X_rf_train.shape[0] == len(y_train)
    assert X_rf_train.shape[1] > 0
    print(f"   Generated RF features: {X_rf_train.shape}")
    print("   Feature Pipeline execution successful.")

    # 5. Random Forest Training & Inference
    print("\n5. Training Random Forest (Fast Mode)...")
    # Train with reduced estimators and subsampling for speed
    rf_model = train_rf(
        X_rf_train,
        y_train,
        X_rf_val,
        y_val,
        n_estimators=10,  # Reduced
        debug_sample_size=200,  # Subsample
    )

    # Predict
    rf_preds = predict_rf(rf_model, X_rf_test)
    assert len(rf_preds) == len(test_ids)
    assert np.all((rf_preds >= 0) & (rf_preds <= 1))
    print("   RF Training and Inference successful.")

    # 6. MLP Training & Inference
    print("\n6. Training MLP (Fast Mode)...")
    # Unpack MLP data
    mlp_train_data = data_dict["mlp"]["train"]
    mlp_val_data = data_dict["mlp"]["val"]
    mlp_test_data = data_dict["mlp"]["test"]

    # Train with minimal epochs and subsampling
    mlp_model = train_mlp(
        mlp_train_data,
        mlp_val_data,
        epochs=2,  # Reduced
        batch_size=16,  # Small batch
        patience=1,
        debug_sample_size=200,  # Subsample
    )

    # Predict
    mlp_preds = predict_mlp(mlp_model, mlp_test_data, batch_size=16)
    assert len(mlp_preds) == len(test_ids)
    assert np.all((mlp_preds >= 0) & (mlp_preds <= 1))
    print("   MLP Training and Inference successful.")

    # 7. Ensemble and Submission Generation
    print("\n7. Generating Submission...")
    # Simple average ensemble
    final_preds = 0.5 * rf_preds + 0.5 * mlp_preds

    submission = pd.DataFrame(
        {"request_id": test_ids, "requester_received_pizza": final_preds}
    )

    submission.to_csv(Config.SUBMISSION_PATH, index=False)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH)
    loaded_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert loaded_sub.shape == (len(test_ids), 2)
    assert list(loaded_sub.columns) == ["request_id", "requester_received_pizza"]

    print(f"   Submission saved to {Config.SUBMISSION_PATH}")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
