import os
import numpy as np
import pandas as pd
import torch
import sys

# Import library components
from library.config import Config
from library.utils import set_seed, save_submission
from library.feature_engineering import prepare_data
from library.rf_pipeline import train_rf_model, predict_rf
from library.mlp_pipeline import train_mlp_model, predict_mlp


def run_demo():
    print("=== Starting Hybrid Ensemble Demo ===")

    # 1. Configuration & Setup
    # Override Config for speed (Debug mode uses subsampled data)
    Config.DEBUG = True
    # Ensure reproducibility
    set_seed(Config.RANDOM_SEED)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Device: {Config.DEVICE}")

    # 2. Feature Engineering Pipeline
    print("\n[Step 1] Running Feature Engineering...")
    # load_cached_data=False forces the pipeline to run from scratch
    train_data, val_data, test_data = prepare_data(load_cached_data=False)

    # Validate Data Structures
    print("Validating processed data structures...")
    required_keys = ["rf_features", "mlp_features", "ids"]
    for key in required_keys:
        assert key in train_data, f"Missing {key} in train_data"
        assert key in test_data, f"Missing {key} in test_data"

    assert "y" in train_data, "Missing targets in train_data"

    # Validate Shapes
    n_train = len(train_data["ids"])
    n_test = len(test_data["ids"])

    assert train_data["rf_features"].shape[0] == n_train
    assert len(train_data["y"]) == n_train
    assert train_data["mlp_features"]["request_emb"].shape[0] == n_train

    print(f"Data successfully processed.")
    print(f"Training Samples: {n_train}")
    print(f"Test Samples: {n_test}")
    print(f"RF Feature Dimensions: {train_data['rf_features'].shape[1]}")

    # 3. Random Forest Stream
    print("\n[Step 2] Training Random Forest Model...")
    # Use reduced hyperparameters for demonstration speed
    rf_model = train_rf_model(
        X_train=train_data["rf_features"],
        y_train=train_data["y"],
        X_val=val_data["rf_features"],
        y_val=val_data["y"],
        n_estimators=10,  # Reduced from default 500
        max_depth=5,  # Constrained depth
        random_state=Config.RANDOM_SEED,
    )

    print("Generating RF predictions...")
    rf_preds_test = predict_rf(rf_model, test_data["rf_features"])

    # Verify predictions are valid probabilities
    assert rf_preds_test.shape == (n_test,)
    assert np.all((rf_preds_test >= 0) & (rf_preds_test <= 1))
    print(
        f"RF Predictions range: [{rf_preds_test.min():.4f}, {rf_preds_test.max():.4f}]"
    )

    # 4. MLP Stream
    print("\n[Step 3] Training MLP (Gated Attention Network)...")
    mlp_model = train_mlp_model(
        train_features=train_data["mlp_features"],
        y_train=train_data["y"],
        val_features=val_data["mlp_features"],
        y_val=val_data["y"],
        batch_size=16,  # Small batch size for debug data
        epochs=2,  # Minimal epochs for demo
        lr=1e-3,
        device=Config.DEVICE,
    )

    print("Generating MLP predictions...")
    mlp_preds_test = predict_mlp(
        mlp_model, test_data["mlp_features"], batch_size=16, device=Config.DEVICE
    )

    # Verify predictions are valid probabilities
    assert mlp_preds_test.shape == (n_test,)
    assert np.all((mlp_preds_test >= 0) & (mlp_preds_test <= 1))
    print(
        f"MLP Predictions range: [{mlp_preds_test.min():.4f}, {mlp_preds_test.max():.4f}]"
    )

    # 5. Ensemble & Submission
    print("\n[Step 4] Creating Ensemble and Submission...")
    # Simple averaging ensemble
    final_preds = (rf_preds_test + mlp_preds_test) / 2.0

    # Define output path
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    save_submission(test_data["ids"], final_preds, filename=submission_path)

    # Final Validation
    if os.path.exists(submission_path):
        df_sub = pd.read_csv(submission_path)
        print(f"Submission file created at: {submission_path}")
        print(f"Submission shape: {df_sub.shape}")
        print("Head of submission:")
        print(df_sub.head())

        # Check consistency
        assert df_sub.shape[0] == n_test
        assert "request_id" in df_sub.columns
        assert "requester_received_pizza" in df_sub.columns
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
