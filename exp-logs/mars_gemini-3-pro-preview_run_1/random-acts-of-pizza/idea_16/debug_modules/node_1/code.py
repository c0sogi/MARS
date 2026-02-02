import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Import provided library modules
from library import config
from library import feature_factory
from library import rf_model
from library import mlp_model
from library import utils

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Pizza Request Prediction Demo ===")

    # -------------------------------------------------------------------------
    # 1. SETUP & CONFIGURATION OVERRIDE
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Override configuration for speed (Debug Mode)
    config.DEBUG_SAMPLE_SIZE = 100  # Run on only 100 samples
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # RF Hyperparameters for speed
    config.RF_PARAMS["n_estimators"] = 10
    config.RF_PARAMS["n_jobs"] = 1  # Avoid overhead

    # MLP Hyperparameters for speed
    config.MLP_PARAMS["epochs"] = 2
    config.MLP_PARAMS["batch_size"] = 16
    config.MLP_PARAMS["hidden_dim"] = 64  # Smaller model

    # Set seeds for reproducibility
    np.random.seed(config.RANDOM_STATE)
    torch.manual_seed(config.RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_STATE)

    # -------------------------------------------------------------------------
    # 2. FEATURE ENGINEERING
    # -------------------------------------------------------------------------
    print("\n[2] Generating features...")

    # Force fresh generation to verify pipeline logic (load_cached_data=False)
    # This will load raw CSVs, process text/history, and generate features.
    data_bundle = feature_factory.create_features(load_cached_data=False)

    # Unpack data bundle
    rf_data = data_bundle["rf"]
    mlp_data = data_bundle["mlp"]
    targets = data_bundle["targets"]
    dims = data_bundle["dims"]

    # Verify Shapes
    n_train = config.DEBUG_SAMPLE_SIZE
    print(f"    Verified Train Size: {rf_data['train'].shape[0]} (Expected ~{n_train})")
    print(f"    RF Feature Dim: {rf_data['train'].shape[1]}")
    print(f"    MLP Vocab Size: {dims['vocab_size']}")

    assert rf_data["train"].shape[0] == len(
        targets["train"]
    ), "RF Train features and targets mismatch"
    assert mlp_data["train"]["text"].shape[0] == len(
        targets["train"]
    ), "MLP Train features and targets mismatch"

    # -------------------------------------------------------------------------
    # 3. STREAM A: RANDOM FOREST MODEL
    # -------------------------------------------------------------------------
    print("\n[3] Training Random Forest (Stream A)...")

    rf = rf_model.LatentSemanticRF()
    rf.train(
        X_train=rf_data["train"],
        y_train=targets["train"],
        X_val=rf_data["val"],
        y_val=targets["val"],
    )

    # Generate RF Predictions
    rf_preds_test = rf.predict(rf_data["test"])

    # Verify Predictions
    assert (
        len(rf_preds_test) == rf_data["test"].shape[0]
    ), "RF prediction count mismatch"
    assert np.all(
        (rf_preds_test >= 0) & (rf_preds_test <= 1)
    ), "RF probabilities out of bounds"
    print(f"    RF Test Predictions Generated. Mean Prob: {np.mean(rf_preds_test):.4f}")

    # -------------------------------------------------------------------------
    # 4. STREAM B: RESIDUAL-ATTENTION MLP
    # -------------------------------------------------------------------------
    print("\n[4] Training Residual-Attention MLP (Stream B)...")

    mlp_trainer = mlp_model.MLPTrainer(dims=dims)
    mlp_trainer.train(
        train_data=mlp_data["train"],
        train_targets=targets["train"],
        val_data=mlp_data["val"],
        val_targets=targets["val"],
    )

    # Generate MLP Predictions
    mlp_preds_test = mlp_trainer.predict(mlp_data["test"])

    # Verify Predictions
    assert (
        len(mlp_preds_test) == mlp_data["test"]["text"].shape[0]
    ), "MLP prediction count mismatch"
    assert np.all(
        (mlp_preds_test >= 0) & (mlp_preds_test <= 1)
    ), "MLP probabilities out of bounds"
    print(
        f"    MLP Test Predictions Generated. Mean Prob: {np.mean(mlp_preds_test):.4f}"
    )

    # -------------------------------------------------------------------------
    # 5. ENSEMBLE & SUBMISSION
    # -------------------------------------------------------------------------
    print("\n[5] Ensembling and Creating Submission...")

    # Simple weighted average
    w_rf = config.ENSEMBLE_WEIGHTS["rf"]
    w_mlp = config.ENSEMBLE_WEIGHTS["mlp"]
    final_preds = (w_rf * rf_preds_test) + (w_mlp * mlp_preds_test)

    # Load Test IDs
    # Since we used DEBUG_SAMPLE_SIZE, we must load the corresponding IDs from the source file
    # utils.load_data logic applies head() after loading, so we do the same here.
    df_test_full = pd.read_csv(config.TEST_PATH)
    test_ids = df_test_full["request_id"].head(config.DEBUG_SAMPLE_SIZE).values

    assert len(test_ids) == len(
        final_preds
    ), f"ID count {len(test_ids)} != Pred count {len(final_preds)}"

    # Create DataFrame
    submission_df = pd.DataFrame(
        {"request_id": test_ids, "requester_received_pizza": final_preds}
    )

    # Save Submission
    submission_path = config.SUBMISSION_PATH
    submission_df.to_csv(submission_path, index=False)

    print(f"    Submission saved to: {submission_path}")
    print(f"    Submission shape: {submission_df.shape}")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
