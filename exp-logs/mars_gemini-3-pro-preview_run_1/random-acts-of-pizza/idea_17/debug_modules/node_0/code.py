import os
import sys
import numpy as np
import pandas as pd
import torch

# Import library modules
from library.config import Config
from library.feature_engineering import FeaturePipeline
from library.model_rf import train_rf_model, predict_rf_model
from library.model_mlp import train_mlp_model, predict_mlp_model
from library.data_loader import load_datasets


def run_demo():
    print("Starting demonstration of the Pizza Success Prediction Pipeline...")

    # ==========================================
    # 1. Configuration Override for Speed
    # ==========================================
    print("Configuring environment for fast execution...")

    # Enable DEBUG mode to use a small subset of data (100 samples)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 100

    # Change working directory to avoid conflicts with production runs
    Config.WORKING_DIR = "./working/demo_execution"
    Config.RF_FEATURES_PATH = os.path.join(Config.WORKING_DIR, "rf_features.npz")
    Config.MLP_FEATURES_PATH = os.path.join(Config.WORKING_DIR, "mlp_features.npz")

    # Reduce Random Forest complexity
    Config.RF_N_ESTIMATORS = 10
    Config.RF_TFIDF_MAX_FEATURES = 500  # Reduce vocabulary size

    # Reduce MLP Training duration
    Config.MLP_EPOCHS = 2
    Config.MLP_PATIENCE = 1
    Config.MLP_HIDDEN_DIM = 64  # Smaller network

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # 2. Feature Engineering
    # ==========================================
    print("\n--- Step 2: Feature Engineering ---")
    pipeline = FeaturePipeline()

    # Force re-computation to demonstrate the pipeline logic (load_cached_data=False)
    rf_features, mlp_features = pipeline.run(load_cached_data=False)

    # Verify RF Feature Structure
    assert "X_train" in rf_features, "RF features missing X_train"
    assert "y_train" in rf_features, "RF features missing y_train"
    assert (
        rf_features["X_train"].shape[0] == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} training samples, got {rf_features['X_train'].shape[0]}"

    # Verify MLP Feature Structure
    assert "text_train" in mlp_features, "MLP features missing text_train"
    assert "hist_train" in mlp_features, "MLP features missing hist_train"
    assert (
        mlp_features["text_train"].shape[0] == Config.DEBUG_SAMPLE_SIZE
    ), "MLP text features dimension mismatch"

    print("Feature engineering completed and verified.")

    # ==========================================
    # 3. Model Training: Random Forest
    # ==========================================
    print("\n--- Step 3: Training Random Forest ---")
    rf_model = train_rf_model(
        rf_features["X_train"],
        rf_features["y_train"],
        rf_features["X_val"],
        rf_features["y_val"],
    )

    assert rf_model is not None, "Random Forest model failed to initialize."
    print("Random Forest trained successfully.")

    # ==========================================
    # 4. Model Training: Attention-Gated MLP
    # ==========================================
    print("\n--- Step 4: Training MLP ---")
    mlp_model = train_mlp_model(mlp_features)

    assert isinstance(mlp_model, torch.nn.Module), "MLP model is not a PyTorch module."
    print("MLP trained successfully.")

    # ==========================================
    # 5. Prediction & Ensembling
    # ==========================================
    print("\n--- Step 5: Inference and Ensembling ---")

    # Generate predictions
    rf_preds = predict_rf_model(rf_model, rf_features["X_test"])
    mlp_preds = predict_mlp_model(mlp_model, mlp_features)

    # Verify prediction shapes
    expected_test_size = Config.DEBUG_SAMPLE_SIZE
    assert (
        len(rf_preds) == expected_test_size
    ), f"RF preds shape mismatch: {len(rf_preds)}"
    assert (
        len(mlp_preds) == expected_test_size
    ), f"MLP preds shape mismatch: {len(mlp_preds)}"

    # Check probability range
    assert np.all(
        (rf_preds >= 0) & (rf_preds <= 1)
    ), "RF predictions out of probability range [0, 1]"
    assert np.all(
        (mlp_preds >= 0) & (mlp_preds <= 1)
    ), "MLP predictions out of probability range [0, 1]"

    # Weighted Ensemble
    w_rf = Config.ENSEMBLE_WEIGHTS["rf"]
    w_mlp = Config.ENSEMBLE_WEIGHTS["mlp"]
    final_preds = (w_rf * rf_preds) + (w_mlp * mlp_preds)

    print(f"Ensemble weights -> RF: {w_rf}, MLP: {w_mlp}")
    print(f"First 5 predictions: {final_preds[:5]}")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    print("\n--- Step 6: Generating Submission File ---")

    # Load test data to get request_ids
    # Note: In DEBUG mode, load_datasets returns the sampled subset used for feature generation
    _, _, df_test = load_datasets()

    submission_df = pd.DataFrame(
        {"request_id": df_test[Config.ID_COL], "requester_received_pizza": final_preds}
    )

    # Save submission
    demo_submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(demo_submission_path, index=False)

    print(f"Submission saved to: {demo_submission_path}")

    # Verify file content
    saved_df = pd.read_csv(demo_submission_path)
    assert (
        len(saved_df) == expected_test_size
    ), "Saved submission has incorrect number of rows."
    assert "request_id" in saved_df.columns, "Submission missing request_id column."
    assert (
        "requester_received_pizza" in saved_df.columns
    ), "Submission missing target column."

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    run_demo()
