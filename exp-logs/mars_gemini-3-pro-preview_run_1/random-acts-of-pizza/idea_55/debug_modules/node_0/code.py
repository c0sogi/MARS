import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library
from library.config import Config
from library.utils import set_seed, save_submission
from library.feature_engineering import FeaturePipeline
from library.model_rf import train_rf_model
from library.model_mlp import train_mlp_model, predict_mlp

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("--- Starting Demonstration Script ---")

    # 1. Configure for Speed and Debugging
    # We override Config attributes to run on a small subset with fewer iterations
    print("Configuring parameters for fast execution...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Process only 50 samples per split
    Config.MLP_EPOCHS = 2  # Train MLP for only 2 epochs
    Config.MLP_BATCH_SIZE = 8  # Small batch size for the small dataset
    Config.RF_N_ESTIMATORS = 10  # Fewer trees for RF
    Config.CACHE_DIR = "./working/demo_cache/"
    Config.SUBMISSION_FILE = "./working/demo_submission.csv"

    # Ensure clean state for demo cache
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Set global seed for reproducibility
    set_seed(Config.SEED)

    # 2. Feature Engineering
    print("\n--- Step 1: Feature Engineering ---")
    pipeline = FeaturePipeline()

    # Force fresh processing (load_cached_data=False) to demonstrate the logic.
    # This triggers loading raw data, text embedding generation, and metadata processing.
    data = pipeline.process_data(load_cached_data=False)

    (
        X_rf_train,
        X_rf_val,
        X_rf_test,
        X_mlp_train,
        X_mlp_val,
        X_mlp_test,
        y_train,
        y_val,
    ) = data

    # Validation of Feature Engineering Outputs
    print("Validating feature shapes...")

    # RF Features: Should be a 2D numpy array (N_samples, N_features)
    assert isinstance(X_rf_train, np.ndarray), "RF Train features must be a numpy array"
    assert X_rf_train.ndim == 2, "RF Train features should be 2D"
    assert (
        X_rf_train.shape[0] == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} training samples"

    # MLP Features: Should be a dictionary of tensors
    assert isinstance(X_mlp_train, dict), "MLP Train features should be a dictionary"
    assert "title_emb" in X_mlp_train, "MLP features missing 'title_emb'"
    assert isinstance(
        X_mlp_train["title_emb"], torch.Tensor
    ), "MLP features should be tensors"
    assert X_mlp_train["title_emb"].shape[0] == Config.DEBUG_SAMPLE_SIZE

    # Targets
    assert y_train.shape[0] == Config.DEBUG_SAMPLE_SIZE
    assert set(np.unique(y_train)).issubset({0, 1}), "Targets must be binary integers"

    print("Feature Engineering validation passed.")

    # 3. Random Forest Model
    print("\n--- Step 2: Random Forest Model ---")

    # Train the Random Forest model
    rf_model = train_rf_model(X_rf_train, y_train, X_rf_val, y_val)

    # Generate predictions on the test set
    rf_probs_test = rf_model.predict_proba(X_rf_test)

    # Validate RF Predictions
    assert (
        len(rf_probs_test) == Config.DEBUG_SAMPLE_SIZE
    ), "RF prediction count mismatch"
    assert np.all(
        (rf_probs_test >= 0) & (rf_probs_test <= 1)
    ), "RF probabilities out of range [0, 1]"
    print(f"RF Test Predictions (First 5): {rf_probs_test[:5]}")

    # 4. MLP Model
    print("\n--- Step 3: MLP Model ---")

    # Train the MLP model
    # train_mlp_model returns the trained model and the trainer instance
    mlp_model, mlp_trainer = train_mlp_model(X_mlp_train, y_train, X_mlp_val, y_val)

    # Generate predictions on the test set
    mlp_probs_test = predict_mlp(mlp_model, X_mlp_test)

    # Validate MLP Predictions
    assert (
        len(mlp_probs_test) == Config.DEBUG_SAMPLE_SIZE
    ), "MLP prediction count mismatch"
    assert np.all(
        (mlp_probs_test >= 0) & (mlp_probs_test <= 1)
    ), "MLP probabilities out of range [0, 1]"
    print(f"MLP Test Predictions (First 5): {mlp_probs_test[:5]}")

    # 5. Ensemble and Submission
    print("\n--- Step 4: Ensemble & Submission ---")

    # Calculate Weighted Average Ensemble
    w_rf, w_mlp = Config.ENSEMBLE_WEIGHTS
    final_probs = (w_rf * rf_probs_test) + (w_mlp * mlp_probs_test)

    # Retrieve Request IDs corresponding to the test set
    # In DEBUG mode, we process the first N rows, so we read the metadata file and take the head.
    df_test_full = pd.read_csv(Config.TEST_DATA_PATH)
    test_ids = df_test_full["request_id"].head(Config.DEBUG_SAMPLE_SIZE).tolist()

    assert len(test_ids) == len(final_probs), "ID count matches prediction count"

    # Save the submission file
    save_submission(test_ids, final_probs, Config.SUBMISSION_FILE)

    # Verify File Creation and Content
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created"

    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    assert df_sub.shape == (
        Config.DEBUG_SAMPLE_SIZE,
        2,
    ), "Submission file shape mismatch"
    assert list(df_sub.columns) == [
        "request_id",
        "requester_received_pizza",
    ], "Submission columns mismatch"

    print("Demo completed successfully!")


if __name__ == "__main__":
    run_demo()
