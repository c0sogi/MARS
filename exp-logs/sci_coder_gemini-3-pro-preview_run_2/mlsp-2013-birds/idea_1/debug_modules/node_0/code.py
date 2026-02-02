import os
import sys
import numpy as np
import pandas as pd
import random
import torch

# Ensure the current directory is in the python path to import library modules
sys.path.append(os.getcwd())

from library.config import Config
from library.data_loader import HistogramDataLoader
from library.model import BirdRandomForest
from library.trainer import Trainer


def set_seed(seed=42):
    """Sets fixed random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("Starting Library Usage Demonstration...")
    set_seed(42)

    # ==========================================
    # 1. Configuration Optimization
    # ==========================================
    print("\n[1] Optimizing Configuration for Speed")
    # Modify the global configuration to use a lightweight model for this demo
    # Reducing n_estimators from 500 to 10 significantly speeds up execution
    original_n_estimators = Config.RF_PARAMS["n_estimators"]
    Config.RF_PARAMS["n_estimators"] = 10
    Config.RF_PARAMS["n_jobs"] = 1  # Reduce overhead for small data
    print(
        f"Reduced RF n_estimators from {original_n_estimators} to {Config.RF_PARAMS['n_estimators']}"
    )

    # ==========================================
    # 2. Data Loader Demonstration
    # ==========================================
    print("\n[2] Testing HistogramDataLoader")
    loader = HistogramDataLoader()

    # Load data splits (Train, Val, Test)
    # We force load_cached_data=False to ensure the raw parsing logic is tested
    (X_train, y_train), (X_val, y_val), (X_test, test_ids) = loader.get_data_splits(
        load_cached_data=False
    )

    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Val shape:   X={X_val.shape}, y={y_val.shape}")
    print(f"Test shape:  X={X_test.shape}, ids={test_ids.shape}")

    # Assertions to verify data integrity
    assert X_train.shape[0] == y_train.shape[0], "Mismatch in training samples/labels"
    assert X_val.shape[0] == y_val.shape[0], "Mismatch in validation samples/labels"
    assert (
        X_train.shape[1] == 100
    ), f"Expected 100 features (histogram bins), got {X_train.shape[1]}"
    assert (
        y_train.shape[1] == Config.NUM_SPECIES
    ), f"Expected {Config.NUM_SPECIES} species labels, got {y_train.shape[1]}"
    assert len(test_ids) == X_test.shape[0], "Mismatch in test IDs count"

    print("Data Loader assertions passed.")

    # ==========================================
    # 3. Model Class Demonstration
    # ==========================================
    print("\n[3] Testing BirdRandomForest Model")
    model = BirdRandomForest()

    # Fit the model on the training data
    print("Fitting model...")
    model.fit(X_train, y_train)

    # Predict probabilities on validation data
    print("Predicting probabilities...")
    y_pred_proba = model.predict_proba(X_val)

    # Assertions to verify model output
    assert (
        y_pred_proba.shape == y_val.shape
    ), f"Prediction shape mismatch: {y_pred_proba.shape} vs {y_val.shape}"
    assert (
        y_pred_proba.min() >= 0.0 and y_pred_proba.max() <= 1.0
    ), "Probabilities must be between 0 and 1"

    # Check persistence (Save/Load)
    model_path = os.path.join(Config.WORKING_DIR, "demo_model.joblib")
    model.save(model_path)
    assert os.path.exists(model_path), "Model file was not saved"

    loaded_model = BirdRandomForest.load(model_path)
    assert loaded_model is not None, "Failed to load model"

    print("Model assertions passed.")

    # ==========================================
    # 4. Trainer Class Demonstration
    # ==========================================
    print("\n[4] Testing Trainer (Integration)")
    trainer = Trainer()

    # Run training pipeline
    # This uses the data loader and model internally
    auc_score = trainer.train(load_cached_data=True)  # Use cache this time
    print(f"Trainer returned AUC: {auc_score:.4f}")

    # Assert reasonable AUC (it might be 0.0 or low due to small data/estimators, but should be a float)
    assert isinstance(auc_score, float), "AUC score should be a float"

    # Generate Submission
    trainer.generate_submission(load_cached_data=True)

    # Verify Submission File
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission file loaded. Shape: {df_sub.shape}")

    # Verify submission format
    assert list(df_sub.columns) == ["Id", "Probability"], "Submission columns mismatch"
    assert (
        df_sub["Id"].dtype == "int64" or df_sub["Id"].dtype == "int32"
    ), "Id column should be integer"

    # Check if number of rows matches Test Samples * Num Species
    expected_rows = X_test.shape[0] * Config.NUM_SPECIES
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(df_sub)}"

    # Check probability range
    assert df_sub["Probability"].min() >= 0.0, "Negative probability found"
    assert df_sub["Probability"].max() <= 1.0, "Probability > 1.0 found"

    print("Trainer assertions passed.")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
