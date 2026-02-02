import os
import sys
import numpy as np
import pandas as pd
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library files
import library.config as config
from library.data_loader import load_data
from library.preprocessing import preprocess_data
from library.model import EnergyPredictor


def run_demo():
    print("=== Starting End-to-End Pipeline Demonstration ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed
    # ---------------------------------------------------------
    # Modify XGBoost parameters in memory to make training very fast for this demo
    print("Configuring hyperparameters for fast execution...")
    config.XGB_PARAMS["n_estimators"] = 10
    config.XGB_PARAMS["max_depth"] = 3
    config.XGB_PARAMS["learning_rate"] = 0.1

    # Ensure we use the debug flag for data loading to process only a subset
    DEBUG_MODE = True
    SAMPLE_SIZE = 50

    # ---------------------------------------------------------
    # 2. Data Loading and Feature Extraction
    # ---------------------------------------------------------
    print("\n[Step 1] Loading Data and Extracting Features...")
    # We set load_cached_data=False to demonstrate the feature extraction logic runs correctly
    # In a real run, you might set this to True to save time.
    train_df, val_df, test_df = load_data(
        load_cached_data=False, debug=DEBUG_MODE, sample_size=SAMPLE_SIZE
    )

    # Verification
    assert not train_df.empty, "Training dataframe is empty"
    assert not val_df.empty, "Validation dataframe is empty"
    assert not test_df.empty, "Test dataframe is empty"

    # Check if features were actually created (e.g., RDF columns)
    rdf_cols = [c for c in train_df.columns if "rdf" in c]
    assert len(rdf_cols) > 0, "No RDF features were generated"
    print(
        f"Successfully loaded {len(train_df)} training samples with {train_df.shape[1]} features."
    )

    # ---------------------------------------------------------
    # 3. Preprocessing
    # ---------------------------------------------------------
    print("\n[Step 2] Preprocessing Data...")
    # This handles log-transform of targets and dropping constant columns
    X_train, y_train, X_val, y_val, X_test = preprocess_data(
        train_df, val_df, test_df, load_cached_data=False
    )

    # Verification
    assert X_train.shape[0] == y_train.shape[0], "Mismatch in X_train and y_train rows"
    assert X_val.shape[0] == y_val.shape[0], "Mismatch in X_val and y_val rows"
    # Ensure targets are log transformed (should be relatively small values)
    # Original formation energies are ~0.2, log1p(0.2) ~ 0.18.
    # If they were huge, log transform might have failed or data is weird.
    assert (
        y_train["formation_energy_ev_natom"].max() < 10.0
    ), "Target values seem unusually large for log-transformed data"

    print(f"Preprocessed Train shape: {X_train.shape}")
    print(f"Preprocessed Val shape:   {X_val.shape}")

    # ---------------------------------------------------------
    # 4. Model Training
    # ---------------------------------------------------------
    print("\n[Step 3] Training Models...")
    predictor = EnergyPredictor()
    predictor.train(X_train, y_train, X_val, y_val, verbose=False)

    # Verification
    assert (
        "formation_energy_ev_natom" in predictor.models
    ), "Model for formation energy not trained"
    assert (
        "bandgap_energy_ev" in predictor.models
    ), "Model for bandgap energy not trained"
    print("Models trained successfully.")

    # ---------------------------------------------------------
    # 5. Prediction and Submission Generation
    # ---------------------------------------------------------
    print("\n[Step 4] Generating Predictions...")
    submission_df = predictor.predict(X_test)

    # Verification
    expected_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
    assert (
        list(submission_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(submission_df.columns)}"
    assert len(submission_df) == len(X_test), "Submission row count mismatch"

    # Check for validity of predictions (e.g., non-negative bandgap)
    # Note: The model predicts in log space and inverse transforms, so values should be positive (exp(x) - 1 >= -1, but usually > 0 for these targets)
    assert (
        submission_df["bandgap_energy_ev"] > -1.0
    ).all(), "Predicted bandgap energies are physically invalid (too negative)"

    print("Sample Predictions:")
    print(submission_df.head())

    # Save submission
    sub_path = os.path.join(config.SUBMISSION_DIR, "demo_submission.csv")
    submission_df.to_csv(sub_path, index=False)
    print(f"\nSubmission saved to {sub_path}")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo()
