import os
import sys
import pandas as pd
import numpy as np
import warnings

# Import provided library modules
import library.config
import library.utils
import library.feature_engineering
import library.training


def run_demonstration():
    # 1. Setup and Configuration Override
    # We override hyperparameters to ensure the script runs quickly as per requirements.
    # Since TRAIN_PARAMS and RF_PARAMS are mutable dictionaries, changes here
    # will be reflected when library.training uses them.
    print("Configuring hyperparameters for fast execution...")

    # Reduce Random Forest complexity for speed
    library.config.RF_PARAMS["n_estimators"] = 10

    # Reduce MLP training duration (epochs and patience)
    library.config.TRAIN_PARAMS["epochs"] = 2
    library.config.TRAIN_PARAMS["patience"] = 1
    library.config.TRAIN_PARAMS["batch_size"] = 32

    # 2. Feature Engineering
    # Instantiate the Preprocessor class
    print("Initializing Preprocessor...")
    preprocessor = library.feature_engineering.Preprocessor()

    # Run the processing pipeline.
    # We set load_cached_data=False to force the execution of the feature engineering logic
    # (TF-IDF, SBERT, Tabular processing) rather than loading potentially incompatible cached files.
    print("Running feature engineering pipeline...")
    data = preprocessor.run(load_cached_data=False)

    # 3. Verify Data Integrity
    print("Verifying processed data integrity...")

    # Check for existence of all required keys in the returned dictionary
    required_keys = [
        "rf_train_tab",
        "rf_val_tab",
        "rf_test_tab",
        "rf_train_tfidf",
        "rf_val_tfidf",
        "rf_test_tfidf",
        "mlp_train_tab",
        "mlp_val_tab",
        "mlp_test_tab",
        "mlp_train_sbert",
        "mlp_val_sbert",
        "mlp_test_sbert",
        "y_train",
        "y_val",
        "test_ids",
    ]

    for key in required_keys:
        if key not in data:
            raise AssertionError(f"Missing key in processed data: {key}")

    # Check dimensions consistency (Train set)
    n_train = len(data["y_train"])
    assert (
        data["rf_train_tab"].shape[0] == n_train
    ), "RF Train Tabular row count mismatch"
    assert (
        data["rf_train_tfidf"].shape[0] == n_train
    ), "RF Train TFIDF row count mismatch"
    assert (
        data["mlp_train_tab"].shape[0] == n_train
    ), "MLP Train Tabular row count mismatch"
    assert (
        data["mlp_train_sbert"].shape[0] == n_train
    ), "MLP Train SBERT row count mismatch"

    # Check dimensions consistency (Test set)
    n_test = len(data["test_ids"])
    assert data["rf_test_tab"].shape[0] == n_test, "RF Test Tabular row count mismatch"

    print(f"Data verification passed. Training on {n_train} samples.")

    # 4. Model Training and Inference
    # Execute the training pipeline which trains RF and MLP, then ensembles them.
    # This function handles model instantiation, training loops, and prediction generation.
    print("Starting training and inference...")
    library.training.run_training(data)

    # 5. Verify Submission Output
    print("Verifying submission file...")
    submission_path = library.config.SUBMISSION_PATH

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file was not created at {submission_path}")

    df_submission = pd.read_csv(submission_path)

    # Check columns
    expected_cols = ["request_id", "requester_received_pizza"]
    if not all(col in df_submission.columns for col in expected_cols):
        raise AssertionError(
            f"Submission file missing required columns. Found: {df_submission.columns}"
        )

    # Check row count
    if len(df_submission) != n_test:
        raise AssertionError(
            f"Submission row count ({len(df_submission)}) does not match test set size ({n_test})"
        )

    # Check value range (probabilities should be between 0 and 1)
    probs = df_submission["requester_received_pizza"]
    if not (probs.between(0, 1).all()):
        raise AssertionError("Submission contains probabilities outside [0, 1] range")

    print("Submission verification passed.")
    print("Demonstration completed successfully.")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Run the main logic
    run_demonstration()
