import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
from scipy import sparse

# Import library modules
from library import (
    config,
    utils,
    data_loader,
    feature_engineering,
    neural_net,
    train_eval,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_subset_data(data_dict, num_samples):
    """
    Helper function to slice the data dictionary (dense, tfidf, embedding, y/ids)
    to a smaller number of samples for demonstration purposes.
    """
    subset = {}
    for key, value in data_dict.items():
        if value is None:
            subset[key] = None
            continue

        # Handle sparse matrices (TF-IDF)
        if sparse.issparse(value):
            subset[key] = value[:num_samples]
        # Handle numpy arrays / tensors
        elif isinstance(value, (np.ndarray, list, pd.Series)):
            subset[key] = value[:num_samples]
        else:
            # Pass through other types if any
            subset[key] = value

    return subset


def main():
    print("=== Starting Library Usage Demonstration ===")

    # 1. Setup and Configuration Overrides
    print("\n[Step 1] Configuring environment and overriding hyperparameters...")
    utils.set_seed(42)

    # Override config for speed
    config.RF_CONFIG["n_estimators"] = 10  # Reduce trees
    config.RF_CONFIG["n_jobs"] = 1  # Avoid overhead in demo

    config.MLP_CONFIG["epochs"] = 2  # Minimal epochs
    config.MLP_CONFIG["batch_size"] = 8  # Small batch for small subset
    config.MLP_CONFIG["patience"] = 1  # Fail fast if needed

    # Ensure working directory exists for demo outputs
    demo_output_dir = "./working/demo_output"
    os.makedirs(demo_output_dir, exist_ok=True)

    # 2. Data Processing
    print("\n[Step 2] Processing features...")
    processor = feature_engineering.FeatureProcessor()

    # Load and process data (this handles loading metadata, cleaning, FE, and embedding)
    # We allow loading from cache if available to speed up, otherwise it computes.
    data = processor.process_data(load_cached_data=True)

    # Validate data structure
    assert "train" in data and "val" in data and "test" in data
    assert "dense" in data["train"]
    assert "tfidf" in data["train"]
    assert "embedding" in data["train"]
    assert "y" in data["train"]

    print("Data processed successfully.")
    print(f"Original Train Shape: {data['train']['dense'].shape}")

    # 3. Create Mini-Datasets for Speed
    print("\n[Step 3] Creating mini-datasets for rapid demonstration...")
    n_train = 50
    n_val = 20
    n_test = 20

    mini_train = create_subset_data(data["train"], n_train)
    mini_val = create_subset_data(data["val"], n_val)
    mini_test = create_subset_data(data["test"], n_test)

    # Assertions to verify subsetting
    assert mini_train["dense"].shape[0] == n_train
    assert mini_train["y"].shape[0] == n_train
    assert mini_val["embedding"].shape[0] == n_val
    assert mini_test["ids"].shape[0] == n_test
    print(f"Mini-Train samples: {n_train}, Mini-Val samples: {n_val}")

    # 4. Train Random Forest
    print("\n[Step 4] Training Random Forest on mini-dataset...")
    rf_model = train_eval.train_rf(mini_train, mini_val)

    # Verify RF Model
    assert hasattr(
        rf_model, "predict_proba"
    ), "RF model should have predict_proba method"
    print("Random Forest training complete.")

    # 5. Train Dual-Branch MLP
    print("\n[Step 5] Training Dual-Branch MLP on mini-dataset...")
    # The wrapper handles initialization and training loop
    mlp_model = train_eval.train_mlp_wrapper(mini_train, mini_val)

    # Verify MLP Model
    assert isinstance(
        mlp_model, neural_net.DualBranchMLP
    ), "Model should be instance of DualBranchMLP"
    print("MLP training complete.")

    # 6. Ensemble Prediction
    print("\n[Step 6] Generating ensemble predictions on mini-test set...")
    # predict_ensemble handles the weighted averaging of RF and MLP
    test_probs = train_eval.predict_ensemble(rf_model, mlp_model, mini_test)

    # Verify Predictions
    assert (
        len(test_probs) == n_test
    ), f"Expected {n_test} predictions, got {len(test_probs)}"
    assert np.all(
        (test_probs >= 0) & (test_probs <= 1)
    ), "Probabilities must be between 0 and 1"
    print(f"Predictions generated. Mean probability: {np.mean(test_probs):.4f}")

    # 7. Save Submission
    print("\n[Step 7] Saving submission file...")
    submission_path = os.path.join(demo_output_dir, "demo_submission.csv")
    utils.save_submission(test_probs, mini_test["ids"], save_path=submission_path)

    # Verify File Creation
    assert os.path.exists(submission_path), "Submission file was not created"

    # Verify Content
    df_sub = pd.read_csv(submission_path)
    assert list(df_sub.columns) == [
        "request_id",
        "requester_received_pizza",
    ], "Incorrect submission columns"
    assert len(df_sub) == n_test, "Incorrect number of rows in submission"

    print(f"Submission saved and verified at: {submission_path}")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
