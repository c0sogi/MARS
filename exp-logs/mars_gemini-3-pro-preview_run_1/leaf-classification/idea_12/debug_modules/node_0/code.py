import os
import sys
import numpy as np
import pandas as pd
import random

# Import from the provided library files
from library.config import Config
from library.data_loader import load_datasets
from library.transductive_preprocessor import TransductivePipeline
from library.model import RobustLDA


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_demo():
    print("--- Starting Leaf Classification Demo ---")

    # 1. Setup
    set_seed(Config.SEED)

    # 2. Data Loading
    print("\n[Step 1] Loading Datasets...")
    # We set load_cached_data=False to demonstrate the processing from metadata
    (X_train, y_train, ids_train), (X_val, y_val, ids_val), (X_test, ids_test) = (
        load_datasets(load_cached_data=False)
    )

    # Validation of loaded data
    print(
        f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}"
    )

    # Assertions to verify data integrity
    assert len(X_train) == len(y_train) == len(ids_train), "Training dimension mismatch"
    assert len(X_val) == len(y_val) == len(ids_val), "Validation dimension mismatch"
    assert len(X_test) == len(ids_test), "Test dimension mismatch"

    # Verify feature columns match the deterministic schema in Config
    assert (
        list(X_train.columns) == Config.FEATURES
    ), "Feature order in Train does not match Config"
    assert (
        list(X_test.columns) == Config.FEATURES
    ), "Feature order in Test does not match Config"

    # 3. Transductive Preprocessing
    print(
        "\n[Step 2] Running Transductive Pipeline (PowerTransform -> Scaling -> PCA)..."
    )
    pipeline = TransductivePipeline()

    # Fit on combined data, transform individual sets
    # Note: We force re-computation by ignoring cache if it exists for this demo
    X_train_trans, X_val_trans, X_test_trans = pipeline.fit_transform_combined(
        X_train, X_val, X_test, load_cached_data=False
    )

    print(f"Transformed Train shape: {X_train_trans.shape}")
    print(f"Transformed Val shape:   {X_val_trans.shape}")
    print(f"Transformed Test shape:  {X_test_trans.shape}")

    # Assertions for preprocessing
    assert (
        X_train_trans.shape[1] == X_val_trans.shape[1] == X_test_trans.shape[1]
    ), "PCA component mismatch"
    assert not np.isnan(X_train_trans).any(), "NaNs found in transformed training data"
    assert isinstance(X_train_trans, np.ndarray), "Output should be numpy array"

    # 4. Model Training & Evaluation
    print("\n[Step 3] Training and Evaluating RobustLDA Model...")
    model = RobustLDA()

    # Fit
    model.fit(X_train_trans, y_train)

    # Evaluate
    metrics = model.evaluate(X_val_trans, y_val, dataset_name="Validation")

    # Assertions for model performance
    # Random guess for ~100 classes is ~1%. We expect much higher.
    assert (
        metrics["accuracy"] > 0.10
    ), f"Model accuracy {metrics['accuracy']} is too low, something is wrong."
    assert (
        metrics["log_loss"] < 5.0
    ), f"Log loss {metrics['log_loss']} is suspiciously high."

    # 5. Prediction & Submission Generation
    print("\n[Step 4] Generating Submission...")

    # Predict probabilities on test set
    test_probs = model.predict_proba(X_test_trans)

    # Verify probability properties
    # Sum of probs for a row should be approx 1.0
    row_sums = np.sum(test_probs, axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1"
    assert test_probs.shape[0] == len(
        ids_test
    ), "Prediction count matches test ID count"
    assert test_probs.shape[1] == len(
        model.classes_
    ), "Probability columns match class count"

    # Create Submission DataFrame
    submission_df = pd.DataFrame(test_probs, columns=model.classes_)
    submission_df.insert(0, "id", ids_test)

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save
    submission_path = Config.SUBMISSION_PATH
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to: {submission_path}")

    # Final Validation
    assert os.path.exists(submission_path), "Submission file was not created"

    # Check loaded submission
    saved_df = pd.read_csv(submission_path)
    assert saved_df.shape == (
        99,
        100,
    ), f"Submission shape incorrect. Expected (99, 100), got {saved_df.shape}"
    assert "id" in saved_df.columns, "id column missing from submission"

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
