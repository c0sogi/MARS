import os
import numpy as np
import pandas as pd
import warnings

# Import provided library modules
from library.config import Config
from library.data_processor import LeafDataManager
from library.model_library import ModelFactory
from library.ensemble_selection import EnsembleSelector


def run_pipeline_demo():
    # 1. Configuration and Setup
    print("1. Initializing Configuration...")
    Config.setup()

    # Override SELECTION_ITERATIONS to a smaller number for this demo execution
    # to ensure it completes very quickly while still demonstrating the logic.
    Config.SELECTION_ITERATIONS = 5

    # Set random seeds for reproducibility
    np.random.seed(Config.RANDOM_SEED)

    # 2. Data Loading and Processing
    print("\n2. Processing Data...")
    data_manager = LeafDataManager()

    # We force load_cached_data=False to demonstrate the feature extraction pipeline.
    # In a production scenario, this would be True to save time.
    data = data_manager.load_and_process_data(load_cached_data=False)

    # Validation: Ensure data dictionary structure is correct
    required_keys = ["train", "val", "test", "classes"]
    for key in required_keys:
        assert key in data, f"Data dictionary missing key: {key}"

    # Validation: Check shapes of loaded data
    n_train_samples = len(data["train"]["ids"])
    n_classes = len(data["classes"])
    print(f"   Loaded {n_train_samples} training samples with {n_classes} classes.")

    # Validation: Verify Macro feature extraction (should not be all zeros)
    # Macro features are: 7 Hu moments + 4 Geometric scalars = 11 features
    X_macro_train = data["train"]["X_macro"]
    assert (
        X_macro_train.shape[1] == 11
    ), f"Expected 11 macro features, found {X_macro_train.shape[1]}"
    # Check if we have non-zero values (indicating successful extraction from images)
    assert (
        np.sum(np.abs(X_macro_train)) > 0
    ), "Macro features are all zeros. Image processing may have failed."

    # 3. Expert Generation
    print("\n3. Generating Model Experts...")
    experts = ModelFactory.get_experts()
    print(f"   Created {len(experts)} experts across different tiers and views.")

    # Validation: Ensure we have experts
    assert len(experts) > 0, "ModelFactory returned no experts."

    # 4. Ensemble Selection (Training Phase)
    print("\n4. Running Ensemble Selection (Greedy Forward Selection)...")
    selector = EnsembleSelector(experts)

    # Fit the selector: Trains experts on Train split, evaluates on Val split,
    # and selects the best combination.
    selector.fit(data)

    # Validation: Check if selection occurred
    print(f"   Selected Ensemble: {selector.selected_experts}")
    assert (
        len(selector.selected_experts) == Config.SELECTION_ITERATIONS
    ), f"Expected {Config.SELECTION_ITERATIONS} selected experts, got {len(selector.selected_experts)}"

    # 5. Final Retraining and Inference
    print("\n5. Retraining and Predicting...")
    # This step combines Train + Val, retrains the selected experts, and predicts on Test.
    test_probs = selector.refit_and_predict(data)

    # Validation: Check prediction shape and constraints
    n_test_samples = len(data["test"]["ids"])
    assert test_probs.shape == (
        n_test_samples,
        n_classes,
    ), f"Prediction shape mismatch. Expected ({n_test_samples}, {n_classes}), got {test_probs.shape}"

    # Check probability validity
    assert np.all(test_probs >= 0) and np.all(
        test_probs <= 1
    ), "Predictions contain values outside [0, 1]"

    # 6. Submission Generation
    print("\n6. Creating Submission File...")
    submission_df = pd.DataFrame(test_probs, columns=data["classes"])
    # Insert 'id' column at the beginning
    submission_df.insert(0, "id", data["test"]["ids"])

    # Save to disk
    submission_path = Config.SUBMISSION_PATH
    submission_df.to_csv(submission_path, index=False)
    print(f"   Saved submission to {submission_path}")

    # Validation: Verify file integrity against sample submission
    assert os.path.exists(submission_path), "Submission file was not created."

    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
    assert (
        submission_df.shape == sample_sub.shape
    ), f"Submission shape {submission_df.shape} does not match sample {sample_sub.shape}"

    # Ensure columns match (order might differ in values, but headers must match)
    # We sort columns to compare sets, excluding ID which is already checked
    sub_cols = sorted([c for c in submission_df.columns if c != "id"])
    sample_cols = sorted([c for c in sample_sub.columns if c != "id"])
    assert (
        sub_cols == sample_cols
    ), "Submission columns do not match sample submission requirements."

    print("\nPipeline execution completed successfully.")


if __name__ == "__main__":
    # Suppress warnings for cleaner output (e.g. sklearn convergence warnings if any)
    warnings.filterwarnings("ignore")
    run_pipeline_demo()
