import os
import numpy as np
import pandas as pd
import warnings
import shutil
from library.config import (
    set_seed,
    WORKING_DIR,
    SUBMISSION_FILE_PATH,
    FEATURE_PREFIXES,
    TRAIN_DATA_PATH,
)
from library.data_loader import load_and_process_data, SemanticPreprocessor
from library.factorized_lda import (
    FactorizedOASDiscriminant,
    run_factorized_lda_pipeline,
)
from library.utils import calculate_log_loss, save_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def demo_implementation():
    print("--- Starting Library Usage Demonstration ---")

    # 1. Setup and Reproducibility
    set_seed(42)
    print("Seed set for reproducibility.")

    # 2. Data Loading and Preprocessing Demonstration
    print("\n[Demo] Data Loading and Processing...")

    # We force reload to demonstrate the processing logic (SemanticPreprocessor)
    # This generates X matrices for margin, shape, and texture groups.
    data, classes = load_and_process_data(load_cached_data=False)

    # Verify Data Structure
    assert (
        "train" in data and "val" in data and "test" in data
    ), "Data dictionary missing splits."
    assert (
        "X" in data["train"] and "y" in data["train"]
    ), "Train split missing features or labels."

    # Verify Feature Groups (Semantic Splitting)
    X_train = data["train"]["X"]
    for group in FEATURE_PREFIXES:
        assert group in X_train, f"Feature group '{group}' missing from processed data."
        assert (
            X_train[group].shape[1] == 64
        ), f"Feature group '{group}' should have 64 columns."

    print(f"Data loaded successfully. Classes: {len(classes)}")
    print(
        f"Train samples: {len(data['train']['y'])}, Val samples: {len(data['val']['y'])}"
    )

    # 3. Semantic Preprocessor Independent Verification
    # Demonstrating how the preprocessor works on raw dataframes
    print("\n[Demo] Semantic Preprocessor (Standalone)...")
    df_raw = pd.read_csv(TRAIN_DATA_PATH)
    # Take a small subset for speed
    df_subset = df_raw.head(50).copy()

    preprocessor = SemanticPreprocessor()
    preprocessor.fit(df_subset)
    transformed_dict = preprocessor.transform(df_subset)

    assert isinstance(transformed_dict, dict)
    assert "margin" in transformed_dict
    assert transformed_dict["margin"].shape == (50, 64)
    # Check standardization (mean approx 0, std approx 1)
    margin_mean = np.mean(transformed_dict["margin"])
    margin_std = np.std(transformed_dict["margin"])
    print(
        f"Subset Transformed Margin Stats - Mean: {margin_mean:.4f}, Std: {margin_std:.4f}"
    )

    # 4. Model Training Demonstration (Factorized OAS LDA)
    print("\n[Demo] Model Training...")
    model = FactorizedOASDiscriminant()

    # Fit the model using the training dictionary (contains separated feature groups)
    model.fit(data["train"]["X"], data["train"]["y"])

    # Verify that experts were created for each group
    assert len(model.experts) == len(
        FEATURE_PREFIXES
    ), "Model failed to create experts for all groups."
    print("Factorized OAS Discriminant fitted successfully.")

    # 5. Prediction and Metric Demonstration
    print("\n[Demo] Prediction and Evaluation...")

    # Predict on validation set
    val_probs = model.predict_proba(data["val"]["X"])

    # Verify Probabilities
    assert val_probs.shape == (
        len(data["val"]["y"]),
        len(classes),
    ), "Probability shape mismatch."
    # Check if rows sum to 1 (within floating point tolerance)
    row_sums = np.sum(val_probs, axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1."

    # Calculate Log Loss using utility
    val_loss = calculate_log_loss(data["val"]["y"], val_probs)
    print(f"Calculated Validation Log Loss: {val_loss:.4f}")

    # Ensure loss is valid
    assert val_loss > 0, "Log loss should be positive."
    assert val_loss < 10, "Log loss is suspiciously high for this task."

    # 6. Submission Generation Demonstration
    print("\n[Demo] Submission Generation...")

    # Predict on test set
    test_probs = model.predict_proba(data["test"]["X"])
    test_ids = data["test"]["ids"]

    # Define a temporary path for demo submission
    demo_submission_path = os.path.join(WORKING_DIR, "demo_submission.csv")

    save_submission(test_ids, test_probs, classes, output_path=demo_submission_path)

    # Verify file creation
    assert os.path.exists(demo_submission_path), "Submission file was not created."

    # Verify file content format
    df_sub = pd.read_csv(demo_submission_path)
    assert "id" in df_sub.columns, "Submission missing 'id' column."
    assert len(df_sub) == len(test_ids), "Submission row count mismatch."
    assert df_sub.shape[1] == len(classes) + 1, "Submission column count mismatch."
    print("Submission file verified.")

    # 7. Full Pipeline Execution
    # The library provides a convenience function to run the whole flow
    print("\n[Demo] Running Full Pipeline Function...")
    run_factorized_lda_pipeline()

    # Verify the final submission from the pipeline exists
    assert os.path.exists(
        SUBMISSION_FILE_PATH
    ), "Pipeline failed to generate final submission."
    print(f"Final submission found at: {SUBMISSION_FILE_PATH}")

    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    demo_implementation()
