import os
import sys
import numpy as np
import pandas as pd
import warnings

# Import library modules
import library.config
import library.data_loader
import library.feature_engineering
import library.model_builder
import library.training_utils


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    print("Initializing demonstration...")
    set_seed(42)

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # =========================================================================
    # 1. Demonstrate Data Loading
    # =========================================================================
    print("\n[Demo] Loading Datasets...")

    # Load Train Data
    # We disable cache loading initially to demonstrate the loading logic
    df_train = library.data_loader.load_dataset_with_metadata(
        "train", load_cached_data=False
    )
    print(f"Train Data Loaded: {df_train.shape}")

    # Validate Train Data
    assert (
        "requester_received_pizza" in df_train.columns
    ), "Target column missing in train"
    assert (
        "request_text" in df_train.columns
        or "request_text_edit_aware" in df_train.columns
    ), "Text column missing"
    assert df_train.shape[0] > 0, "Train dataframe is empty"

    # Load Test Data
    df_test = library.data_loader.load_dataset_with_metadata(
        "test", load_cached_data=False
    )
    print(f"Test Data Loaded: {df_test.shape}")

    # Validate Test Data
    # Test data should not have the target label column (or it might be present but ignored,
    # but strictly speaking for inference we focus on features)
    assert df_test.shape[0] > 0, "Test dataframe is empty"

    # =========================================================================
    # 2. Demonstrate Feature Engineering
    # =========================================================================
    print("\n[Demo] Generating Features (includes Transformer encoding)...")

    # Generate features for the training set.
    # This triggers the SentenceTransformer model and QuantileTransformer.
    # We use load_cached_data=False to ensure the pipeline runs for this demo.
    X_train, y_train = library.feature_engineering.get_features(
        "train", load_cached_data=False
    )

    print(f"Features Generated. Shape: {X_train.shape}")

    # Validate Features
    # Expected dimensions: 768 (MPNet embeddings) + 10 (Numerical features) = 778
    expected_dim = 768 + 10
    assert (
        X_train.shape[1] == expected_dim
    ), f"Expected {expected_dim} features, got {X_train.shape[1]}"
    assert X_train.shape[0] == df_train.shape[0], "Feature row count mismatch"
    assert not np.isnan(X_train).any(), "Features contain NaNs"
    assert y_train is not None, "Labels are None for training set"

    # =========================================================================
    # 3. Configure for Fast Execution (Monkey Patching)
    # =========================================================================
    print("\n[Demo] Patching configuration for fast demonstration...")

    # Modify the C_GRID in training_utils to reduce search space
    # Original: [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    library.training_utils.C_GRID = [0.1, 1.0]

    # Modify the BAGGING_PARAMS in model_builder to reduce ensemble size
    # Original: n_estimators=100
    library.model_builder.BAGGING_PARAMS["n_estimators"] = 5
    library.model_builder.BAGGING_PARAMS["max_samples"] = (
        0.5  # Use smaller subsamples for speed
    )

    print(f"Modified C_GRID: {library.training_utils.C_GRID}")
    print(f"Modified BAGGING_PARAMS: {library.model_builder.BAGGING_PARAMS}")

    # =========================================================================
    # 4. Demonstrate Model Building (Unit Test)
    # =========================================================================
    print("\n[Demo] Building Model Instance...")
    model = library.model_builder.create_bagged_linear_model(C=1.0)

    # Verify model structure
    from sklearn.ensemble import BaggingClassifier
    from sklearn.linear_model import LogisticRegression

    assert isinstance(model, BaggingClassifier), "Model is not a BaggingClassifier"
    assert isinstance(
        model.estimator, LogisticRegression
    ), "Base estimator is not LogisticRegression"
    assert model.n_estimators == 5, "Patching n_estimators failed"

    # =========================================================================
    # 5. Demonstrate Stratified Cross-Validation
    # =========================================================================
    print("\n[Demo] Running Stratified CV...")

    # Run CV with reduced splits (n_splits=2) for speed
    # load_cached_data=True allows it to pick up the X_train we generated in step 2
    best_C, best_auc = library.training_utils.run_stratified_cv(
        n_splits=2, load_cached_data=True
    )

    print(f"CV Complete. Best C: {best_C}, Best AUC: {best_auc:.4f}")
    assert best_C in library.training_utils.C_GRID, "Best C not in grid"
    assert 0 <= best_auc <= 1, "AUC score out of range"

    # =========================================================================
    # 6. Demonstrate Final Training and Submission
    # =========================================================================
    print("\n[Demo] Training Final Model and Generating Submission...")

    # This step will:
    # 1. Retrain the model on the full training set using best_C
    # 2. Generate features for the test set (triggering transformer inference on test data)
    # 3. Predict and save the submission file
    submission_df = library.training_utils.train_and_predict_submission(
        best_C, load_cached_data=True
    )

    print("Submission Generated.")
    print(submission_df.head())

    # Validate Submission DataFrame
    assert "request_id" in submission_df.columns
    assert "requester_received_pizza" in submission_df.columns
    assert len(submission_df) == len(
        df_test
    ), f"Submission length mismatch: {len(submission_df)} vs {len(df_test)}"
    assert submission_df["requester_received_pizza"].min() >= 0
    assert submission_df["requester_received_pizza"].max() <= 1

    # Validate File Existence
    submission_path = os.path.join(library.config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file not found on disk"

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    main()
