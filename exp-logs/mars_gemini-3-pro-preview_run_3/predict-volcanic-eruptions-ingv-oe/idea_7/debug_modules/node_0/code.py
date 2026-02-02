import os
import sys
import numpy as np
import pandas as pd
import warnings

# Set random seeds for reproducibility
np.random.seed(42)

# Import library functions
from library.feature_extraction import extract_segment_features
from library.dataset import get_train_data, get_test_data
from library.models import get_lightgbm_regressor, get_xgboost_regressor
from library.trainer import run_stratified_cv


def demo_feature_extraction():
    """
    Demonstrates how to extract features from a single data segment.
    """
    print("\n=== Demo: Feature Extraction ===")

    # Load training metadata to find a valid segment
    train_meta_path = "./metadata/train.csv"
    if not os.path.exists(train_meta_path):
        raise FileNotFoundError(f"Metadata not found at {train_meta_path}")

    meta_df = pd.read_csv(train_meta_path)

    # Pick the first segment
    sample_row = meta_df.iloc[0]
    segment_id = sample_row["segment_id"]
    file_path = sample_row["file_path"]

    print(f"Extracting features for Segment ID: {segment_id}")

    # Extract features
    features = extract_segment_features(segment_id, file_path)

    # Validation
    assert features is not None, "Feature extraction returned None."
    assert isinstance(features, dict), "Features should be returned as a dictionary."
    assert (
        features["segment_id"] == segment_id
    ), "Segment ID mismatch in extracted features."

    # Check for expected feature groups (Kinematic, Spectral, Temporal)
    keys = features.keys()
    has_kinematic = any("vel_mean" in k for k in keys)
    has_spectral = any("spec_centroid" in k for k in keys)
    has_temporal = any("win0_mean" in k for k in keys)

    assert has_kinematic, "Kinematic features missing."
    assert has_spectral, "Spectral features missing."
    assert has_temporal, "Temporal features missing."

    print(f"Successfully extracted {len(features)} features from one segment.")


def demo_dataset_loading():
    """
    Demonstrates how to load training and test datasets using the library.
    Uses a small debug_size for speed.
    """
    print("\n=== Demo: Dataset Loading ===")

    debug_size = 20
    print(f"Loading debug datasets (size={debug_size})...")

    # 1. Load Training Data
    # load_cached_data=False ensures we test the generation logic
    X_train, y_train = get_train_data(debug_size=debug_size, load_cached_data=False)

    # Validation
    assert (
        len(X_train) == debug_size
    ), f"Expected {debug_size} training samples, got {len(X_train)}."
    assert (
        len(y_train) == debug_size
    ), f"Expected {debug_size} target values, got {len(y_train)}."
    assert not X_train.isnull().values.any(), "Training features contain NaN values."

    # 2. Load Test Data
    X_test, test_ids = get_test_data(debug_size=debug_size, load_cached_data=False)

    # Validation
    assert (
        len(X_test) == debug_size
    ), f"Expected {debug_size} test samples, got {len(X_test)}."
    assert (
        len(test_ids) == debug_size
    ), f"Expected {debug_size} test IDs, got {len(test_ids)}."
    assert (
        X_train.shape[1] == X_test.shape[1]
    ), "Feature count mismatch between Train and Test."

    print("Dataset loading verified successfully.")
    return X_train, y_train


def demo_models(X, y):
    """
    Demonstrates initialization and training of LightGBM and XGBoost models.
    """
    print("\n=== Demo: Model Training ===")

    # Split data for a quick validation check
    split_idx = len(X) // 2
    X_train_sub, X_val_sub = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_sub, y_val_sub = y[:split_idx], y[split_idx:]

    # 1. LightGBM
    print("Training LightGBM...")
    lgb_model = get_lightgbm_regressor(n_estimators=10, learning_rate=0.1)
    lgb_model.fit(
        X_train_sub,
        y_train_sub,
        eval_set=[(X_val_sub, y_val_sub)],
        eval_metric="mae",
        callbacks=[
            import_lightgbm().early_stopping(stopping_rounds=5, verbose=False),
            import_lightgbm().log_evaluation(period=0),
        ],
    )
    preds_lgb = lgb_model.predict(X_val_sub)
    assert len(preds_lgb) == len(y_val_sub), "LightGBM prediction shape mismatch."

    # 2. XGBoost
    print("Training XGBoost...")
    # Using CPU for this small demo to avoid overhead, though A100 is available.
    xgb_model = get_xgboost_regressor(n_estimators=10, learning_rate=0.1, device="cpu")
    xgb_model.fit(
        X_train_sub, y_train_sub, eval_set=[(X_val_sub, y_val_sub)], verbose=False
    )
    preds_xgb = xgb_model.predict(X_val_sub)
    assert len(preds_xgb) == len(y_val_sub), "XGBoost prediction shape mismatch."

    print("Model training and inference verified.")


def demo_full_pipeline():
    """
    Demonstrates the full Stratified CV pipeline using the trainer module.
    """
    print("\n=== Demo: Full Stratified CV Pipeline ===")

    # Run pipeline with minimal parameters for speed
    # n_splits=2 is the minimum for CV
    # debug_size=30 ensures enough data for splits
    mae = run_stratified_cv(
        n_splits=2,
        debug_size=30,
        load_cached_data=True,  # Leverage cache if available
        generate_submission=True,
        n_estimators=5,
        learning_rate=0.1,
    )

    print(f"Pipeline finished with CV MAE: {mae:.4f}")

    # Verify Submission File
    submission_path = "./submission/submission.csv"
    assert os.path.exists(submission_path), "Submission file was not generated."

    sub_df = pd.read_csv(submission_path)
    assert "segment_id" in sub_df.columns, "Submission missing segment_id."
    assert "time_to_eruption" in sub_df.columns, "Submission missing target column."
    assert len(sub_df) > 0, "Submission file is empty."

    print(f"Submission file verified at {submission_path}")


def import_lightgbm():
    """Helper to import lightgbm locally to access callback classes."""
    import lightgbm

    return lightgbm


if __name__ == "__main__":
    # Suppress specific warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    try:
        # 1. Feature Extraction
        demo_feature_extraction()

        # 2. Dataset Loading
        X_debug, y_debug = demo_dataset_loading()

        # 3. Model Training
        demo_models(X_debug, y_debug)

        # 4. Full Pipeline
        demo_full_pipeline()

        print("\nAll demonstrations completed successfully.")

    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        raise e
