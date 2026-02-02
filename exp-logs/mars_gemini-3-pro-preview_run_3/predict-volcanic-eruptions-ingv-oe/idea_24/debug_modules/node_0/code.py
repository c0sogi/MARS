import os
import sys
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from library import config
from library import feature_engineering
from library import data_manager
from library import model_handler
from library import workflow_orchestrator

# =============================================================================
# CONFIGURATION & UTILS
# =============================================================================
DEBUG_SIZE = 20  # Small sample size for fast execution
RANDOM_SEED = 42

# Set seeds for reproducibility
np.random.seed(RANDOM_SEED)


def print_header(msg):
    print(f"\n{'='*60}\n{msg}\n{'='*60}")


def mock_get_lgbm_params(overrides=None):
    """
    Replacement function for config.get_lgbm_params.
    Returns lightweight parameters for fast demonstration.
    """
    params = {
        "objective": "regression_l2",
        "metric": "mae",
        "boosting_type": "gbdt",
        "n_estimators": 10,  # Reduced from 5000 for speed
        "learning_rate": 0.05,
        "num_leaves": 8,  # Reduced complexity
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbosity": -1,  # Silent
        "n_jobs": 1,
        "seed": RANDOM_SEED,
        "early_stopping_rounds": 5,
    }
    if overrides:
        params.update(overrides)
    return params


# Apply Monkey Patch to model_handler to ensure it uses fast params
# This affects both train_ensemble and train_fold_model calls within that module
model_handler.get_lgbm_params = mock_get_lgbm_params
print("Monkey-patched LightGBM parameters for fast execution.")


# =============================================================================
# 1. FEATURE ENGINEERING VERIFICATION
# =============================================================================
def test_feature_engineering():
    print_header("Testing Feature Engineering")

    # Load raw metadata to get a valid file path
    train_meta = pd.read_csv(config.TRAIN_META_PATH)
    sample_row = train_meta.iloc[0].to_dict()

    print(f"Processing single segment: {sample_row['segment_id']}")

    # Test single segment processing
    features = feature_engineering.process_segment(sample_row)

    # Assertions
    assert features is not None, "process_segment returned None"
    assert isinstance(features, dict), "process_segment should return a dictionary"
    assert "segment_id" in features, "segment_id missing from features"
    assert features["segment_id"] == sample_row["segment_id"], "segment_id mismatch"
    assert "time_to_eruption" in features, "Target variable missing"

    # Check for specific feature groups (e.g., sensor_1_mean, sensor_1_trend_vel_std)
    keys = list(features.keys())
    sensor_1_feats = [k for k in keys if k.startswith("sensor_1_")]
    assert len(sensor_1_feats) > 0, "No features generated for sensor_1"

    print(f"Successfully extracted {len(features)} features for one segment.")
    print("Sample features:", list(features.keys())[:5])


# =============================================================================
# 2. DATA MANAGER VERIFICATION
# =============================================================================
def test_data_manager():
    print_header("Testing Data Manager")

    # Test Loading Train Data
    print(f"Loading training data (debug_size={DEBUG_SIZE})...")
    X_train, y_train = data_manager.get_train_data(
        load_cached_data=False, debug_size=DEBUG_SIZE
    )

    assert isinstance(X_train, pd.DataFrame), "X_train should be a DataFrame"
    assert isinstance(y_train, pd.Series), "y_train should be a Series"
    assert (
        len(X_train) == DEBUG_SIZE
    ), f"Expected {DEBUG_SIZE} training samples, got {len(X_train)}"
    assert (
        len(y_train) == DEBUG_SIZE
    ), f"Expected {DEBUG_SIZE} target values, got {len(y_train)}"
    assert "segment_id" not in X_train.columns, "segment_id should be dropped from X"

    print(f"Train Data Shape: {X_train.shape}")

    # Test Loading Test Data
    print(f"Loading test data (debug_size={DEBUG_SIZE})...")
    X_test, segment_ids = data_manager.get_test_data(
        load_cached_data=False, debug_size=DEBUG_SIZE
    )

    assert len(X_test) == DEBUG_SIZE, f"Expected {DEBUG_SIZE} test samples"
    assert len(segment_ids) == DEBUG_SIZE, "Segment IDs length mismatch"
    assert (
        X_test.shape[1] == X_train.shape[1]
    ), "Feature count mismatch between train and test"

    print(f"Test Data Shape: {X_test.shape}")


# =============================================================================
# 3. MODEL TRAINING VERIFICATION (UNIT)
# =============================================================================
def test_model_training():
    print_header("Testing Model Training (Unit)")

    # Get data
    X, y = data_manager.get_train_data(load_cached_data=False, debug_size=DEBUG_SIZE)

    # Split manually for unit test
    split_idx = int(DEBUG_SIZE * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    params = mock_get_lgbm_params()

    print("Training single fold model...")
    model = model_handler.train_fold_model(X_train, y_train, X_val, y_val, params)

    assert isinstance(
        model, lgb.LGBMRegressor
    ), "Model should be an LGBMRegressor instance"

    # Verify prediction works
    preds = model.predict(X_val)
    assert len(preds) == len(X_val), "Prediction length mismatch"

    mae = np.mean(np.abs(y_val - preds))
    print(f"Unit Test Model MAE: {mae:.4f}")


# =============================================================================
# 4. WORKFLOW ORCHESTRATION VERIFICATION
# =============================================================================
def test_workflow():
    print_header("Testing Full Workflow (Cross-Validation + Submission)")

    # Clean up previous models in working dir to force retraining
    working_dir = config.WORKING_DIR
    if os.path.exists(working_dir):
        for f in os.listdir(working_dir):
            if f.endswith(".joblib"):
                os.remove(os.path.join(working_dir, f))

    # Run the submission generation
    # This will:
    # 1. Check for models (none found)
    # 2. Trigger run_cross_validation (train 5 folds)
    # 3. Load test data
    # 4. Predict and save to submission/submission.csv
    workflow_orchestrator.generate_submission(
        load_cached_data=False, debug_size=DEBUG_SIZE
    )

    # Verify Submission File
    sub_path = config.SUBMISSION_PATH
    assert os.path.exists(sub_path), f"Submission file not found at {sub_path}"

    df_sub = pd.read_csv(sub_path)
    print("Submission file loaded.")
    print(df_sub.head())

    assert list(df_sub.columns) == [
        "segment_id",
        "time_to_eruption",
    ], "Invalid submission columns"
    assert len(df_sub) == DEBUG_SIZE, f"Submission should have {DEBUG_SIZE} rows"
    assert not df_sub.isnull().values.any(), "Submission contains NaNs"

    print("Workflow completed successfully.")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    try:
        test_feature_engineering()
        test_data_manager()
        test_model_training()
        test_workflow()

        print_header("ALL TESTS PASSED")

    except AssertionError as e:
        print(f"\n[FAILED] Assertion Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        # Print stack trace for debugging
        import traceback

        traceback.print_exc()
        sys.exit(1)
