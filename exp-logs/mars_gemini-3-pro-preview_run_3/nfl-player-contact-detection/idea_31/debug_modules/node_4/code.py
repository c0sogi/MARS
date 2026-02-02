import os
import sys
import shutil
import pandas as pd
import numpy as np

# Import classes from the provided library files
from library.config import Config
from library.feature_engineering import FeatureEngineer
from library.data_pipeline import DataPipeline
from library.modeling import DualStreamModel
from library.inference import InferenceEngine


def run_demo():
    print(">>> [Setup] Configuring environment for rapid demonstration...")

    # 1. Override Configuration for Speed and Efficiency
    # Limit training data size
    Config.DEBUG_SAMPLE_SIZE = 2000

    # Simplify Feature Engineering (reduce temporal window complexity)
    Config.EXP_LAGS = [1]

    # Reduce Model Complexity for Demo
    Config.STREAM_A_PARAMS["n_estimators"] = 5
    Config.STREAM_B_PARAMS["n_estimators"] = 5
    Config.EARLY_STOPPING_ROUNDS = 2
    Config.VERBOSE_EVAL = False

    # Set a specific working directory for this demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_PATH = "./working/demo_run/submission/submission.csv"

    # Clean up previous demo runs if they exist
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    # Re-initialize environment (creates directories based on new paths)
    Config.setup()

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Debug Sample Size: {Config.DEBUG_SAMPLE_SIZE}")

    # =========================================================================
    # 2. Test Feature Engineering
    # =========================================================================
    print("\n>>> [FeatureEngineer] Testing Stream A (Interaction) processing...")
    fe = FeatureEngineer()

    # Process Stream A (Player-Player) for Training
    # We force load_cached_data=False to ensure logic runs
    X_a, y_a, ids_a = fe.process_stream_a(mode="train", load_cached_data=False)

    print(f"    Output Shapes -> X: {X_a.shape}, y: {y_a.shape}, ids: {ids_a.shape}")

    # Assertions
    if len(X_a) == 0:
        raise RuntimeError(
            "Feature Engineering produced empty dataframe. Check input data or sampling."
        )

    assert isinstance(X_a, pd.DataFrame), "X must be a pandas DataFrame"
    assert isinstance(y_a, np.ndarray), "y must be a numpy array"
    assert (
        len(X_a) == len(y_a) == len(ids_a)
    ), "Mismatch in sample counts between X, y, and ids"

    # Verify specific features exist
    expected_features = ["distance", "sideline_iou", "closure_rate"]
    for feat in expected_features:
        assert (
            feat in X_a.columns
        ), f"Expected feature '{feat}' missing from Stream A output"

    # Verify Lag features generation
    lag_cols = [c for c in X_a.columns if "_lag1" in c]
    assert len(lag_cols) > 0, "Temporal lag features were not generated"

    print("    FeatureEngineer validation passed.")

    # =========================================================================
    # 3. Test Data Pipeline (Undersampling)
    # =========================================================================
    print("\n>>> [DataPipeline] Testing Data Loading and Undersampling...")
    dp = DataPipeline()

    # Load data via pipeline (Stream A)
    # This should pick up the cached data from step 2, but apply undersampling
    X_pipe, y_pipe, ids_pipe = dp.load_data(mode="train", stream="streamA")

    n_pos = np.sum(y_pipe == 1)
    n_neg = np.sum(y_pipe == 0)

    print(f"    Pipeline Output -> Positives: {n_pos}, Negatives: {n_neg}")

    # Assertions for Undersampling
    if n_pos > 0:
        # Check if ratio is roughly respected (allow small margin for rounding)
        ratio = n_neg / n_pos
        # Note: If original data didn't have enough negatives, ratio might be lower,
        # but it should never be significantly higher than Config.NEG_POS_RATIO
        assert (
            ratio <= Config.NEG_POS_RATIO + 1.0
        ), f"Undersampling failed: Ratio {ratio:.2f} exceeds limit"

    print("    DataPipeline validation passed.")

    # =========================================================================
    # 4. Test Model Training
    # =========================================================================
    print("\n>>> [DualStreamModel] Testing Training Loop...")
    model = DualStreamModel()

    # Train both streams
    # This will internally process Stream B (Player-Ground) as well
    model.train()

    # Verify Artifacts
    model_dir = os.path.join(Config.WORKING_DIR, "models")
    model_a_path = os.path.join(model_dir, "xgb_stream_a.json")
    model_b_path = os.path.join(model_dir, "xgb_stream_b.json")
    thresh_path = os.path.join(model_dir, "thresholds.joblib")

    assert os.path.exists(model_a_path), "Stream A model file not found"
    assert os.path.exists(model_b_path), "Stream B model file not found"
    assert os.path.exists(thresh_path), "Thresholds file not found"

    print("    Models and thresholds saved successfully.")
    print("    DualStreamModel training validation passed.")

    # =========================================================================
    # 5. Test Inference
    # =========================================================================
    print("\n>>> [InferenceEngine] Testing Prediction on Test Set...")
    inf = InferenceEngine()

    # Run prediction
    # use_validation=True triggers threshold optimization using the validation set
    inf.predict(use_validation=True)

    # Verify Submission
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    print(f"    Submission Shape: {df_sub.shape}")
    print(f"    Columns: {list(df_sub.columns)}")

    # Assertions
    assert "contact_id" in df_sub.columns, "Missing 'contact_id' column"
    assert "contact" in df_sub.columns, "Missing 'contact' column"
    assert df_sub["contact"].isin([0, 1]).all(), "Predictions contain non-binary values"

    # Check against sample submission length (should match exactly)
    df_sample = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
    assert len(df_sub) == len(
        df_sample
    ), f"Submission length mismatch. Expected {len(df_sample)}, got {len(df_sub)}"

    print("    InferenceEngine validation passed.")
    print("\n>>> All demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
