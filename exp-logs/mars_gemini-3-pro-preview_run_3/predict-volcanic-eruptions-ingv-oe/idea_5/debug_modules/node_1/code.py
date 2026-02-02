import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
import shutil

# Import library components
import library.config as config
import library.data_loader as data_loader
import library.feature_engineering as fe
import library.feature_selection as fs
import library.model as model_lib

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


def run_demo():
    print("=== Starting Library Usage Demo ===\n")

    # ==========================================
    # 1. Data Loading Demo
    # ==========================================
    print("--- 1. Data Loader Demo ---")

    # Load a small subset of metadata for training and validation
    # We use a small sample size to keep the demo fast
    DEMO_SIZE = 20
    train_meta = data_loader.load_metadata("train", sample_size=DEMO_SIZE)
    val_meta = data_loader.load_metadata("val", sample_size=DEMO_SIZE)

    print(f"Loaded {len(train_meta)} training metadata rows.")
    print(f"Loaded {len(val_meta)} validation metadata rows.")

    # Validate Metadata Structure
    assert "segment_id" in train_meta.columns
    assert "time_to_eruption" in train_meta.columns
    assert "file_path" in train_meta.columns

    # Demonstrate loading a single raw sensor segment
    sample_file_path = train_meta.iloc[0]["file_path"]
    print(f"Loading sensor segment: {sample_file_path}")
    sensor_df = data_loader.load_sensor_segment(sample_file_path, fill_na=True)

    # Validate Sensor Data
    assert sensor_df.shape == (60001, 10), f"Unexpected shape: {sensor_df.shape}"
    assert not sensor_df.isnull().values.any(), "Loaded sensor data contains NaNs"
    print("Sensor segment loaded successfully. Shape: (60001, 10)\n")

    # ==========================================
    # 2. Feature Engineering Demo
    # ==========================================
    print("--- 2. Feature Engineering Demo ---")

    # Initialize Extractor
    # We use a temporary cache dir for the demo to avoid messing with real experiments
    demo_cache_dir = os.path.join(config.WORKING_DIR, "demo_cache")
    os.makedirs(demo_cache_dir, exist_ok=True)

    extractor = fe.FeatureExtractor(cache_dir=demo_cache_dir)

    # Generate features for the training subset
    # load_cached_data=False forces the calculation to run
    print("Generating features for training subset...")
    X_train_full = extractor.generate_features(
        train_meta, "train_demo", load_cached_data=False
    )

    # Generate features for the validation subset
    print("Generating features for validation subset...")
    X_val_full = extractor.generate_features(
        val_meta, "val_demo", load_cached_data=False
    )

    # Validate Feature Output
    assert len(X_train_full) == DEMO_SIZE
    assert "segment_id" in X_train_full.columns
    # Check for a few expected feature columns
    expected_cols = ["sensor_1_mean", "band_low_mean", "mfcc_0_mean"]
    for col in expected_cols:
        assert col in X_train_full.columns, f"Missing expected feature: {col}"

    print(f"Feature generation complete. Feature count: {X_train_full.shape[1]}\n")

    # ==========================================
    # 3. Feature Selection Demo
    # ==========================================
    print("--- 3. Feature Selection Demo ---")

    y_train = train_meta["time_to_eruption"]

    # Monkeypatch RFE config for speed in this demo
    # We reduce the number of estimators and features to select just for the test
    original_rfe_estimators = config.RFE_PARAMS["estimator_params"]["n_estimators"]
    config.RFE_PARAMS["estimator_params"]["n_estimators"] = 10
    config.RFE_PARAMS["n_features_to_select"] = 10

    # Run Feature Selection
    # subset_size=1.0 uses all 20 samples provided in X_train_full
    selected_features = fs.select_features(
        X_train_full.drop(columns=["segment_id"]),
        y_train,
        load_cached_data=False,
        subset_size=1.0,
    )

    # Restore config (good practice)
    config.RFE_PARAMS["estimator_params"]["n_estimators"] = original_rfe_estimators

    print(f"Selected {len(selected_features)} features.")
    assert len(selected_features) > 0
    assert set(selected_features).issubset(X_train_full.columns)

    # Apply selection
    X_train_selected = X_train_full[selected_features]
    X_val_selected = X_val_full[selected_features]
    print("Feature selection applied.\n")

    # ==========================================
    # 4. Model Training Demo
    # ==========================================
    print("--- 4. Model Training Demo ---")

    y_val = val_meta["time_to_eruption"]

    # Define lightweight params for instant training
    demo_lgbm_params = {
        "objective": "regression_l1",
        "metric": "mae",
        "n_estimators": 20,  # Very few trees
        "learning_rate": 0.1,
        "num_leaves": 10,
        "min_child_samples": 2,  # Low because we only have 20 samples
        "verbosity": -1,
        "n_jobs": 1,
        "random_state": 42,
    }

    print("Training LightGBM model...")
    model = model_lib.train_model(
        X_train_selected, y_train, X_val_selected, y_val, params=demo_lgbm_params
    )

    assert model is not None
    print("Model training complete.\n")

    # ==========================================
    # 5. Prediction & Submission Demo
    # ==========================================
    print("--- 5. Prediction & Submission Demo ---")

    # Load Test Metadata
    test_meta = data_loader.load_metadata("test", sample_size=5)
    print(f"Loaded {len(test_meta)} test metadata rows.")

    # Generate Test Features
    print("Generating features for test subset...")
    X_test_full = extractor.generate_features(
        test_meta, "test_demo", load_cached_data=False
    )

    # Apply Feature Selection (must include segment_id for submission)
    X_test_selected = X_test_full[selected_features].copy()
    X_test_selected["segment_id"] = X_test_full["segment_id"]

    # Generate Submission
    # We'll save to a demo path to avoid overwriting real submissions
    original_sub_path = model_lib.SUBMISSION_PATH
    demo_sub_path = os.path.join(config.SUBMISSION_DIR, "demo_submission.csv")
    model_lib.SUBMISSION_PATH = demo_sub_path

    print("Generating submission file...")
    model_lib.generate_submission(model, X_test_selected)

    # Verify Submission
    assert os.path.exists(demo_sub_path), "Submission file was not created"
    sub_df = pd.read_csv(demo_sub_path)

    assert list(sub_df.columns) == ["segment_id", "time_to_eruption"]
    assert len(sub_df) == 5
    assert (
        sub_df["time_to_eruption"].dtype == float
        or sub_df["time_to_eruption"].dtype == int
    )

    print(f"Submission verified at {demo_sub_path}")
    print("\n=== Demo Completed Successfully ===")

    # Cleanup (Optional)
    if os.path.exists(demo_cache_dir):
        shutil.rmtree(demo_cache_dir)
    # Restore original path
    model_lib.SUBMISSION_PATH = original_sub_path


if __name__ == "__main__":
    run_demo()
