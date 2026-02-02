import os
import shutil
import warnings
import numpy as np
import pandas as pd

# Import from the provided library
from library.config import Config
import library.feature_engineering as fe
import library.data_loader as dl
from library.model import VolcanoLGBM
import library.trainer as tr


def main():
    print("Starting Volcano Eruption Prediction Demo...")

    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Override
    # -------------------------------------------------------------------------
    # Set seeds for reproducibility
    np.random.seed(42)
    warnings.filterwarnings("ignore")

    # Override Config for a fast demonstration run
    # We create a separate directory for this demo to avoid overwriting real work
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    print(f"Configuring environment. Working directory: {demo_dir}")

    # Update paths in Config
    Config.WORKING_DIR = demo_dir
    Config.TRAIN_FEATURES_PATH = os.path.join(demo_dir, "train_features.parquet")
    Config.VAL_FEATURES_PATH = os.path.join(demo_dir, "val_features.parquet")
    Config.TEST_FEATURES_PATH = os.path.join(demo_dir, "test_features.parquet")
    Config.MODEL_OUTPUT_PATH = os.path.join(demo_dir, "lgbm_model.txt")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission_demo.csv")

    # Reduce Model Complexity for Speed
    Config.MODEL_PARAMS["n_estimators"] = 10  # Very few iterations
    Config.MODEL_PARAMS["num_leaves"] = 8  # Simple trees

    # Reduce CV Complexity
    Config.TRAIN_PARAMS["n_folds"] = 2  # Minimum folds
    Config.TRAIN_PARAMS["early_stopping_rounds"] = 5

    # -------------------------------------------------------------------------
    # 2. Test Feature Engineering
    # -------------------------------------------------------------------------
    print("\n[Test] Feature Engineering...")
    # Generate features for a tiny subset (e.g., 5 samples)
    debug_limit_fe = 5

    # We explicitly force regeneration (load_cached_data=False)
    train_feats_df = fe.generate_features(
        Config.TRAIN_METADATA_PATH,
        Config.TRAIN_FEATURES_PATH,
        load_cached_data=False,
        debug_limit=debug_limit_fe,
    )

    # Validation
    assert isinstance(train_feats_df, pd.DataFrame)
    assert (
        len(train_feats_df) == debug_limit_fe
    ), f"Expected {debug_limit_fe} rows, got {len(train_feats_df)}"
    assert (
        "time_to_eruption" in train_feats_df.columns
    ), "Target column missing from features"
    assert "segment_id" in train_feats_df.columns, "Segment ID missing from features"
    # Check if we have a reasonable number of features (sensors * features_per_sensor)
    # 10 sensors, ~43 features each (Kinematic + Spectral) -> >400 columns
    assert (
        train_feats_df.shape[1] > 400
    ), f"Too few features generated: {train_feats_df.shape[1]}"
    print("Feature Engineering verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Test Data Loader
    # -------------------------------------------------------------------------
    print("\n[Test] Data Loader...")
    # Load the dataset we just created (load_cached_data=True)
    X_train, y_train = dl.create_dataset(
        "train", load_cached_data=True, debug_limit=debug_limit_fe
    )

    # Validation
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert X_train.shape[0] == debug_limit_fe
    assert len(y_train) == debug_limit_fe
    assert not np.isnan(X_train).any(), "Feature matrix contains NaNs"
    print("Data Loader verified successfully.")

    # -------------------------------------------------------------------------
    # 4. Test Model Wrapper
    # -------------------------------------------------------------------------
    print("\n[Test] Model Training...")
    model = VolcanoLGBM()

    # Split the tiny dataset manually for this unit test
    # Ensure we have at least 1 sample for train and 1 for val
    split_idx = max(1, debug_limit_fe - 1)
    X_t, y_t = X_train[:split_idx], y_train[:split_idx]
    X_v, y_v = X_train[split_idx:], y_train[split_idx:]

    # Train
    model.train(X_t, y_t, X_v, y_v)

    # Predict
    preds = model.predict(X_v)

    # Validation
    assert len(preds) == len(y_v)
    assert np.isfinite(preds).all(), "Predictions contain non-finite values"
    assert os.path.exists(Config.MODEL_OUTPUT_PATH), "Model file was not saved"
    print("Model training and prediction verified successfully.")

    # -------------------------------------------------------------------------
    # 5. Test Full Pipeline (Trainer)
    # -------------------------------------------------------------------------
    print("\n[Test] Full Pipeline (Train & Predict)...")
    # We use a slightly larger limit to ensure StratifiedKFold has enough samples per bin
    # 30 samples from Train + 30 samples from Val = 60 samples total
    # With 2 folds, we have 30 samples per fold.
    debug_limit_pipeline = 30

    # This function orchestrates: Loading -> CV Training -> Test Prediction -> Submission
    mae = tr.train_and_predict(load_cached_data=False, debug_limit=debug_limit_pipeline)

    # Validation
    print(f"Pipeline finished with MAE: {mae:.4f}")
    assert isinstance(mae, float)
    assert mae >= 0, "MAE cannot be negative"

    # Check Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Check Submission Format
    assert list(submission_df.columns) == ["segment_id", "time_to_eruption"]
    assert (
        len(submission_df) == debug_limit_pipeline
    ), f"Submission should have {debug_limit_pipeline} rows (limited by debug_limit), got {len(submission_df)}"

    print("Full Pipeline verified successfully.")
    print(f"Demo completed. Artifacts stored in {demo_dir}")


if __name__ == "__main__":
    main()
