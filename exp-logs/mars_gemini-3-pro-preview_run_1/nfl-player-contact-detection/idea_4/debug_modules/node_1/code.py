import os
import pandas as pd
import numpy as np
import warnings
import shutil
from library.config import Config
from library.feature_engineering import FeatureProcessor
from library.models import LGBMWrapper, XGBWrapper
from library.train import train_ensemble
from library.inference import predict_and_submit

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_mini_environment():
    """
    Sets up a temporary working directory and creates mini datasets
    to allow the pipeline to run quickly for demonstration purposes.
    """
    print("Setting up mini-environment and datasets...")

    # 1. Override Config for speed and isolation
    Config.WORKING_DIR = "./working/demo_run"
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.XGB_PARAMS["n_estimators"] = 10
    Config.WINDOW_HALF_SIZE = 2  # Reduce window size to speed up feature creation
    Config.VERBOSE_EVAL = -1  # Silence training output

    # Ensure working dir exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Create Mini Training Data (Sample 2 plays)
    # We read the original metadata and tracking, sample them, and save to working dir
    full_train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    sample_plays = full_train_meta["game_play"].unique()[:2]  # Take first 2 plays

    mini_train_meta = full_train_meta[
        full_train_meta["game_play"].isin(sample_plays)
    ].copy()

    # Create a mini validation set from the same sample (just for code path verification)
    mini_val_meta = mini_train_meta.iloc[:50].copy()

    # Load tracking for these plays
    full_train_track = pd.read_csv(Config.TRAIN_TRACKING_PATH)
    mini_train_track = full_train_track[
        full_train_track["game_play"].isin(sample_plays)
    ].copy()

    # Save mini files
    path_train_meta = os.path.join(Config.WORKING_DIR, "mini_train_metadata.csv")
    path_val_meta = os.path.join(Config.WORKING_DIR, "mini_val_metadata.csv")
    path_train_track = os.path.join(Config.WORKING_DIR, "mini_train_tracking.csv")

    mini_train_meta.to_csv(path_train_meta, index=False)
    mini_val_meta.to_csv(path_val_meta, index=False)
    mini_train_track.to_csv(path_train_track, index=False)

    # 3. Create Mini Test Data
    full_test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
    mini_test_meta = full_test_meta.head(100).copy()  # Take first 100 interaction rows

    full_test_track = pd.read_csv(Config.TEST_TRACKING_PATH)
    test_plays = mini_test_meta["game_play"].unique()
    mini_test_track = full_test_track[
        full_test_track["game_play"].isin(test_plays)
    ].copy()

    path_test_meta = os.path.join(Config.WORKING_DIR, "mini_test_metadata.csv")
    path_test_track = os.path.join(Config.WORKING_DIR, "mini_test_tracking.csv")

    mini_test_meta.to_csv(path_test_meta, index=False)
    mini_test_track.to_csv(path_test_track, index=False)

    # 4. Patch Config paths to point to mini files
    Config.TRAIN_METADATA_PATH = path_train_meta
    Config.VAL_METADATA_PATH = path_val_meta
    Config.TRAIN_TRACKING_PATH = path_train_track
    Config.TEST_METADATA_PATH = path_test_meta
    Config.TEST_TRACKING_PATH = path_test_track

    print("Mini-datasets created and Config updated.")


def demo_feature_engineering():
    print("\n=== Demo: Feature Engineering ===")
    processor = FeatureProcessor()

    # Process 'train' split. load_cached_data=False forces regeneration.
    df_features = processor.process_split("train", load_cached_data=False)

    # Verifications
    print(f"Generated feature matrix shape: {df_features.shape}")

    # Check for critical columns
    expected_cols = ["distance", "speed_diff", "contact", "game_play"]
    for col in expected_cols:
        assert col in df_features.columns, f"Missing expected column: {col}"

    # Check for windowed features (lag/lead)
    # Since WINDOW_HALF_SIZE=2, we expect lag1, lag2, lead1, lead2
    assert (
        "distance_lag1" in df_features.columns
    ), "Windowed feature 'distance_lag1' missing."
    assert (
        "speed_p1_lead2" in df_features.columns
    ), "Windowed feature 'speed_p1_lead2' missing."

    print("Feature Engineering logic verified.")
    return df_features


def demo_model_training(df_train):
    print("\n=== Demo: Model Training ===")

    # Prepare Features
    exclude_cols = [
        "contact_id",
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "contact",
        "datetime",
        "p1_id",
        "p2_id",
    ]
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]

    X = df_train[feature_cols]
    y = df_train["contact"]

    # 1. LightGBM
    print("Training LightGBM...")
    lgbm = LGBMWrapper()
    lgbm.fit(X, y, X, y)  # Train and validate on same data for demo speed

    # Predict
    probs = lgbm.predict_proba(X)
    assert len(probs) == len(X), "LGBM prediction length mismatch."
    assert (
        probs.min() >= 0 and probs.max() <= 1
    ), "LGBM probabilities out of [0,1] range."

    # Save/Load
    lgbm.save("demo_lgbm.joblib")
    assert os.path.exists(
        os.path.join(Config.WORKING_DIR, "demo_lgbm.joblib")
    ), "LGBM model file not found."

    # 2. XGBoost
    print("Training XGBoost...")
    xgb_model = XGBWrapper()
    xgb_model.fit(X, y, X, y)

    probs_xgb = xgb_model.predict_proba(X)
    assert len(probs_xgb) == len(X), "XGB prediction length mismatch."

    xgb_model.save("demo_xgb.joblib")
    assert os.path.exists(
        os.path.join(Config.WORKING_DIR, "demo_xgb.joblib")
    ), "XGB model file not found."

    print("Model training and persistence verified.")


def demo_full_pipeline():
    print("\n=== Demo: Full Pipeline (Train & Inference) ===")

    # 1. Train Ensemble
    # This function inside library.train handles loading data, processing features,
    # training both models, and saving them to the working directory.
    # It returns the optimized threshold.
    print("Running train_ensemble()...")
    best_threshold = train_ensemble(load_cached_data=False)

    print(f"Optimal Threshold found: {best_threshold}")
    assert 0 < best_threshold < 1, "Threshold is invalid."

    # 2. Inference
    # This function inside library.inference loads test data, processes it,
    # loads the saved models, predicts, and saves submission.csv.
    print("Running predict_and_submit()...")
    submission = predict_and_submit(threshold=best_threshold, load_cached_data=False)

    # Verification
    print("Verifying submission file...")
    assert not submission.empty, "Submission DataFrame is empty."
    assert "contact_id" in submission.columns, "contact_id column missing."
    assert "contact" in submission.columns, "contact column missing."

    # Check file on disk
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Check content
    saved_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(saved_df) == len(submission), "Saved submission length mismatch."
    assert (
        saved_df["contact"].isin([0, 1]).all()
    ), "Predictions contain non-binary values."

    print("Full pipeline execution verified successfully.")


if __name__ == "__main__":
    # Ensure reproducible results
    np.random.seed(Config.SEED)

    # 1. Setup
    setup_mini_environment()

    # 2. Feature Engineering Demo
    df_features = demo_feature_engineering()

    # 3. Model Training Demo (Unit test style)
    demo_model_training(df_features)

    # 4. Full Integration Test
    demo_full_pipeline()

    print("\nAll demonstrations completed successfully.")
