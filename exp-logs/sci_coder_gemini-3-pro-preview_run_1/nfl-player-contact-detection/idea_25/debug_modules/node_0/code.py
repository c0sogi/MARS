import os
import sys
import numpy as np
import pandas as pd
import shutil
import warnings
import joblib

# Suppress warnings for clean output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Import library components
from library.config import Config
from library.data_loader import load_metadata, load_tracking
from library.features import FeatureEngineer
from library.trainer import Trainer
from library.inference import InferenceManager
from library.models import TriEnsemble


def setup_demo_environment():
    """
    Sets up a lightweight environment for demonstration:
    1. Creates a working directory for demo outputs.
    2. Creates mini versions of metadata and tracking files to speed up processing.
    3. Monkey-patches the Config class to use these mini files and reduced model parameters.
    """
    print("Setting up demo environment...")

    # Define demo paths
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Monkey-patch Config to use this directory
    Config.WORKING_DIR = demo_dir
    Config.CACHE_MODELS = os.path.join(demo_dir, "models")
    Config.SUBMISSION_OUTPUT_PATH = os.path.join(demo_dir, "submission.csv")

    # Update cache paths
    Config.CACHE_TRAIN_FEATURES = os.path.join(demo_dir, "features_train.parquet")
    Config.CACHE_VAL_FEATURES = os.path.join(demo_dir, "features_val.parquet")
    Config.CACHE_TEST_FEATURES = os.path.join(demo_dir, "features_test.parquet")
    Config.CACHE_HARD_NEGATIVES = os.path.join(demo_dir, "hard_negative_indices.npy")

    os.makedirs(Config.CACHE_MODELS, exist_ok=True)

    # --- Create Mini Datasets ---
    # We take a small sample of plays to ensure the pipeline runs quickly.

    # 1. Train Metadata
    print("Creating mini train metadata...")
    full_train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    # Select 2 unique plays for training
    train_plays = full_train_meta["game_play"].unique()[:2]
    mini_train_meta = full_train_meta[
        full_train_meta["game_play"].isin(train_plays)
    ].copy()

    mini_train_meta_path = os.path.join(demo_dir, "mini_train_metadata.csv")
    mini_train_meta.to_csv(mini_train_meta_path, index=False)
    Config.TRAIN_METADATA_PATH = mini_train_meta_path

    # 2. Val Metadata
    print("Creating mini val metadata...")
    full_val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    # Select 1 unique play for validation
    val_plays = full_val_meta["game_play"].unique()[:1]
    mini_val_meta = full_val_meta[full_val_meta["game_play"].isin(val_plays)].copy()

    mini_val_meta_path = os.path.join(demo_dir, "mini_val_metadata.csv")
    mini_val_meta.to_csv(mini_val_meta_path, index=False)
    Config.VAL_METADATA_PATH = mini_val_meta_path

    # 3. Test Metadata
    print("Creating mini test metadata...")
    full_test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
    # Select 1 unique play for testing
    test_plays = full_test_meta["game_play"].unique()[:1]
    mini_test_meta = full_test_meta[full_test_meta["game_play"].isin(test_plays)].copy()

    mini_test_meta_path = os.path.join(demo_dir, "mini_test_metadata.csv")
    mini_test_meta.to_csv(mini_test_meta_path, index=False)
    Config.TEST_METADATA_PATH = mini_test_meta_path

    # 4. Tracking Data
    # We need to filter tracking data to only include the plays we selected
    print("Creating mini tracking data...")
    selected_plays = set(train_plays) | set(val_plays)

    # Load full tracking (this might take a few seconds)
    full_tracking = pd.read_csv(Config.TRAIN_TRACKING_PATH)
    mini_tracking = full_tracking[
        full_tracking["game_play"].isin(selected_plays)
    ].copy()

    mini_tracking_path = os.path.join(demo_dir, "mini_train_tracking.csv")
    mini_tracking.to_csv(mini_tracking_path, index=False)
    Config.TRAIN_TRACKING_PATH = mini_tracking_path

    # Test tracking
    full_test_tracking = pd.read_csv(Config.TEST_TRACKING_PATH)
    mini_test_tracking = full_test_tracking[
        full_test_tracking["game_play"].isin(test_plays)
    ].copy()

    mini_test_tracking_path = os.path.join(demo_dir, "mini_test_tracking.csv")
    mini_test_tracking.to_csv(mini_test_tracking_path, index=False)
    Config.TEST_TRACKING_PATH = mini_test_tracking_path

    # --- Reduce Model Complexity for Speed ---
    print("Adjusting hyperparameters for fast execution...")

    # LightGBM
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["num_leaves"] = 8

    # XGBoost
    Config.XGB_PARAMS["n_estimators"] = 10
    Config.XGB_PARAMS["max_depth"] = 3

    # CatBoost
    Config.CATBOOST_PARAMS["iterations"] = 10
    Config.CATBOOST_PARAMS["depth"] = 3

    # General
    Config.EARLY_STOPPING_ROUNDS = 5
    Config.NUM_BOOST_ROUND = 10


def verify_feature_engineering():
    print("\n=== Verifying Feature Engineering ===")
    fe = FeatureEngineer()

    # Process Train
    df_train = fe.process_data("train", load_cached_data=False)
    print(f"Generated Train Features: {df_train.shape}")

    # Assertions
    assert not df_train.empty, "Train dataframe is empty."
    assert (
        "quadratic_min_dist" in df_train.columns
    ), "Feature 'quadratic_min_dist' missing."
    assert "radial_velocity" in df_train.columns, "Feature 'radial_velocity' missing."
    assert "contact" in df_train.columns, "Target 'contact' missing."

    # Check cache creation
    assert os.path.exists(
        Config.CACHE_TRAIN_FEATURES
    ), "Train features cache not created."

    # Process Val
    df_val = fe.process_data("val", load_cached_data=False)
    print(f"Generated Val Features: {df_val.shape}")
    assert not df_val.empty, "Val dataframe is empty."


def verify_training_pipeline():
    print("\n=== Verifying Training Pipeline (Trainer) ===")
    trainer = Trainer()

    # Run the full training pipeline
    # This includes: Scout training, Hard Negative Mining, Expert Training, Threshold Optimization, Submission Gen
    trainer.run()

    # Verify Model Artifacts
    models_dir = Config.CACHE_MODELS
    expected_files = [
        "scout_lgbm.joblib",
        "scout_xgb.joblib",
        "scout_cat.joblib",
        "expert_lgbm.joblib",
        "expert_xgb.joblib",
        "expert_cat.joblib",
        "best_threshold.npy",
    ]

    for f in expected_files:
        f_path = os.path.join(models_dir, f)
        assert os.path.exists(f_path), f"Model artifact {f} was not generated."

    # Verify Hard Negatives Cache
    assert os.path.exists(
        Config.CACHE_HARD_NEGATIVES
    ), "Hard negative indices cache not found."

    # Verify Submission
    assert os.path.exists(
        Config.SUBMISSION_OUTPUT_PATH
    ), "Submission file not generated by Trainer."

    df_sub = pd.read_csv(Config.SUBMISSION_OUTPUT_PATH)
    print(f"Trainer Submission Shape: {df_sub.shape}")
    assert "contact_id" in df_sub.columns
    assert "contact" in df_sub.columns


def verify_inference_manager():
    print("\n=== Verifying Inference Manager ===")

    # Instantiate InferenceManager
    # It should load the 'expert' models we just trained
    manager = InferenceManager()

    # 1. Optimize Threshold
    # We force it to ignore cache to verify the logic runs
    threshold = manager.optimize_threshold(load_cached_data=True)
    print(f"Optimized Threshold: {threshold}")
    assert 0.0 < threshold < 1.0, "Threshold optimization produced invalid value."

    # 2. Generate Predictions
    # This overwrites the submission file from the trainer, which is fine
    manager.generate_predictions(threshold=threshold, load_cached_data=False)

    # Verify Output
    assert os.path.exists(
        Config.SUBMISSION_OUTPUT_PATH
    ), "Submission file not generated by InferenceManager."
    df_sub = pd.read_csv(Config.SUBMISSION_OUTPUT_PATH)

    # Check against mini test metadata
    df_test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
    assert len(df_sub) == len(
        df_test_meta
    ), f"Submission row count ({len(df_sub)}) matches test metadata ({len(df_test_meta)})."

    print("Inference Manager verification successful.")


def verify_model_logic():
    print("\n=== Verifying TriEnsemble Logic ===")
    # Load models
    ensemble = TriEnsemble()
    ensemble.load_models(prefix="expert")

    # Create dummy data matching feature count
    n_features = len(Config.FEATURES)
    X_dummy = pd.DataFrame(np.random.rand(10, n_features), columns=Config.FEATURES)

    # Test predict_proba
    probs = ensemble.predict_proba(X_dummy)
    assert probs.shape == (10, 2), "predict_proba shape mismatch."
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities out of bounds."

    # Test predict
    preds = ensemble.predict(X_dummy, threshold=0.5)
    assert preds.shape == (10,), "predict shape mismatch."
    assert np.all(np.isin(preds, [0, 1])), "Predictions contain non-binary values."

    print("TriEnsemble logic verification successful.")


if __name__ == "__main__":
    # 1. Setup
    setup_demo_environment()

    # 2. Feature Engineering Verification
    verify_feature_engineering()

    # 3. Training Pipeline Verification
    verify_training_pipeline()

    # 4. Model Logic Verification
    verify_model_logic()

    # 5. Inference Manager Verification
    verify_inference_manager()

    print("\nAll demonstrations and verifications passed successfully.")
