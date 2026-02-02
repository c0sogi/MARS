import os
import pandas as pd
import numpy as np
import warnings
import shutil

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Import provided library modules
import library.config as config
import library.data_loader as data_loader
import library.features as features
import library.models as models
import library.trainer as trainer
import library.inference as inference


def create_mini_dataset(output_dir):
    """Creates a small subset of the data for demonstration purposes."""
    print(f"Creating mini dataset in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load and sample Metadata
    # We select one game_play to ensure consistency between metadata and tracking
    full_train_meta = pd.read_csv(config.TRAIN_METADATA_PATH)
    sample_play_id = full_train_meta["game_play"].unique()[0]

    # Take all interactions for this single play
    mini_train_meta = full_train_meta[
        full_train_meta["game_play"] == sample_play_id
    ].copy()

    # Ensure we have at least one positive sample for training stability
    if mini_train_meta["contact"].sum() == 0:
        mini_train_meta.iloc[0, mini_train_meta.columns.get_loc("contact")] = 1

    mini_train_path = os.path.join(output_dir, "mini_train_metadata.csv")
    mini_train_meta.to_csv(mini_train_path, index=False)

    # Use the same for validation to guarantee file existence and format
    mini_val_path = os.path.join(output_dir, "mini_val_metadata.csv")
    mini_train_meta.to_csv(mini_val_path, index=False)

    # Sample Test Metadata (from sample_submission format)
    full_test_meta = pd.read_csv(config.TEST_METADATA_PATH)
    mini_test_meta = full_test_meta.head(50).copy()
    mini_test_path = os.path.join(output_dir, "mini_test_metadata.csv")
    mini_test_meta.to_csv(mini_test_path, index=False)

    # 2. Load and sample Tracking Data
    full_train_tracking = pd.read_csv(config.TRAIN_TRACKING_PATH)
    mini_train_tracking = full_train_tracking[
        full_train_tracking["game_play"] == sample_play_id
    ].copy()
    mini_train_track_path = os.path.join(output_dir, "mini_train_tracking.csv")
    mini_train_tracking.to_csv(mini_train_track_path, index=False)

    # For test tracking, finding exact matches for the head(50) might be empty if random.
    # We'll create a dummy test tracking file using the train tracking structure but mapped to test IDs if needed.
    # However, the simplest way for a runnable demo is to use the train tracking data and mock the test metadata
    # to match the game_play in the tracking data, OR just use the provided test_tracking.csv if it matches.
    # Let's try to find matches in provided test tracking.
    full_test_tracking = pd.read_csv(config.TEST_TRACKING_PATH)
    test_plays = mini_test_meta["game_play"].unique()
    mini_test_tracking = full_test_tracking[
        full_test_tracking["game_play"].isin(test_plays)
    ].copy()

    # Fallback if no matches found in head(50)
    if mini_test_tracking.empty:
        # Use a chunk of test tracking
        mini_test_tracking = full_test_tracking.head(1000).copy()
        # Force metadata to match this tracking chunk
        valid_play = mini_test_tracking["game_play"].iloc[0]
        mini_test_meta["game_play"] = valid_play
        # Fix contact_id to match new game_play
        mini_test_meta["contact_id"] = mini_test_meta.apply(
            lambda x: f"{x['game_play']}_{x['step']}_{x['nfl_player_id_1']}_{x['nfl_player_id_2']}",
            axis=1,
        )
        mini_test_meta.to_csv(mini_test_path, index=False)

    mini_test_track_path = os.path.join(output_dir, "mini_test_tracking.csv")
    mini_test_tracking.to_csv(mini_test_track_path, index=False)

    return {
        "train_meta": mini_train_path,
        "val_meta": mini_val_path,
        "test_meta": mini_test_path,
        "train_track": mini_train_track_path,
        "test_track": mini_test_track_path,
    }


def patch_modules(paths, working_dir):
    """Patches library modules to use mini datasets and fast hyperparameters."""
    print("Patching library modules for fast execution...")

    cache_dir = os.path.join(working_dir, "cache")
    models_dir = os.path.join(working_dir, "models")
    submission_path = os.path.join(working_dir, "submission.csv")

    # 1. Patch Config Module
    config.WORKING_DIR = working_dir
    config.CACHE_DIR = cache_dir
    config.SUBMISSION_PATH = submission_path
    config.HARD_NEGATIVE_INDICES_PATH = os.path.join(
        cache_dir, "hard_negative_indices.npy"
    )
    config.BEST_THRESHOLD_PATH = os.path.join(models_dir, "best_threshold.npy")

    # 2. Patch Data Loader Module (Directly, as it imports variables)
    data_loader.TRAIN_METADATA_PATH = paths["train_meta"]
    data_loader.VAL_METADATA_PATH = paths["val_meta"]
    data_loader.TEST_METADATA_PATH = paths["test_meta"]
    data_loader.TRAIN_TRACKING_PATH = paths["train_track"]
    data_loader.TEST_TRACKING_PATH = paths["test_track"]

    # 3. Patch Features Module
    features.CACHE_DIR = cache_dir
    features.TRAIN_METADATA_PATH = paths["train_meta"]
    features.VAL_METADATA_PATH = paths["val_meta"]
    features.TEST_METADATA_PATH = paths["test_meta"]
    features.TRAIN_TRACKING_PATH = paths["train_track"]
    features.TEST_TRACKING_PATH = paths["test_track"]

    # 4. Patch Trainer Module
    trainer.WORKING_DIR = working_dir
    trainer.SUBMISSION_PATH = submission_path
    trainer.HARD_NEGATIVE_INDICES_PATH = config.HARD_NEGATIVE_INDICES_PATH
    trainer.BEST_THRESHOLD_PATH = config.BEST_THRESHOLD_PATH

    # Define fast parameters
    fast_lgbm = {
        "n_estimators": 5,
        "num_leaves": 8,
        "max_depth": 3,
        "early_stopping_rounds": 2,
        "verbosity": -1,
        "n_jobs": 2,
        "random_state": 42,
    }
    fast_xgb = {
        "n_estimators": 5,
        "max_depth": 3,
        "early_stopping_rounds": 2,
        "verbosity": 0,
        "n_jobs": 2,
        "random_state": 42,
        "device": "cpu",  # Use CPU for tiny data to avoid overhead
    }

    # Update param dictionaries in trainer
    trainer.SCOUT_LGBM_PARAMS.update(fast_lgbm)
    trainer.LGBM_PARAMS.update(fast_lgbm)
    trainer.SCOUT_XGB_PARAMS.update(fast_xgb)
    trainer.XGB_PARAMS.update(fast_xgb)

    # 5. Patch Inference Module
    inference.WORKING_DIR = working_dir
    inference.SUBMISSION_PATH = submission_path
    inference.BEST_THRESHOLD_PATH = config.BEST_THRESHOLD_PATH


if __name__ == "__main__":
    # Setup working directories
    DEMO_DIR = "./working/demo_execution"
    DATA_DIR = os.path.join(DEMO_DIR, "data")

    # 1. Create Mini Dataset
    paths = create_mini_dataset(DATA_DIR)

    # 2. Patch Modules
    patch_modules(paths, DEMO_DIR)

    # 3. Demonstrate Feature Engineering
    print("\n=== DEMO: Feature Engineering ===")
    # Instantiate FeatureEngineer
    fe = features.FeatureEngineer(cache_dir=config.CACHE_DIR)

    # Generate Train Features (Force reload to ensure we use mini data)
    df_features = fe.generate_features(split="train", load_cached_data=False)

    print(f"Generated features shape: {df_features.shape}")

    # Validation
    assert not df_features.empty, "Feature dataframe is empty!"
    assert (
        "distance_lag0" in df_features.columns
    ), "Base feature 'distance_lag0' missing."
    assert (
        "distance_lag1" in df_features.columns
    ), "Lagged feature 'distance_lag1' missing."
    assert "closing_speed_lag0" in df_features.columns, "Geometric feature missing."
    print("Feature engineering logic verified.")

    # 4. Demonstrate Full Training Pipeline
    print("\n=== DEMO: Training Pipeline ===")
    # Instantiate Trainer
    t = trainer.Trainer()
    # Manually update models_dir since it's set in __init__ using the (now patched) WORKING_DIR
    t.models_dir = os.path.join(DEMO_DIR, "models")

    # Run Pipeline (Scouts -> Hard Negatives -> Experts -> Inference)
    # We use the cached features from step 3 to save time
    t.run(train_features_cache=True)

    # Validation
    models_path = os.path.join(DEMO_DIR, "models")
    assert os.path.exists(
        os.path.join(models_path, "scout_lgbm.joblib")
    ), "Scout LGBM not saved."
    assert os.path.exists(
        os.path.join(models_path, "expert_xgb.joblib")
    ), "Expert XGB not saved."
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not created."
    print("Training pipeline completed and artifacts verified.")

    # 5. Demonstrate Inference Pipeline (Standalone)
    print("\n=== DEMO: Inference Pipeline ===")
    # Instantiate InferencePipeline
    inf_pipeline = inference.InferencePipeline()
    inf_pipeline.models_dir = models_path  # Update path

    # Run Inference
    # This will load the models trained in Step 4 and regenerate test features
    inf_pipeline.run(load_cached_features=True)

    # Final Validation
    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    print(f"Final Submission Shape: {df_sub.shape}")
    print(df_sub.head(3))

    assert list(df_sub.columns) == [
        "contact_id",
        "contact",
    ], "Invalid submission columns."
    assert df_sub["contact"].isin([0, 1]).all(), "Predictions must be binary."

    print("\nAll demonstrations passed successfully.")
