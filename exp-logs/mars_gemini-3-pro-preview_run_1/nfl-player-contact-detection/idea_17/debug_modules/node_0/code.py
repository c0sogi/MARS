import os
import pandas as pd
import numpy as np
import shutil
import warnings
from library.config import Config
from library.feature_engineering import FeatureEngineer
from library.data_manager import DataManager
from library.mining_strategy import MiningStrategy
from library.model_factory import LGBMWrapper, XGBWrapper


# ------------------------------------------------------------------------------
# Setup & Configuration
# ------------------------------------------------------------------------------
def setup_demo_environment():
    """
    Sets up a temporary environment for the demo, overriding Config paths
    and hyperparameters to ensure speed and isolation.
    """
    # 1. Define Demo Directory
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # 2. Override Config Global Settings
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = demo_dir
    Config.SEED = 42

    # 3. Override Hyperparameters for Speed
    Config.N_ESTIMATORS = 10  # Reduced from 3000
    Config.EARLY_STOPPING_ROUNDS = 5  # Reduced from 100
    Config.VERBOSE_EVAL = -1  # Silent

    # 4. Override File Paths (Point to mini datasets we will create)
    Config.TRAIN_METADATA_PATH = os.path.join(demo_dir, "mini_train_metadata.csv")
    Config.VAL_METADATA_PATH = os.path.join(demo_dir, "mini_val_metadata.csv")
    Config.TEST_METADATA_PATH = os.path.join(demo_dir, "mini_test_metadata.csv")

    Config.TRAIN_TRACKING_PATH = os.path.join(demo_dir, "mini_train_tracking.csv")
    Config.TEST_TRACKING_PATH = os.path.join(demo_dir, "mini_test_tracking.csv")

    # 5. Override Cache Paths
    Config.CACHE_TRAIN_FEATURES = os.path.join(
        demo_dir, "data_cache/train_features_gated_full.parquet"
    )
    Config.CACHE_VAL_FEATURES = os.path.join(
        demo_dir, "data_cache/val_features_gated.parquet"
    )
    Config.CACHE_TEST_FEATURES = os.path.join(
        demo_dir, "data_cache/test_features_full.parquet"
    )
    Config.CACHE_HARD_NEGATIVE_INDICES = os.path.join(
        demo_dir, "data_cache/hard_negative_indices.npy"
    )

    # 6. Override Model Paths
    model_dir = os.path.join(demo_dir, "models")
    Config.MODEL_SCOUT_LGBM_PATH = os.path.join(model_dir, "scout_lgbm.joblib")
    Config.MODEL_SCOUT_XGB_PATH = os.path.join(model_dir, "scout_xgb.joblib")
    Config.MODEL_EXPERT_LGBM_PATH = os.path.join(model_dir, "expert_lgbm.joblib")
    Config.MODEL_EXPERT_XGB_PATH = os.path.join(model_dir, "expert_xgb.joblib")
    Config.BEST_THRESHOLD_PATH = os.path.join(model_dir, "best_threshold.npy")

    return demo_dir


def create_mini_datasets():
    """
    Creates small subsets of the original data to demonstrate the pipeline quickly.
    """
    print("Creating mini datasets for demonstration...")

    # Load original metadata
    orig_train_meta = pd.read_csv("./metadata/train_metadata.csv")
    orig_val_meta = pd.read_csv("./metadata/val_metadata.csv")
    orig_test_meta = pd.read_csv("./metadata/test_metadata.csv")

    # Sample Game Plays (2 train, 1 val, 1 test)
    # We select plays that are guaranteed to exist in tracking data
    train_plays = orig_train_meta["game_play"].unique()[:2]
    val_plays = orig_val_meta["game_play"].unique()[:1]
    test_plays = orig_test_meta["game_play"].unique()[:1]

    # Filter Metadata
    mini_train_meta = orig_train_meta[
        orig_train_meta["game_play"].isin(train_plays)
    ].copy()
    mini_val_meta = orig_val_meta[orig_val_meta["game_play"].isin(val_plays)].copy()
    mini_test_meta = orig_test_meta[orig_test_meta["game_play"].isin(test_plays)].copy()

    # Save Metadata
    mini_train_meta.to_csv(Config.TRAIN_METADATA_PATH, index=False)
    mini_val_meta.to_csv(Config.VAL_METADATA_PATH, index=False)
    mini_test_meta.to_csv(Config.TEST_METADATA_PATH, index=False)

    # Load and Filter Tracking Data
    # Note: Train and Val usually share the 'train_player_tracking.csv' source
    print("Loading and filtering tracking data (this may take a moment)...")
    orig_train_tracking = pd.read_csv("./input/train_player_tracking.csv")
    orig_test_tracking = pd.read_csv("./input/test_player_tracking.csv")

    required_train_plays = np.concatenate([train_plays, val_plays])
    mini_train_tracking = orig_train_tracking[
        orig_train_tracking["game_play"].isin(required_train_plays)
    ].copy()
    mini_test_tracking = orig_test_tracking[
        orig_test_tracking["game_play"].isin(test_plays)
    ].copy()

    # Save Tracking
    mini_train_tracking.to_csv(Config.TRAIN_TRACKING_PATH, index=False)
    mini_test_tracking.to_csv(Config.TEST_TRACKING_PATH, index=False)

    print(f"Mini Train Metadata: {len(mini_train_meta)} rows")
    print(f"Mini Train Tracking: {len(mini_train_tracking)} rows")


# ------------------------------------------------------------------------------
# Main Execution Pipeline
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

    # 1. Setup
    demo_dir = setup_demo_environment()
    create_mini_datasets()

    # 2. Feature Engineering
    print("\n=== Step 2: Feature Engineering ===")
    fe = FeatureEngineer()

    # Generate Train Features (with Gating)
    df_train = fe.create_train_features(load_cached_data=False)
    assert not df_train.empty, "Train features dataframe is empty!"
    assert "distance_0" in df_train.columns, "Distance feature missing!"
    assert "ttc" in df_train.columns, "TTC feature missing!"
    print(f"Generated Train Features: {df_train.shape}")

    # Generate Val Features (with Gating)
    df_val = fe.create_val_features(load_cached_data=False)
    assert not df_val.empty, "Val features dataframe is empty!"
    print(f"Generated Val Features: {df_val.shape}")

    # Generate Test Features (No Gating - Inference Mode)
    df_test = fe.create_test_features(load_cached_data=False)
    assert not df_test.empty, "Test features dataframe is empty!"
    print(f"Generated Test Features: {df_test.shape}")

    # 3. Scout Training (Phase 1)
    print("\n=== Step 3: Scout Training (Balanced) ===")
    ms = MiningStrategy()

    # Train Scouts
    scout_lgbm, scout_xgb = ms.train_scouts(df_train, load_cached_data=False)

    # Verify Models
    assert scout_lgbm.model is not None, "Scout LGBM model is None"
    assert scout_xgb.model is not None, "Scout XGB model is None"
    assert os.path.exists(Config.MODEL_SCOUT_LGBM_PATH), "Scout LGBM file not saved"

    # 4. Hard Negative Mining (Phase 2)
    print("\n=== Step 4: Diversity Mining (Hard Negatives) ===")
    hard_indices = ms.mine_hard_negatives(
        df_train, scout_lgbm, scout_xgb, load_cached_data=False
    )

    # Verify Indices
    assert isinstance(hard_indices, np.ndarray), "Hard indices should be a numpy array"
    if len(hard_indices) > 0:
        assert hard_indices.max() < len(df_train), "Hard indices out of bounds"

    # 5. Expert Training (Phase 3)
    print("\n=== Step 5: Expert Training (Imbalanced/Weighted) ===")
    dm = DataManager()

    # Construct Expert Dataset
    X_exp, y_exp = dm.get_expert_dataset(df_train, hard_indices)
    assert len(X_exp) > 0, "Expert dataset is empty"

    # Train Expert LGBM
    expert_lgbm = LGBMWrapper(mode="expert")
    expert_lgbm.fit(X_exp, y_exp)  # No validation set for demo speed
    expert_lgbm.save(Config.MODEL_EXPERT_LGBM_PATH)

    # Train Expert XGB
    expert_xgb = XGBWrapper(mode="expert")
    expert_xgb.fit(X_exp, y_exp)
    expert_xgb.save(Config.MODEL_EXPERT_XGB_PATH)

    # 6. Inference & Submission
    print("\n=== Step 6: Inference & Submission ===")

    # Prepare Test Data
    feature_cols = [c for c in df_test.columns if c not in dm.metadata_cols]
    X_test = df_test[feature_cols]

    # Predict (Ensemble Average)
    preds_lgbm = expert_lgbm.predict(X_test)
    preds_xgb = expert_xgb.predict(X_test)
    preds_ensemble = (preds_lgbm + preds_xgb) / 2.0

    # Verify Predictions
    assert len(preds_ensemble) == len(df_test), "Prediction length mismatch"
    assert (preds_ensemble >= 0).all() and (
        preds_ensemble <= 1
    ).all(), "Predictions out of probability range"

    # Create Submission DataFrame
    submission = pd.DataFrame(
        {
            "contact_id": df_test["contact_id"],
            "contact": (preds_ensemble > 0.5).astype(int),
        }
    )

    # Save Submission
    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")
    print(submission.head())

    print("\nPipeline execution completed successfully.")
