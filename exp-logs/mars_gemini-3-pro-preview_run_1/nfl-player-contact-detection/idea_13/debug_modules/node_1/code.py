import os
import pandas as pd
import numpy as np
import shutil
import warnings
import library.config as config
from library.utils import seed_everything, compute_mcc
from library.data_loader import get_data
from library.feature_engine import generate_features
from library.mining_curriculum import MiningCurriculum
from library.model_factory import UnifiedEnsemble

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def create_mini_dataset(
    source_meta_path, source_track_path, dest_meta_path, dest_track_path, n_plays=2
):
    """
    Creates a mini dataset by selecting a few plays and their corresponding tracking data.
    """
    print(f"Creating mini-dataset from {source_meta_path}...")

    # Load full metadata
    df_meta = pd.read_csv(source_meta_path)

    # Sample specific plays to ensure data consistency
    unique_plays = df_meta["game_play"].unique()
    if len(unique_plays) > n_plays:
        selected_plays = unique_plays[:n_plays]
    else:
        selected_plays = unique_plays

    df_mini_meta = df_meta[df_meta["game_play"].isin(selected_plays)].copy()

    # Load full tracking (only columns needed)
    # We read in chunks or just read full if memory allows (it does for this env)
    df_track = pd.read_csv(source_track_path)
    df_mini_track = df_track[df_track["game_play"].isin(selected_plays)].copy()

    # Save mini files
    df_mini_meta.to_csv(dest_meta_path, index=False)
    df_mini_track.to_csv(dest_track_path, index=False)

    print(f"  Saved {len(df_mini_meta)} metadata rows to {dest_meta_path}")
    print(f"  Saved {len(df_mini_track)} tracking rows to {dest_track_path}")


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration and Setup
    # -------------------------------------------------------------------------
    print("--- 1. Configuration and Setup ---")
    seed_everything(config.SEED)

    # Define demo directory
    DEMO_DIR = os.path.join(config.WORKING_DIR, "demo_execution")
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Monkey-patch config paths to point to our mini-datasets
    config.WORKING_DIR = DEMO_DIR
    config.TRAIN_METADATA_PATH = os.path.join(DEMO_DIR, "mini_train_metadata.csv")
    config.VAL_METADATA_PATH = os.path.join(DEMO_DIR, "mini_val_metadata.csv")
    config.TEST_METADATA_PATH = os.path.join(DEMO_DIR, "mini_test_metadata.csv")

    # Note: Train and Val usually share the same tracking file in the original setup
    config.TRAIN_TRACKING_PATH = os.path.join(DEMO_DIR, "mini_train_tracking.csv")
    config.TEST_TRACKING_PATH = os.path.join(DEMO_DIR, "mini_test_tracking.csv")

    # Monkey-patch Model Hyperparameters for Speed
    # Reduce estimators significantly for demo
    config.LGBM_PARAMS["n_estimators"] = 10
    config.LGBM_PARAMS["num_leaves"] = 31
    config.XGB_PARAMS["n_estimators"] = 10
    config.XGB_PARAMS["max_depth"] = 4

    # -------------------------------------------------------------------------
    # 2. Prepare Mini Datasets
    # -------------------------------------------------------------------------
    print("\n--- 2. Preparing Mini Datasets ---")

    # Create Mini Train (2 plays)
    create_mini_dataset(
        source_meta_path="./metadata/train_metadata.csv",
        source_track_path="./input/train_player_tracking.csv",
        dest_meta_path=config.TRAIN_METADATA_PATH,
        dest_track_path=config.TRAIN_TRACKING_PATH,
        n_plays=3,
    )

    # Create Mini Val (1 play)
    create_mini_dataset(
        source_meta_path="./metadata/val_metadata.csv",
        source_track_path="./input/train_player_tracking.csv",
        dest_meta_path=config.VAL_METADATA_PATH,
        dest_track_path=config.TRAIN_TRACKING_PATH,  # Re-using train tracking file for val split logic
        n_plays=1,
    )

    # Create Mini Test (1 play)
    # Note: Test metadata comes from test_metadata.csv, tracking from test_player_tracking.csv
    create_mini_dataset(
        source_meta_path="./metadata/test_metadata.csv",
        source_track_path="./input/test_player_tracking.csv",
        dest_meta_path=config.TEST_METADATA_PATH,
        dest_track_path=config.TEST_TRACKING_PATH,
        n_plays=1,
    )

    # -------------------------------------------------------------------------
    # 3. Feature Engineering
    # -------------------------------------------------------------------------
    print("\n--- 3. Feature Engineering Pipeline ---")

    # Process Train
    print("Processing Train Split...")
    df_train_raw = get_data("train", load_cached_data=False, apply_gating=True)
    df_train_feats = generate_features(df_train_raw, "train", load_cached_data=False)

    # Process Val
    print("Processing Val Split...")
    df_val_raw = get_data("val", load_cached_data=False, apply_gating=True)
    df_val_feats = generate_features(df_val_raw, "val", load_cached_data=False)

    # Process Test (No gating for test usually, but for consistency/speed in demo we can)
    # Typically test set should not be gated if we want to predict for all rows in sample_submission
    print("Processing Test Split...")
    df_test_raw = get_data("test", load_cached_data=False, apply_gating=False)
    df_test_feats = generate_features(df_test_raw, "test", load_cached_data=False)

    # Verification
    print(f"Train Features Shape: {df_train_feats.shape}")
    print(f"Val Features Shape: {df_val_feats.shape}")

    # Identify Feature Columns (exclude metadata)
    exclude_cols = [
        "contact_id",
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "datetime",
        "contact",
        "video_path_endzone",
        "video_path_sideline",
        "video_path_all29",
        "p2_join_id",
        "p2_join",
    ]
    # Filter for numeric columns only to avoid passing strings/objects to models
    # Cite debug_lesson_13: Define Feature Schemas in a Single Source of Truth
    # Cite debug_lesson_11: Restrict Imputation to Numeric Columns (applied here to feature selection)
    numeric_cols = df_train_feats.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    print(f"Selected {len(feature_cols)} features for training.")

    # Assert features exist
    assert "speed_p1" in feature_cols, "Basic tracking feature missing"
    assert "iks_dist_to_p1_min" in feature_cols, "IKS feature missing"

    # -------------------------------------------------------------------------
    # 4. Mining Curriculum (Scout -> Expert)
    # -------------------------------------------------------------------------
    print("\n--- 4. Mining Curriculum ---")

    miner = MiningCurriculum(feature_cols=feature_cols, target_col="contact")

    # Step A: Run Scout Mining to find Hard Negatives
    # Using a high threshold for demo to ensure we find some 'hard' negatives easily or just run logic
    # In a real scenario, threshold is tuned.
    hard_neg_indices = miner.run_scout_mining(
        df_train=df_train_feats,
        df_val=df_val_feats,
        hard_neg_threshold=0.01,  # Low threshold to ensure we flag things as hard negatives in this small sample
        load_cached_mining=False,
    )

    assert isinstance(hard_neg_indices, np.ndarray)

    # Step B: Construct Expert Dataset
    df_expert = miner.prepare_expert_dataset(
        df_train=df_train_feats, hard_neg_indices=hard_neg_indices, buffer_ratio=0.5
    )

    assert len(df_expert) > 0, "Expert dataset is empty"
    assert "contact" in df_expert.columns

    # -------------------------------------------------------------------------
    # 5. Model Training (Unified Ensemble)
    # -------------------------------------------------------------------------
    print("\n--- 5. Training Unified Ensemble ---")

    model = UnifiedEnsemble()
    model.fit(
        df_train=df_expert,
        df_val=df_val_feats,
        feature_cols=feature_cols,
        target_col="contact",
    )

    # Validate on Validation Set
    preds_val = model.predict(df_val_feats, feature_cols)

    # Compute MCC
    # Threshold optimization is usually done here, we'll use 0.5 for demo
    y_val = df_val_feats["contact"].values
    y_pred_bin = (preds_val > 0.5).astype(int)
    mcc = compute_mcc(y_val, y_pred_bin)

    print(f"Validation MCC (Threshold 0.5): {mcc:.4f}")

    # -------------------------------------------------------------------------
    # 6. Inference and Submission
    # -------------------------------------------------------------------------
    print("\n--- 6. Inference on Test Set ---")

    # Predict
    preds_test = model.predict(df_test_feats, feature_cols)

    # Create Submission DataFrame
    df_sub = df_test_feats[["contact_id"]].copy()
    df_sub["contact"] = (preds_test > 0.5).astype(int)

    # Save
    submission_path = os.path.join(DEMO_DIR, "submission.csv")
    df_sub.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    print("Head of submission:")
    print(df_sub.head())

    # Final Assertions
    assert df_sub.shape[0] == df_test_feats.shape[0], "Submission row count mismatch"
    assert df_sub["contact"].isin([0, 1]).all(), "Submission contains non-binary values"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
