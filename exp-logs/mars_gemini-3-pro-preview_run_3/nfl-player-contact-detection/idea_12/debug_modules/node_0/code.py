import os
import sys
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Import components from the provided library
from library.config import Config
from library.utils import seed_everything
from library.train import run_training_pipeline
from library.inference import generate_predictions
from library.feature_engineering import FeatureEngineer
from library.data_loader import load_metadata


def main():
    print(">>> Starting NFL Contact Detection Pipeline Demonstration...")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Modify Config to run extremely fast for demonstration purposes
    print(">>> Configuring environment for speed...")

    # Reduce XGBoost complexity
    Config.XGB_PARAMS["n_estimators"] = 10  # Very few trees for speed
    Config.XGB_PARAMS["early_stopping_rounds"] = 5
    Config.VERBOSE_EVAL = 0  # Silent training

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Run Training Pipeline
    # -------------------------------------------------------------------------
    print("\n>>> Executing Training Pipeline (Debug Mode)...")
    # debug_mode=True samples a small subset of games to process
    # load_cached_data=False forces the code to run feature engineering from scratch
    run_training_pipeline(debug_mode=True, load_cached_data=False)

    # Validate Training Artifacts
    model_a_path = os.path.join(Config.WORKING_DIR, "stream_a_interaction_model.json")
    model_b_path = os.path.join(Config.WORKING_DIR, "stream_b_impact_model.json")
    thresholds_path = os.path.join(Config.WORKING_DIR, "thresholds.json")

    if not os.path.exists(model_a_path):
        raise FileNotFoundError(f"Stream A model failed to save at {model_a_path}")
    if not os.path.exists(model_b_path):
        raise FileNotFoundError(f"Stream B model failed to save at {model_b_path}")
    if not os.path.exists(thresholds_path):
        raise FileNotFoundError(f"Thresholds file failed to save at {thresholds_path}")

    print(">>> Training artifacts verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Run Inference Pipeline
    # -------------------------------------------------------------------------
    print("\n>>> Executing Inference Pipeline (Debug Mode)...")
    # Generates predictions using the models trained above on a subset of test data
    generate_predictions(debug_mode=True, load_cached_data=False)

    # Validate Submission File
    submission_path = Config.SUBMISSION_PATH
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df_sub = pd.read_csv(submission_path)

    # Check schema
    required_cols = ["contact_id", "contact"]
    for col in required_cols:
        if col not in df_sub.columns:
            raise AssertionError(f"Submission missing required column: {col}")

    # Check content
    if df_sub.empty:
        raise AssertionError("Submission file is empty.")

    # Check values are binary
    unique_vals = df_sub["contact"].unique()
    if not all(v in [0, 1] for v in unique_vals):
        raise AssertionError(f"Submission contains non-binary values: {unique_vals}")

    print(f">>> Submission verified. Shape: {df_sub.shape}")

    # -------------------------------------------------------------------------
    # 4. Component Logic Verification (Feature Engineering)
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Feature Engineering Logic...")

    # Instantiate FeatureEngineer
    fe = FeatureEngineer()

    # Load a tiny slice of metadata directly
    df_meta_sample = load_metadata("train").head(20)

    # Test Stream A (Interaction) Feature Generation
    # This verifies that tracking data is merged and visual features are computed
    X_a, y_a, ids_a = fe.create_features(
        metadata_df=df_meta_sample,
        stream_config=Config.STREAM_A,
        dataset_type="train",
        load_cached_data=False,
    )

    if not X_a.empty:
        # Verify dimensions match
        assert len(X_a) == len(y_a), "Stream A: Feature/Label length mismatch"
        assert len(X_a) == len(ids_a), "Stream A: Feature/ID length mismatch"

        # Verify specific features exist
        # 'p1_speed_lag_0' comes from tracking
        assert "p1_speed_lag_0" in X_a.columns, "Stream A: Missing tracking feature"
        # 'p1_p2_dist' comes from interaction logic
        assert "p1_p2_dist" in X_a.columns, "Stream A: Missing interaction feature"
        # Visual features (if enabled)
        if Config.STREAM_A["use_visuals"]:
            visual_present = any("iou" in col for col in X_a.columns)
            assert visual_present, "Stream A: Missing visual (IoU) feature"

        print(">>> Stream A features validated.")
    else:
        print(">>> Stream A validation skipped (no player-player contacts in sample).")

    # Test Stream B (Impact) Feature Generation
    # This verifies kinematic computations like Jerk
    X_b, y_b, ids_b = fe.create_features(
        metadata_df=df_meta_sample,
        stream_config=Config.STREAM_B,
        dataset_type="train",
        load_cached_data=False,
    )

    if not X_b.empty:
        # Verify dimensions
        assert len(X_b) == len(y_b), "Stream B: Feature/Label length mismatch"

        # Verify Stream B specific features
        # 'jerk' is computed in _compute_kinematics
        jerk_present = any("jerk" in col for col in X_b.columns)
        assert jerk_present, "Stream B: Missing 'jerk' feature"

        # Verify Visuals are NOT present (as per config)
        visual_present = any("iou" in col for col in X_b.columns)
        assert not visual_present, "Stream B: Visual features present despite config"

        print(">>> Stream B features validated.")
    else:
        print(">>> Stream B validation skipped (no player-ground contacts in sample).")

    print("\n>>> Demonstration completed successfully!")


if __name__ == "__main__":
    main()
