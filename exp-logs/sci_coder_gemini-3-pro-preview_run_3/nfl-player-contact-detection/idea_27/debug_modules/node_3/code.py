import os
import sys
import shutil
import numpy as np
import pandas as pd
import xgboost as xgb

# Import library components
from library.config import Config
from library.data_loader import DataLoader
from library.feature_engineering import FeatureEngineer
from library.model_wrapper import DualStreamXGBoost
from library.pipeline_manager import PipelineManager


def set_seeds(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)


def main():
    print("=== NFL Contact Detection Library Demo ===")
    set_seeds(Config.SEED)

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Fast Execution
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Enable Debug mode to limit data volume (loads top N rows only)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 1000  # Process only 1000 rows for speed

    # Configure XGBoost for speed (few trees, CPU execution for low overhead)
    # We override the dictionary parameters directly
    Config.XGB_PARAMS_A["n_estimators"] = 5
    Config.XGB_PARAMS_A["tree_method"] = "hist"
    Config.XGB_PARAMS_B["n_estimators"] = 5
    Config.XGB_PARAMS_B["tree_method"] = "hist"

    # Reset Working Directory to ensure a clean run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Loader Demonstration
    # -------------------------------------------------------------------------
    print("\n[2] Demonstrating DataLoader...")
    loader = DataLoader(mode="train", debug=True)

    # Load Metadata
    print("Loading metadata...")
    df_meta = loader.load_metadata()
    print(f"Metadata Shape: {df_meta.shape}")
    assert not df_meta.empty, "Metadata should not be empty."

    # Load Tracking Data for specific plays
    # We pick the first few plays from the metadata to ensure matches
    sample_plays = df_meta["game_play"].unique()[:2]
    print(f"Loading tracking data for plays: {sample_plays}")
    df_tracking = loader.load_tracking_data(sample_plays, load_cached_data=False)
    print(f"Tracking Data Shape: {df_tracking.shape}")
    assert "x_position" in df_tracking.columns, "Tracking data missing 'x_position'."

    # Merge Labels with Tracking
    # We filter metadata to the sampled plays to test the merge
    df_meta_sample = df_meta[df_meta["game_play"].isin(sample_plays)].copy()
    print("Merging metadata with tracking...")
    df_merged = loader.merge_labels_with_tracking(df_meta_sample, df_tracking)
    print(f"Merged Data Shape: {df_merged.shape}")

    # Validation: Check if Player 1 tracking data exists in merged result
    # Note: Some rows might be NaN if tracking is missing for that step, but column must exist
    assert (
        "x_position_p1" in df_merged.columns
    ), "Merged data missing Player 1 tracking columns."

    # -------------------------------------------------------------------------
    # 3. Feature Engineering Demonstration
    # -------------------------------------------------------------------------
    print("\n[3] Demonstrating FeatureEngineer...")
    fe = FeatureEngineer(mode="train", debug=True)

    # Construct Stream A: Player-Player Interactions
    print("Constructing Stream A (Interaction) features...")
    X_A, y_A, ids_A = fe.construct_stream_a(load_cached_data=False)
    print(f"Stream A - Features: {X_A.shape}, Labels: {y_A.shape}")

    # Validation Stream A
    if X_A.shape[0] > 0:
        assert len(X_A) == len(y_A), "Stream A feature/label length mismatch."
        assert "distance" in X_A.columns, "Stream A missing 'distance' feature."
        # Check for temporal pyramid lags (e.g., '_lag_1')
        assert any(
            "lag" in col for col in X_A.columns
        ), "Stream A missing temporal lag features."

    # Construct Stream B: Player-Ground Impacts
    print("Constructing Stream B (Impact) features...")
    X_B, y_B, ids_B = fe.construct_stream_b(load_cached_data=False)
    print(f"Stream B - Features: {X_B.shape}, Labels: {y_B.shape}")

    # Validation Stream B
    if X_B.shape[0] > 0:
        assert len(X_B) == len(y_B), "Stream B feature/label length mismatch."
        assert "v_surge" in X_B.columns, "Stream B missing invariant feature 'v_surge'."

    # -------------------------------------------------------------------------
    # 4. Model Wrapper Demonstration
    # -------------------------------------------------------------------------
    print("\n[4] Demonstrating DualStreamXGBoost...")
    model_wrapper = DualStreamXGBoost()

    # Train Stream A Model (if data available)
    if X_A.shape[0] > 20:
        print("Training Stream A model...")
        model_wrapper.fit_stream(X_A.values, y_A, stream="A")

        # Test Prediction
        preds_A = model_wrapper.predict_stream(X_A.values, stream="A")
        print(f"Stream A Predictions (Mean Prob): {preds_A.mean():.4f}")
        assert len(preds_A) == len(X_A), "Prediction count mismatch."

    # Train Stream B Model (if data available)
    if X_B.shape[0] > 20:
        print("Training Stream B model...")
        model_wrapper.fit_stream(X_B.values, y_B, stream="B")

        # Test Prediction
        preds_B = model_wrapper.predict_stream(X_B.values, stream="B")
        print(f"Stream B Predictions (Mean Prob): {preds_B.mean():.4f}")

    # Save Models
    print("Saving models...")
    model_wrapper.save_models()

    # Verify persistence
    model_path_a = os.path.join(Config.WORKING_DIR, "model_stream_a.json")
    model_path_b = os.path.join(Config.WORKING_DIR, "model_stream_b.json")
    # At least one model should exist if data was sufficient
    if X_A.shape[0] > 20 or X_B.shape[0] > 20:
        assert os.path.exists(model_path_a) or os.path.exists(
            model_path_b
        ), "Model files were not saved."

    # -------------------------------------------------------------------------
    # 5. Pipeline Manager Demonstration (Full Workflow)
    # -------------------------------------------------------------------------
    print("\n[5] Demonstrating PipelineManager (End-to-End)...")
    pipeline = PipelineManager(debug=True)

    # A. Run Training Pipeline
    # This orchestrates FE (Train/Val), Training, and Threshold Optimization
    print("Running Training Pipeline...")
    # We use load_cached_data=True to leverage features computed in Step 3 where applicable
    thresholds = pipeline.run_training_pipeline(load_cached_data=True)

    print(f"Optimized Thresholds: {thresholds}")
    assert "A" in thresholds and "B" in thresholds, "Threshold dictionary incomplete."

    # B. Run Inference Pipeline
    # This generates the submission file
    print("Running Inference Pipeline...")
    pipeline.run_inference_pipeline(thresholds=thresholds, load_cached_data=False)

    # Validation: Check Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission File Loaded. Shape: {df_sub.shape}")
    print(df_sub.head())

    # Verify submission schema
    assert "contact_id" in df_sub.columns, "Submission missing 'contact_id'."
    assert "contact" in df_sub.columns, "Submission missing 'contact'."
    assert (
        df_sub["contact"].dtype == int or df_sub["contact"].dtype == np.int64
    ), "Contact column must be integer."

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
