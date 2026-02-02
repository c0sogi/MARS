import os
import pandas as pd
import numpy as np
import shutil
from library.config import Config
from library.data_loader import DataLoader
from library.feature_engineering import FeatureGenerator
from library.train import TrainPipeline
from library.inference import InferencePipeline
from library.utils import setup_seed

if __name__ == "__main__":
    print("=== Starting NFL Contact Detection Demo Script ===")

    # 1. Setup Environment and Configuration Overrides
    # We use a specific demo directory to avoid messing with real training artifacts
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    print(f"Setting up demo configuration in {demo_dir}...")

    # Override Global Config for Speed
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_OUTPUT_PATH = os.path.join(demo_dir, "submission.csv")

    # Reduce complexity for demo speed
    # Only use immediate context (current frame + 1 lag) instead of deep history
    Config.LAGS = [0, 1]

    # Reduce Model complexity (very few trees for fast training)
    Config.XGB_PARAMS_STREAM_A["n_estimators"] = 10
    Config.XGB_PARAMS_STREAM_A["early_stopping_rounds"] = (
        None  # Disable early stopping for fixed small iters
    )
    Config.XGB_PARAMS_STREAM_B["n_estimators"] = 10
    Config.XGB_PARAMS_STREAM_B["early_stopping_rounds"] = None

    # 2. Create Mini Datasets (Subsetting)
    # We read the generated metadata and save small subsets to the demo directory.
    # This forces the pipelines to run on small data without modifying the library code.
    print("Creating mini datasets for rapid testing...")

    # Load original metadata
    df_train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    df_val_meta = pd.read_csv(Config.VAL_META_PATH)
    df_test_meta = pd.read_csv(Config.TEST_META_PATH)

    # Select a small number of plays (groups)
    train_plays = df_train_meta["game_play"].unique()[:2]  # 2 plays for training
    val_plays = df_val_meta["game_play"].unique()[:1]  # 1 play for validation
    test_plays = df_test_meta["game_play"].unique()[:1]  # 1 play for testing

    # Filter DataFrames
    mini_train = df_train_meta[df_train_meta["game_play"].isin(train_plays)].copy()
    mini_val = df_val_meta[df_val_meta["game_play"].isin(val_plays)].copy()
    mini_test = df_test_meta[df_test_meta["game_play"].isin(test_plays)].copy()

    # Save Mini Metadata
    mini_train_path = os.path.join(demo_dir, "mini_train.csv")
    mini_val_path = os.path.join(demo_dir, "mini_validation.csv")
    mini_test_path = os.path.join(demo_dir, "mini_test.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    print(f"Mini Train shape: {mini_train.shape}")
    print(f"Mini Val shape: {mini_val.shape}")
    print(f"Mini Test shape: {mini_test.shape}")

    # Point Config to Mini Metadata
    Config.TRAIN_META_PATH = mini_train_path
    Config.VAL_META_PATH = mini_val_path
    Config.TEST_META_PATH = mini_test_path

    # 3. Demonstrate Data Loading & Merging Logic (Explicit Check)
    print("\n--- Verifying Data Loader Logic ---")
    loader = DataLoader(run_mode="train")

    # Load metadata (should be our mini version)
    meta = loader.load_metadata()
    assert len(meta) == len(
        mini_train
    ), "DataLoader did not load the mini dataset correctly."

    # Load tracking for these plays
    track = loader.load_tracking(meta["game_play"].unique())
    assert not track.empty, "Tracking data should not be empty."

    # Load helmets
    helm = loader.load_helmets(meta["game_play"].unique())
    assert not helm.empty, "Helmet data should not be empty."

    # Merge
    merged_df = loader.merge_data(meta, track, helm, load_cached_data=False)
    assert (
        "x_position_p1" in merged_df.columns
    ), "Merged data missing Player 1 tracking columns."
    assert (
        "view_sideline_left_p1" in merged_df.columns
    ), "Merged data missing Player 1 helmet columns."
    print("Data Loader logic verified.")

    # 4. Demonstrate Feature Engineering Logic (Explicit Check)
    print("\n--- Verifying Feature Engineering Logic ---")
    fg = FeatureGenerator(run_mode="train")

    # Generate Stream A features
    X_a, y_a, ids_a = fg.generate_features(
        merged_df, stream="stream_a", load_cached_data=False
    )
    assert not X_a.empty, "Stream A features should not be empty."
    assert (
        "dist_p1_p2_lag0" in X_a.columns
    ), "Stream A features missing calculated distance."
    assert (
        "dist_p1_p2_lag_neg1" in X_a.columns
    ), "Stream A features missing lagged columns."

    # Generate Stream B features
    X_b, y_b, ids_b = fg.generate_features(
        merged_df, stream="stream_b", load_cached_data=False
    )
    assert not X_b.empty, "Stream B features should not be empty."
    assert (
        "surge_v_lag0" in X_b.columns
    ), "Stream B features missing ego-centric columns."
    print("Feature Engineering logic verified.")

    # 5. Run Training Pipeline
    print("\n--- Running Training Pipeline (Mini Mode) ---")
    trainer = TrainPipeline()
    # We disable caching to force the pipeline to use our new mini datasets and parameters
    trainer.run_training(load_cached_data=False)

    # Verify outputs
    assert os.path.exists(
        os.path.join(demo_dir, "model_stream_a.json")
    ), "Stream A model not saved."
    assert os.path.exists(
        os.path.join(demo_dir, "model_stream_b.json")
    ), "Stream B model not saved."
    assert os.path.exists(
        os.path.join(demo_dir, "thresholds.json")
    ), "Thresholds file not saved."
    print("Training Pipeline execution successful.")

    # 6. Run Inference Pipeline
    print("\n--- Running Inference Pipeline (Mini Mode) ---")
    inferencer = InferencePipeline()
    inferencer.run_inference()

    # Verify Submission
    submission_path = Config.SUBMISSION_OUTPUT_PATH
    assert os.path.exists(submission_path), "Submission file not created."

    df_sub = pd.read_csv(submission_path)

    # The submission should contain rows for the mini test set.
    # Note: The InferencePipeline merges with sample_submission.csv.
    # Since we only predicted for the mini test set, the pipeline fills the rest with 0.
    # However, for the rows we *did* predict (from mini_test.csv), we should check existence.

    # Get IDs from our mini test set
    expected_ids = mini_test["contact_id"].unique()

    # Check if these IDs are in the submission
    submission_ids = df_sub["contact_id"].values
    missing_ids = [mid for mid in expected_ids if mid not in submission_ids]

    assert (
        len(missing_ids) == 0
    ), f"Submission is missing IDs from the test set: {missing_ids[:5]}"

    # Check that we have a 'contact' column with binary values
    assert "contact" in df_sub.columns, "Submission missing 'contact' column."
    assert (
        df_sub["contact"].isin([0, 1]).all()
    ), "Submission contains non-binary values."

    print(
        f"Inference Pipeline execution successful. Submission generated at {submission_path}"
    )
    print("\n=== Demo Completed Successfully ===")
