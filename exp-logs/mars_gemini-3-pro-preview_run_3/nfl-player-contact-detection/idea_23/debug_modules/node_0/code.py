import os
import sys
import pandas as pd
import numpy as np
import warnings
import shutil

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.data_loader import DataLoader
from library.feature_engineering import FeatureEngineer
from library.model_trainer import ModelTrainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demo Run ===")

    # 1. Setup & Configuration Overrides for Speed
    # We modify the Config class attributes directly to ensure fast execution for the demo.
    print("Configuring environment for fast demonstration...")
    seed_everything(Config.SEED)

    # Reduce number of estimators for XGBoost to ensure training finishes in seconds
    Config.XGB_COMMON_PARAMS["n_estimators"] = 10
    Config.XGB_PARAMS_STREAM_A["n_estimators"] = 10
    Config.XGB_PARAMS_STREAM_B["n_estimators"] = 10

    # Use a separate working directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Data Loading
    print("\n[Step 1] Loading Data...")
    loader = DataLoader()

    # Load a tiny subset of plays (max_plays=3) for Train, Validation, and Test
    # This ensures the heavy feature engineering runs quickly.
    df_meta_train, df_track_train, df_helm_train = loader.load_dataset(
        mode="train", max_plays=3
    )
    df_meta_val, df_track_val, df_helm_val = loader.load_dataset(
        mode="validation", max_plays=3
    )
    df_meta_test, df_track_test, df_helm_test = loader.load_dataset(
        mode="test", max_plays=3
    )

    # Validation: Ensure data is loaded
    assert not df_meta_train.empty, "Training metadata should not be empty"
    assert not df_track_train.empty, "Training tracking data should not be empty"
    assert not df_helm_train.empty, "Training helmet data should not be empty"
    print(
        f"Loaded {len(df_meta_train)} training labels across {df_meta_train['game_play'].nunique()} plays."
    )

    # 3. Stream Splitting
    print(
        "\n[Step 2] Splitting Data into Streams (A: Player-Player, B: Player-Ground)..."
    )

    # Train Split
    train_stream_a, train_stream_b = loader.get_stream_data(df_meta_train)
    # Validation Split
    val_stream_a, val_stream_b = loader.get_stream_data(df_meta_val)
    # Test Split
    test_stream_a, test_stream_b = loader.get_stream_data(df_meta_test)

    # Validation: Check logic of split
    # Stream B should only contain 'G' as player 2
    if not train_stream_b.empty:
        assert (
            train_stream_b["nfl_player_id_2"].eq("G").all()
        ), "Stream B must only contain Ground contacts"
    # Stream A should not contain 'G'
    if not train_stream_a.empty:
        assert (
            train_stream_a["nfl_player_id_2"] != "G"
        ).all(), "Stream A must not contain Ground contacts"

    print(
        f"Train Split - Stream A: {len(train_stream_a)}, Stream B: {len(train_stream_b)}"
    )

    # 4. Feature Engineering
    print("\n[Step 3] Generating Features...")
    engineer = FeatureEngineer()

    # --- Stream A (Interaction) ---
    print("Processing Stream A Features...")
    # Train
    X_train_a, y_train_a, ids_train_a = engineer.process_stream_a(
        train_stream_a, df_track_train, df_helm_train, load_cached_data=False
    )
    # Validation
    X_val_a, y_val_a, ids_val_a = engineer.process_stream_a(
        val_stream_a, df_track_val, df_helm_val, load_cached_data=False
    )
    # Test
    X_test_a, y_test_a, ids_test_a = engineer.process_stream_a(
        test_stream_a, df_track_test, df_helm_test, load_cached_data=False
    )

    # Validation: Feature alignment
    assert len(X_train_a) == len(y_train_a), "Stream A Train X and y length mismatch"
    assert len(X_train_a) == len(
        ids_train_a
    ), "Stream A Train X and IDs length mismatch"

    # --- Stream B (Impact) ---
    print("Processing Stream B Features...")
    # Train
    X_train_b, y_train_b, ids_train_b = engineer.process_stream_b(
        train_stream_b, df_track_train, load_cached_data=False
    )
    # Validation
    X_val_b, y_val_b, ids_val_b = engineer.process_stream_b(
        val_stream_b, df_track_val, load_cached_data=False
    )
    # Test
    X_test_b, y_test_b, ids_test_b = engineer.process_stream_b(
        test_stream_b, df_track_test, load_cached_data=False
    )

    # Validation: Feature alignment
    assert len(X_train_b) == len(y_train_b), "Stream B Train X and y length mismatch"

    # 5. Model Training
    print("\n[Step 4] Training Models...")
    trainer = ModelTrainer()

    # Train Stream A
    if not X_train_a.empty and not X_val_a.empty:
        model_a, thresh_a, mcc_a = trainer.train_stream(
            X_train_a, y_train_a, X_val_a, y_val_a, "A"
        )
        assert "A" in trainer.models, "Model A was not registered in trainer"
        assert 0.0 < thresh_a < 1.0, f"Threshold A {thresh_a} is out of expected range"
    else:
        print("Skipping Stream A training (insufficient data in subset)")

    # Train Stream B
    if not X_train_b.empty and not X_val_b.empty:
        model_b, thresh_b, mcc_b = trainer.train_stream(
            X_train_b, y_train_b, X_val_b, y_val_b, "B"
        )
        assert "B" in trainer.models, "Model B was not registered in trainer"
    else:
        print("Skipping Stream B training (insufficient data in subset)")

    # 6. Prediction & Submission
    print("\n[Step 5] Generating Submission...")

    # Override submission path to demo folder
    Config.SUBMISSION_PATH = os.path.join(
        Config.WORKING_DIR, "submission", "submission.csv"
    )
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    trainer.predict_and_submit(ids_test_a, X_test_a, ids_test_b, X_test_b)

    # 7. Final Verification
    print("\n[Step 6] Verifying Submission...")
    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file created successfully.")
        print(f"Shape: {df_sub.shape}")
        print(f"Columns: {list(df_sub.columns)}")

        assert "contact_id" in df_sub.columns, "Missing contact_id column"
        assert "contact" in df_sub.columns, "Missing contact column"
        assert df_sub["contact"].isin([0, 1]).all(), "Predictions must be binary"

        # Check against sample submission IDs (intersection only, since we used a subset)
        # Note: In a real run, we must match exactly. Here we just check format.
        print("Verification passed.")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demo Run Completed Successfully ===")


if __name__ == "__main__":
    main()
