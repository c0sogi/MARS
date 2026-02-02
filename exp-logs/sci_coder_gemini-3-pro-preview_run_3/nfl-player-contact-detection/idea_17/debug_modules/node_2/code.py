import os
import shutil
import pandas as pd
import numpy as np
import xgboost as xgb
import sys

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, calc_mcc, optimize_thresholds
from library.data_factory import DataFactory
from library.feature_engine import FeatureEngine
from library.model_trainer import StreamTrainer, generate_submission
from library.orchestrator import Pipeline


def run_demo():
    print("=== Setting up Demo Configuration ===")
    # 1. Runtime Configuration Overrides for Speed
    # We modify the Config class directly to ensure these settings propagate to all modules
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Update Cache Paths in Config to point to demo dir
    Config.CACHE_STREAM_A_TRAIN = os.path.join(DEMO_DIR, "streamA_train.parquet")
    Config.CACHE_STREAM_A_VAL = os.path.join(DEMO_DIR, "streamA_val.parquet")
    Config.CACHE_STREAM_A_TEST = os.path.join(DEMO_DIR, "streamA_test.parquet")
    Config.CACHE_STREAM_B_TRAIN = os.path.join(DEMO_DIR, "streamB_train.parquet")
    Config.CACHE_STREAM_B_VAL = os.path.join(DEMO_DIR, "streamB_val.parquet")
    Config.CACHE_STREAM_B_TEST = os.path.join(DEMO_DIR, "streamB_test.parquet")

    # Reduce Model Complexity for Demo
    Config.XGB_PARAMS["n_estimators"] = 5
    Config.XGB_PARAMS["early_stopping_rounds"] = 1
    Config.XGB_PARAMS["verbosity"] = 0
    Config.EARLY_STOPPING_ROUNDS = 1
    Config.VERBOSE_EVAL = False

    # Set seed
    seed_everything(Config.SEED)
    print("Configuration updated for rapid execution.")

    # -------------------------------------------------------------------------
    # 2. Verify Utilities
    # -------------------------------------------------------------------------
    print("\n=== Verifying Utilities ===")
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 1])
    mcc = calc_mcc(y_true, y_pred)
    print(f"Calculated MCC: {mcc:.4f}")
    assert -1.0 <= mcc <= 1.0, "MCC should be between -1 and 1"

    y_probs = np.array([0.9, 0.1, 0.4, 0.8, 0.6])
    best_thresh, best_score = optimize_thresholds(y_true, y_probs, num_steps=20)
    print(f"Optimized Threshold: {best_thresh:.4f}, Score: {best_score:.4f}")
    assert 0.0 < best_thresh < 1.0, "Threshold should be valid probability"

    # -------------------------------------------------------------------------
    # 3. Verify Data Factory
    # -------------------------------------------------------------------------
    print("\n=== Verifying Data Factory ===")
    factory = DataFactory()

    # Load small subset (debug mode)
    # Using a small debug_size to ensure speed
    debug_size = 500
    df_meta, df_track, df_helmets = factory.load_dataset(
        mode="train", load_cached_data=False, debug=True, debug_size=debug_size
    )

    print(f"Loaded Meta Shape: {df_meta.shape}")
    print(f"Loaded Tracking Shape: {df_track.shape}")
    print(f"Loaded Helmets Shape: {df_helmets.shape}")

    assert not df_meta.empty, "Metadata should not be empty"
    assert not df_track.empty, "Tracking data should not be empty"
    # Helmets might be empty if the sampled plays don't have helmet data, but unlikely with 500 rows.
    # However, we won't assert df_helmets is not empty strictly if the sample is very unlucky,
    # but for 500 rows it should be fine.

    # Split Streams
    df_a, df_b = factory.split_contact_ids(df_meta)
    print(f"Stream A (Player-Player) count: {len(df_a)}")
    print(f"Stream B (Player-Ground) count: {len(df_b)}")

    # Validation: Stream B should have 'G' as player 2
    if not df_b.empty:
        assert (
            df_b["nfl_player_id_2"] == "G"
        ).all(), "Stream B must contain only Ground contacts"

    # Validation: Stream A should NOT have 'G' as player 2
    if not df_a.empty:
        assert (
            df_a["nfl_player_id_2"] != "G"
        ).all(), "Stream A must not contain Ground contacts"

    # -------------------------------------------------------------------------
    # 4. Verify Feature Engine
    # -------------------------------------------------------------------------
    print("\n=== Verifying Feature Engine ===")
    engine = FeatureEngine()

    # Build Stream A Features
    # We use the data loaded above
    if not df_a.empty:
        print("Building Stream A features...")
        feats_a = engine.build_stream_a(
            df_a, df_track, df_helmets, mode="train", load_cached_data=False
        )
        print(f"Stream A Features Shape: {feats_a.shape}")

        # Check for specific engineered columns
        expected_cols_a = ["dist_p1_p2", "closure_rate"]
        for col in expected_cols_a:
            assert col in feats_a.columns, f"Stream A missing column: {col}"

        # Check target exists
        assert "contact" in feats_a.columns, "Stream A missing target column 'contact'"

    # Build Stream B Features
    if not df_b.empty:
        print("Building Stream B features...")
        feats_b = engine.build_stream_b(
            df_b, df_track, df_helmets, mode="train", load_cached_data=False
        )
        print(f"Stream B Features Shape: {feats_b.shape}")

        # Check for specific engineered columns (Ego motion)
        expected_cols_b = ["surge_a", "sway_a"]
        for col in expected_cols_b:
            assert col in feats_b.columns, f"Stream B missing column: {col}"

    # -------------------------------------------------------------------------
    # 5. Verify Model Trainer
    # -------------------------------------------------------------------------
    print("\n=== Verifying Model Trainer ===")

    # We will simulate a train/val split from the features generated above
    # Use Stream A for demonstration if available, else Stream B
    demo_df = feats_a if not df_a.empty else feats_b
    stream_name = "StreamA" if not df_a.empty else "StreamB"

    if not demo_df.empty:
        # Simple split
        split_idx = int(len(demo_df) * 0.8)
        train_df = demo_df.iloc[:split_idx]
        val_df = demo_df.iloc[split_idx:]

        # Ensure we have both classes in train/val for valid training
        # If not, we mock it for the sake of code verification
        if train_df["contact"].nunique() < 2:
            print(
                "Warning: Training set missing a class. Injecting dummy data for verification."
            )
            row = train_df.iloc[0].copy()
            row["contact"] = 1 - row["contact"]
            train_df = pd.concat([train_df, pd.DataFrame([row])], ignore_index=True)

        if val_df["contact"].nunique() < 2:
            print(
                "Warning: Validation set missing a class. Injecting dummy data for verification."
            )
            row = val_df.iloc[0].copy()
            row["contact"] = 1 - row["contact"]
            val_df = pd.concat([val_df, pd.DataFrame([row])], ignore_index=True)

        trainer = StreamTrainer(stream_name)

        # Train
        trainer.train(train_df, val_df)

        assert trainer.model is not None, "Model should be trained"
        assert 0.0 < trainer.best_threshold < 1.0, "Threshold should be optimized"

        # Predict
        preds = trainer.predict(val_df)
        print(f"Predictions Shape: {preds.shape}")
        assert "contact" in preds.columns, "Predictions missing 'contact' column"
        assert "prob" in preds.columns, "Predictions missing 'prob' column"

        # Save/Load
        model_file = "demo_model.json"
        trainer.save_model(model_file)
        assert os.path.exists(
            os.path.join(Config.WORKING_DIR, model_file)
        ), "Model file not saved"

        trainer.load_model(model_file)
        print("Model save/load verified.")

    # -------------------------------------------------------------------------
    # 6. Verify Orchestrator (Full Pipeline)
    # -------------------------------------------------------------------------
    print("\n=== Verifying Orchestrator (Pipeline) ===")
    pipeline = Pipeline()

    # Run Training Pipeline (Debug)
    # This runs the full flow: Load -> Split -> Engineer -> Train -> Save
    pipeline.run_training(debug=True, debug_size=200)

    assert os.path.exists(
        pipeline.thresholds_path
    ), "Thresholds file not created by pipeline"

    # Run Inference Pipeline (Debug)
    # This runs: Load Test -> Split -> Engineer -> Load Model -> Predict -> Submit
    pipeline.run_inference(debug=True, debug_size=200)

    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), "Submission file not created by pipeline"

    # Check submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission File Rows: {len(sub_df)}")
    assert (
        "contact_id" in sub_df.columns and "contact" in sub_df.columns
    ), "Invalid submission schema"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
