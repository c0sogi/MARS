import os
import sys
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Import library components
from library.config import Config
from library.data_loader import DataLoader
from library.feature_engineering import FeatureEngine
from library.model_factory import ModelFactory
from library.training import Trainer
from library.inference import Predictor
from library.utils import setup_seed


def run_demo():
    print("=" * 50)
    print("Starting End-to-End Pipeline Demonstration")
    print("=" * 50)

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[Step 1] Configuring Environment...")

    # Set fixed seed for reproducibility
    setup_seed(42)

    # Override Config for Speed
    # Enable Debug mode to use a small subset of data (approx 5% of plays)
    Config.DEBUG = True

    # Reduce model complexity to ensure instant training for demonstration
    # We modify the dictionaries in Config directly
    Config.XGB_PARAMS_STREAM_A["n_estimators"] = 2
    Config.XGB_PARAMS_STREAM_A["max_depth"] = 2
    Config.XGB_PARAMS_STREAM_B["n_estimators"] = 2
    Config.XGB_PARAMS_STREAM_B["max_depth"] = 2

    print("Configuration updated: DEBUG=True, n_estimators=2")

    # -------------------------------------------------------------------------
    # 2. Data Loading Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 2] Demonstrating DataLoader...")

    dl = DataLoader(debug=True)

    # Load Metadata
    df_train_meta = dl.load_metadata("train")
    print(f"Loaded Train Metadata: {df_train_meta.shape}")

    # Validate Metadata
    assert not df_train_meta.empty, "Train metadata should not be empty"
    assert "game_play" in df_train_meta.columns
    assert "contact" in df_train_meta.columns

    # Load Tracking (Subset for efficiency)
    # In a real run, we might load all, but here we just verify it loads
    df_tracking = dl.load_tracking("train")
    print(f"Loaded Tracking Data: {df_tracking.shape}")
    assert not df_tracking.empty

    # Load Helmets
    df_helmets = dl.load_helmets("train")
    print(f"Loaded Helmets Data: {df_helmets.shape}")
    assert not df_helmets.empty

    # -------------------------------------------------------------------------
    # 3. Feature Engineering Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 3] Demonstrating FeatureEngine...")

    fe = FeatureEngine(debug=True)

    # Create a tiny slice for feature generation demo to avoid processing the whole debug set twice
    # (The Trainer will process the full debug set later)
    sample_play = df_train_meta["game_play"].unique()[0]
    df_meta_sample = (
        df_train_meta[df_train_meta["game_play"] == sample_play].head(50).copy()
    )
    df_tracking_sample = df_tracking[df_tracking["game_play"] == sample_play].copy()
    df_helmets_sample = df_helmets[df_helmets["game_play"] == sample_play].copy()

    # Demo Stream A (Player-Player) Feature Gen
    # Filter for non-ground contacts
    df_sample_A = df_meta_sample[df_meta_sample["nfl_player_id_2"] != "G"].copy()
    if not df_sample_A.empty:
        print("Generating Stream A features for sample...")
        feats_A = fe.generate_stream_a_features(
            df_sample_A, df_tracking_sample, df_helmets_sample, load_cached_data=False
        )
        print(f"Stream A Features Shape: {feats_A.shape}")

        # Verify columns
        expected_cols = [c for c in Config.STREAM_A_FEATURES if c in feats_A.columns]
        assert len(expected_cols) > 0, "Stream A features were not generated."
        assert "contact" in feats_A.columns

    # Demo Stream B (Player-Ground) Feature Gen
    # Filter for ground contacts
    df_sample_B = df_meta_sample[df_meta_sample["nfl_player_id_2"] == "G"].copy()
    if not df_sample_B.empty:
        print("Generating Stream B features for sample...")
        feats_B = fe.generate_stream_b_features(
            df_sample_B, df_tracking_sample, load_cached_data=False
        )
        print(f"Stream B Features Shape: {feats_B.shape}")

        expected_cols_B = [c for c in Config.STREAM_B_FEATURES if c in feats_B.columns]
        assert len(expected_cols_B) > 0, "Stream B features were not generated."

    # -------------------------------------------------------------------------
    # 4. Training Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 4] Demonstrating Trainer (Training Pipeline)...")

    trainer = Trainer(debug=True)

    # This executes the full training loop:
    # Load Data -> Generate Features (cached) -> Undersample -> Train XGBoost -> Optimize Threshold
    training_results = trainer.train()

    # Validate Results
    assert "A" in training_results, "Stream A training failed or returned no results."
    assert "B" in training_results, "Stream B training failed or returned no results."

    model_a = training_results["A"]["model"]
    thresh_a = training_results["A"]["threshold"]
    model_b = training_results["B"]["model"]
    thresh_b = training_results["B"]["threshold"]

    print(
        f"Stream A - Threshold: {thresh_a:.4f}, Score: {training_results['A']['score']:.4f}"
    )
    print(
        f"Stream B - Threshold: {thresh_b:.4f}, Score: {training_results['B']['score']:.4f}"
    )

    assert model_a is not None
    assert model_b is not None

    # -------------------------------------------------------------------------
    # 5. Inference Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 5] Demonstrating Predictor (Inference Pipeline)...")

    predictor = Predictor(model_a, thresh_a, model_b, thresh_b, debug=True)

    # Run prediction on the test set (debug mode uses a subset of test metadata if implemented,
    # but test set is usually small enough. The DataLoader debug logic handles sampling).
    predictor.predict()

    # Validate Submission File
    submission_path = Config.SUBMISSION_PATH
    if os.path.exists(submission_path):
        df_sub = pd.read_csv(submission_path)
        print(f"Submission file generated at: {submission_path}")
        print(f"Submission Shape: {df_sub.shape}")
        print(f"Submission Columns: {df_sub.columns.tolist()}")
        print("First 5 rows:")
        print(df_sub.head())

        # Assertions
        assert "contact_id" in df_sub.columns
        assert "contact" in df_sub.columns
        assert (
            df_sub["contact"].isin([0, 1]).all()
        ), "Predictions must be binary (0 or 1)"
        assert len(df_sub) > 0, "Submission file is empty"
    else:
        raise FileNotFoundError(f"Submission file was not created at {submission_path}")

    print("\n" + "=" * 50)
    print("Demonstration Completed Successfully")
    print("=" * 50)


if __name__ == "__main__":
    run_demo()
