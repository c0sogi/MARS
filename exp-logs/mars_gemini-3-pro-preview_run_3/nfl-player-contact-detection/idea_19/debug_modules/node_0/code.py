import pandas as pd
import numpy as np
import os
import xgboost as xgb
import sys

# Import library components
from library.config import Config
from library.data_manager import DataManager
from library.feature_stream_a import StreamAFeatureGenerator
from library.feature_stream_b import StreamBFeatureGenerator
from library.model_factory import DualStreamModel
from library.optimizer import ThresholdOptimizer


def run_demo():
    print("=== NFL Contact Detection Pipeline Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration Patching (Optimize for Speed)
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")
    # Reduce boosting rounds to ensure training finishes instantly
    Config.TRAIN_CONFIG["num_boost_round"] = 10
    Config.TRAIN_CONFIG["early_stopping_rounds"] = 5
    Config.TRAIN_CONFIG["verbose_eval"] = 5

    # Set global seed
    np.random.seed(Config.SEED)

    # ---------------------------------------------------------
    # 2. Mocking Data Manager (Subset Data)
    # ---------------------------------------------------------
    print("\n[2] Patching DataManager to use small dataset subsets...")

    # Store original method to call it inside mock
    original_load_dataset = DataManager.load_dataset

    def mocked_load_dataset(self, mode="train", load_cached_data=False):
        """
        Intercepts data loading to return a small 5000-row sample of the
        validation set, regardless of the requested mode. This avoids
        processing millions of rows during the demo.
        """
        print(f"    [Mock] Loading data subset (overriding mode='{mode}')...")
        # Always load validation set as it is smaller (800k rows) than train (3.4M)
        # We allow the DataManager to read the CSV, then we slice it immediately.
        # Note: In a real run, we would let it cache the full file.
        df = original_load_dataset(
            self, mode="validation", load_cached_data=load_cached_data
        )

        # Sample 5000 rows to ensure we have a mix of P-P and P-G interactions
        # Using a fixed seed for reproducibility
        if len(df) > 5000:
            print(f"    [Mock] Sampling 5000 rows from {len(df)} rows.")
            df_subset = df.sample(n=5000, random_state=Config.SEED).copy()
            return df_subset
        return df

    def mocked_load_helmets(self, mode="train"):
        """
        Returns an empty dataframe with correct schema to skip loading
        the massive helmets CSV file.
        """
        print("    [Mock] Returning dummy helmets data to save time.")
        cols = [
            "game_play",
            "game_key",
            "play_id",
            "view",
            "video",
            "frame",
            "nfl_player_id",
            "player_label",
            "left",
            "width",
            "top",
            "height",
        ]
        return pd.DataFrame(columns=cols)

    # Apply patches
    DataManager.load_dataset = mocked_load_dataset
    DataManager.load_helmets = mocked_load_helmets

    # ---------------------------------------------------------
    # 3. Feature Generation Demo
    # ---------------------------------------------------------
    print("\n[3] Generating Features (Stream A & Stream B)...")

    # --- Stream A: Player-Player ---
    print("  -> Generating Stream A (Interaction) features...")
    gen_a = StreamAFeatureGenerator()
    # Force load_cached_data=False to ensure our mock is triggered
    df_a = gen_a.generate_features(mode="train", load_cached_data=False)

    # Verify Stream A
    assert not df_a.empty, "Stream A features should not be empty."
    assert "dist_p1_p2" in df_a.columns, "Missing relational feature 'dist_p1_p2'."
    assert "closure_rate" in df_a.columns, "Missing relational feature 'closure_rate'."
    print(f"     Stream A Shape: {df_a.shape}")

    # --- Stream B: Player-Ground ---
    print("  -> Generating Stream B (Impact) features...")
    gen_b = StreamBFeatureGenerator()
    df_b = gen_b.generate_features(mode="train", load_cached_data=False)

    # Verify Stream B
    assert not df_b.empty, "Stream B features should not be empty."
    assert "v_surge" in df_b.columns, "Missing ego-centric feature 'v_surge'."
    assert "jerk_mag_p1" in df_b.columns, "Missing kinematic feature 'jerk_mag_p1'."
    print(f"     Stream B Shape: {df_b.shape}")

    # ---------------------------------------------------------
    # 4. Model Training Demo
    # ---------------------------------------------------------
    print("\n[4] Training DualStreamModel...")
    model = DualStreamModel()

    # Create synthetic Train/Val splits from our generated subsets
    # We use 80% for train, 20% for val
    def split_data(df):
        split_idx = int(len(df) * 0.8)
        return df.iloc[:split_idx], df.iloc[split_idx:]

    train_a, val_a = split_data(df_a)
    train_b, val_b = split_data(df_b)

    # Train the model
    model.train(train_a, train_b, val_a, val_b)

    # Verify models exist
    assert "stream_a" in model.models, "Stream A model failed to train."
    assert "stream_b" in model.models, "Stream B model failed to train."
    print("    Training complete.")

    # ---------------------------------------------------------
    # 5. Threshold Optimization Demo
    # ---------------------------------------------------------
    print("\n[5] Demonstrating Threshold Optimization...")

    # We will manually optimize threshold for Stream A validation set to show usage
    # Extract features and labels
    feats_a = model.features["stream_a"]
    X_val_a = val_a[feats_a]
    y_val_a = val_a["contact"].values

    # Get raw probabilities
    dval_a = xgb.DMatrix(X_val_a)
    probs_a = model.models["stream_a"].predict(dval_a)

    # Use Optimizer
    optimizer = ThresholdOptimizer()
    best_thresh, best_mcc = optimizer.optimize_thresholds(
        y_val_a, probs_a, num_steps=10
    )

    print(
        f"    Stream A Optimization -> Threshold: {best_thresh:.4f}, MCC: {best_mcc:.4f}"
    )

    # ---------------------------------------------------------
    # 6. Inference & Submission Demo
    # ---------------------------------------------------------
    print("\n[6] Generating Predictions and Saving Submission...")

    # We use our validation subsets as "test" data for this demo
    # The predict method handles feature selection internally
    preds = model.predict(val_a, val_b)

    # Verify Prediction Output
    assert "contact_id" in preds.columns, "Prediction output missing 'contact_id'."
    assert "contact" in preds.columns, "Prediction output missing 'contact'."
    print(f"    Generated {len(preds)} predictions.")

    # Save Submission
    # This merges our partial predictions with the full sample_submission.csv
    model.save_submission(preds)

    # Verify File Creation
    submission_path = Config.PATH_CONFIG["submission_path"]
    if os.path.exists(submission_path):
        print(f"    Success: Submission file found at {submission_path}")
    else:
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
