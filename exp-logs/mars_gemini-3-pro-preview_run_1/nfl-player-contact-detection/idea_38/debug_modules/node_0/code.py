import os
import sys
import numpy as np
import pandas as pd
import warnings
import shutil
import gc

# Import from the provided library
from library.config import Config
from library.data_factory import DataFactory
from library.feature_engine import FeatureEngine
from library.training_pipeline import TrainingPipeline
from library.model_zoo import EnsemblePredictor, LGBMWrapper, XGBWrapper
from library.utils import load_npy, load_joblib

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def main():
    print("Initializing Demo Script...")

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Speed and Demo Isolation
    # -------------------------------------------------------------------------
    # Set a specific cache directory for this run to avoid conflicts
    DEMO_DIR = "./working/demo_run/"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config global settings
    Config.CACHE_DIR = DEMO_DIR
    Config.N_JOBS = 4  # Limit threads for demo

    # Override Model Hyperparameters for extremely fast training
    # We just want to prove the code runs, not achieve high accuracy here.
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["num_leaves"] = 8
    Config.LGBM_PARAMS["learning_rate"] = 0.1

    Config.XGB_PARAMS["n_estimators"] = 10
    Config.XGB_PARAMS["max_depth"] = 3
    Config.XGB_PARAMS["learning_rate"] = 0.1

    Config.EARLY_STOPPING_ROUNDS = 2

    # Set Seed
    np.random.seed(Config.SEED)

    print(f"Configuration configured. Output directory: {Config.CACHE_DIR}")

    # -------------------------------------------------------------------------
    # 2. Run Training Pipeline
    # -------------------------------------------------------------------------
    print("\n--- Starting Training Pipeline Demo ---")

    pipeline = TrainingPipeline()

    # Run pipeline with a small sample size (5000 rows)
    # This triggers:
    # 1. Feature Generation (Train/Val)
    # 2. Scout Model Training
    # 3. Hard Negative Mining
    # 4. Expert Model Training
    # 5. Threshold Optimization
    pipeline.run(load_cached_data=False, sample_size=5000)

    # -------------------------------------------------------------------------
    # 3. Verify Artifacts
    # -------------------------------------------------------------------------
    print("\n--- Verifying Generated Artifacts ---")

    expected_files = [
        "models/scout_lgbm.joblib",
        "models/scout_xgb.joblib",
        "models/expert_lgbm.joblib",
        "models/expert_lgbm.joblib",
        "models/best_threshold.npy",
        "hard_negative_indices.npy",
        "features_train_sample_5000.parquet",
        "features_val_sample_5000.parquet",
    ]

    for fname in expected_files:
        path = os.path.join(Config.CACHE_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Pipeline failed to generate expected file: {fname}"
            )
        print(f"Verified: {fname}")

    # Check threshold value validity
    thresh_path = os.path.join(Config.CACHE_DIR, "models/best_threshold.npy")
    best_threshold = load_npy("models/best_threshold.npy")[0]

    if not (0.0 < best_threshold < 1.0):
        raise ValueError(f"Invalid threshold optimized: {best_threshold}")
    print(f"Optimized Threshold: {best_threshold:.4f}")

    # -------------------------------------------------------------------------
    # 4. Inference Demonstration
    # -------------------------------------------------------------------------
    print("\n--- Starting Inference Demo ---")

    # Generate features for Test set (using a small sample)
    # Note: In a real submission, we would process the whole file.
    # Here we sample 2000 rows from the test metadata.
    print("Generating Test Features...")
    df_test_features = FeatureEngine.generate_features(
        split="test", load_cached_data=False, sample_size=2000
    )

    if df_test_features.empty:
        raise ValueError("Test feature generation returned empty DataFrame.")

    print(f"Test Features Shape: {df_test_features.shape}")

    # Prepare input features
    # Filter columns to match training features
    exclude_cols = [
        "contact_id",
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "contact",
        "datetime",
        "step_offset",
    ]
    feature_cols = [c for c in df_test_features.columns if c not in exclude_cols]

    X_test = df_test_features[feature_cols]

    # Load Ensemble
    print("Loading Models...")
    predictor = EnsemblePredictor()
    predictor.load_models(
        lgbm_path="models/expert_lgbm.joblib", xgb_path="models/expert_xgb.joblib"
    )

    # Predict
    print("Predicting...")
    probs = predictor.predict(X_test)

    # Apply Threshold
    preds = (probs >= best_threshold).astype(int)

    # Create Submission DataFrame
    submission = pd.DataFrame(
        {"contact_id": df_test_features["contact_id"], "contact": preds}
    )

    # Basic Validation of Submission
    assert len(submission) == len(df_test_features), "Submission length mismatch"
    assert submission["contact"].isin([0, 1]).all(), "Invalid prediction values"

    # Save Submission
    sub_path = os.path.join(Config.CACHE_DIR, "mini_sample_submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")
    print("\nTop 5 Predictions:")
    print(submission.head())

    # -------------------------------------------------------------------------
    # 5. Cleanup
    # -------------------------------------------------------------------------
    print("\n--- Demo Completed Successfully ---")
    # We leave the files in ./working/demo_run/ for inspection as per instructions
    # to use the working directory.


if __name__ == "__main__":
    main()
