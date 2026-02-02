import os
import numpy as np
import pandas as pd
import warnings
import shutil

# Import library components
from library.config import Config
from library.data_processing import DataLoader
from library.feature_engineering import FeatureEngine
from library.models import UnifiedEnsemble
from library.evaluation import ThresholdOptimizer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def run_demo():
    print("Starting Library Usage Demo...")

    # --------------------------------------------------------------------------
    # 1. Configuration Setup
    # --------------------------------------------------------------------------
    # Override Config parameters for speed
    print("Configuring parameters for fast execution...")
    Config.N_ESTIMATORS = 10
    Config.EARLY_STOPPING_ROUNDS = 5
    Config.WINDOW_SIZE = 1  # Small window to reduce feature count
    Config.USE_TOPOLOGY = False  # Disable graph metrics for speed in demo
    Config.WORKING_DIR = "./working/demo_run"

    # Ensure clean working directory
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set global seed
    np.random.seed(Config.SEED)

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    loader = DataLoader()

    # Load a small sample of metadata to ensure quick processing
    # We use 200 rows to demonstrate the pipeline without long waits
    df_meta = loader.load_metadata(split="train", sample_size=200)
    assert len(df_meta) == 200, "Metadata sampling failed"

    # Load tracking data (full file, filtering happens later)
    df_tracking = loader.load_tracking(split="train")
    assert not df_tracking.empty, "Tracking data is empty"

    # Merge metadata with tracking data
    # We disable cache loading to demonstrate the merge logic
    print("Merging metadata and tracking data...")
    df_merged = loader.merge_tracking_data(
        df_meta, df_tracking, split_name="demo_train", load_cached_data=False
    )

    # Verify merge results
    expected_p1_col = "x_position_p1"
    assert expected_p1_col in df_merged.columns, "Merged data missing player 1 features"
    assert len(df_merged) == 200, "Merged data row count mismatch"

    # --------------------------------------------------------------------------
    # 3. Feature Engineering
    # --------------------------------------------------------------------------
    engine = FeatureEngine()

    print("Generating features...")
    # Generate features (Physics derivatives + Temporal windows)
    df_features = engine.generate_features(
        df_merged, df_tracking, split_name="demo_train", load_cached_data=False
    )

    # Verify feature generation
    # Check for a lag feature created by windowing
    lag_col = "speed_p1_lag_1"
    assert lag_col in df_features.columns, "Temporal features not generated"
    assert "contact" in df_features.columns, "Target column missing from features"

    # --------------------------------------------------------------------------
    # 4. Model Training (Unified Ensemble)
    # --------------------------------------------------------------------------
    print("Training Unified Ensemble (LGBM + XGB)...")

    # Split into mock train/val sets (80/20)
    split_idx = int(len(df_features) * 0.8)
    train_df = df_features.iloc[:split_idx].copy()
    val_df = df_features.iloc[split_idx:].copy()

    # Ensure we have both classes in train/val for the demo to work without errors
    # If random sampling resulted in only one class, we force a dummy label for demo purposes
    if train_df["contact"].nunique() < 2:
        train_df.iloc[0, train_df.columns.get_loc("contact")] = 1
        train_df.iloc[1, train_df.columns.get_loc("contact")] = 0

    ensemble = UnifiedEnsemble()
    ensemble.train(train_df, val_df, target_col="contact")

    # Save models
    model_save_path = os.path.join(Config.WORKING_DIR, "models")
    ensemble.save(model_save_path)

    # Verify files created
    assert os.path.exists(os.path.join(model_save_path, "lgbm_model.joblib"))
    assert os.path.exists(os.path.join(model_save_path, "xgb_model.joblib"))

    # Reload models to demonstrate loading capability
    ensemble_loaded = UnifiedEnsemble()
    ensemble_loaded.load(model_save_path)

    # --------------------------------------------------------------------------
    # 5. Evaluation & Inference
    # --------------------------------------------------------------------------
    print("Running Inference and Optimization...")

    # Predict probabilities
    probs = ensemble_loaded.predict_proba(val_df)

    assert len(probs) == len(val_df), "Prediction length mismatch"
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities out of bounds"

    # Optimize Threshold
    optimizer = ThresholdOptimizer(start=0.1, end=0.9, step=0.05)
    best_thresh, best_mcc = optimizer.optimize(val_df["contact"], probs)

    print(f"\nDemo Completed Successfully.")
    print(f"Final MCC on Demo Val Set: {best_mcc:.4f}")
    print(f"Optimal Threshold: {best_thresh:.2f}")


if __name__ == "__main__":
    run_demo()
