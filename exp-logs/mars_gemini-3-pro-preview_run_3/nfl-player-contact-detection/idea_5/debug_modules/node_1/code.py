import os
import sys
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# 1. Import and Patch Configuration for Speed
from library.config import Config

# Patch Config for rapid execution
Config.DEBUG = True
Config.XGB_PARAMS["n_estimators"] = 10  # Very few trees for demo
Config.XGB_PARAMS["early_stopping_rounds"] = 5
Config.MICRO_WINDOW_SIZE = 1  # Small window to reduce feature calc time
Config.MACRO_WINDOW_SIZE = 3
Config.WORKING_DIR = "./working/demo_run"
os.makedirs(Config.WORKING_DIR, exist_ok=True)

# Update paths in Config to point to the new working dir for artifacts
Config.CACHE_TRAIN_A_X = os.path.join(Config.WORKING_DIR, "train_streamA_X.parquet")
Config.CACHE_TRAIN_A_Y = os.path.join(Config.WORKING_DIR, "train_streamA_y.npy")
Config.CACHE_VAL_A_X = os.path.join(Config.WORKING_DIR, "val_streamA_X.parquet")
Config.CACHE_VAL_A_Y = os.path.join(Config.WORKING_DIR, "val_streamA_y.npy")
Config.CACHE_TRAIN_B_X = os.path.join(Config.WORKING_DIR, "train_streamB_X.parquet")
Config.CACHE_TRAIN_B_Y = os.path.join(Config.WORKING_DIR, "train_streamB_y.npy")
Config.CACHE_VAL_B_X = os.path.join(Config.WORKING_DIR, "val_streamB_X.parquet")
Config.CACHE_VAL_B_Y = os.path.join(Config.WORKING_DIR, "val_streamB_y.npy")

# Import library components after patching Config
from library.utils import seed_everything
from library.data_manager import DataManager
from library.model_trainer import DualStreamPredictor
from library.optimizer import ThresholdOptimizer


def run_demo():
    print("Starting End-to-End Demo...")
    seed_everything(42)

    # ==========================================
    # 1. Data Preparation
    # ==========================================
    print("\n[1] Initializing DataManager (Debug Mode)...")
    dm = DataManager(debug=True)

    # --- Stream A: Player-Player ---
    print("Loading Stream A (Player-Player) data...")
    # load_cached=False ensures we exercise the feature engineering logic
    X_train_a, y_train_a, X_val_a, y_val_a = dm.prepare_stream_datasets(
        stream_type="A", load_cached=False
    )

    # Validation assertions
    assert not X_train_a.empty, "Stream A training features should not be empty"
    assert len(X_train_a) == len(y_train_a), "Stream A train features/labels mismatch"
    assert not X_val_a.empty, "Stream A validation features should not be empty"
    print(f"Stream A Train Shape: {X_train_a.shape}, Val Shape: {X_val_a.shape}")

    # --- Stream B: Player-Ground ---
    print("Loading Stream B (Player-Ground) data...")
    X_train_b, y_train_b, X_val_b, y_val_b = dm.prepare_stream_datasets(
        stream_type="B", load_cached=False
    )

    # Validation assertions
    assert not X_train_b.empty, "Stream B training features should not be empty"
    assert len(X_train_b) == len(y_train_b), "Stream B train features/labels mismatch"
    print(f"Stream B Train Shape: {X_train_b.shape}, Val Shape: {X_val_b.shape}")

    # ==========================================
    # 2. Model Training
    # ==========================================
    print("\n[2] Training DualStreamPredictor...")
    predictor = DualStreamPredictor()

    # Patch predictor paths to use our demo working directory
    predictor.model_paths["A"] = os.path.join(Config.WORKING_DIR, "model_a.json")
    predictor.model_paths["B"] = os.path.join(Config.WORKING_DIR, "model_b.json")
    predictor.threshold_paths["A"] = os.path.join(Config.WORKING_DIR, "thresh_a.joblib")
    predictor.threshold_paths["B"] = os.path.join(Config.WORKING_DIR, "thresh_b.joblib")

    # Train Stream A
    predictor.train_stream(X_train_a, y_train_a, X_val_a, y_val_a, "A")
    assert os.path.exists(predictor.model_paths["A"]), "Stream A model file not saved"

    # Train Stream B
    predictor.train_stream(X_train_b, y_train_b, X_val_b, y_val_b, "B")
    assert os.path.exists(predictor.model_paths["B"]), "Stream B model file not saved"

    # ==========================================
    # 3. Threshold Optimization
    # ==========================================
    print("\n[3] Optimizing Thresholds...")

    # Generate probabilities for validation sets
    # Note: In a real scenario, we might use out-of-fold predictions,
    # but here we use the validation set used for early stopping for demonstration.
    probs_a = predictor.models["A"].predict_proba(X_val_a)[:, 1]
    probs_b = predictor.models["B"].predict_proba(X_val_b)[:, 1]

    optimizer = ThresholdOptimizer(steps=20)  # Low steps for speed
    thresh_a, thresh_b, combined_mcc = optimizer.optimize_thresholds(
        y_val_a, probs_a, y_val_b, probs_b
    )

    # Manually update predictor thresholds with the globally optimized ones
    # (though independent optimization in train_stream does this, this step verifies the optimizer class)
    predictor.thresholds["A"] = thresh_a
    predictor.thresholds["B"] = thresh_b

    assert 0.0 < thresh_a < 1.0, "Threshold A out of bounds"
    assert 0.0 < thresh_b < 1.0, "Threshold B out of bounds"
    print(f"Optimized Thresholds -> A: {thresh_a:.4f}, B: {thresh_b:.4f}")

    # ==========================================
    # 4. Inference
    # ==========================================
    print("\n[4] Running Inference on Test Set...")

    # Stream A Inference
    X_test_a, ids_a = dm.get_test_data("A", load_cached=False)
    if not X_test_a.empty:
        preds_a = predictor.predict(X_test_a, "A")
        df_res_a = pd.DataFrame({"contact_id": ids_a, "contact": preds_a})
    else:
        df_res_a = pd.DataFrame(columns=["contact_id", "contact"])

    # Stream B Inference
    X_test_b, ids_b = dm.get_test_data("B", load_cached=False)
    if not X_test_b.empty:
        preds_b = predictor.predict(X_test_b, "B")
        df_res_b = pd.DataFrame({"contact_id": ids_b, "contact": preds_b})
    else:
        df_res_b = pd.DataFrame(columns=["contact_id", "contact"])

    # Combine
    submission = pd.concat([df_res_a, df_res_b], ignore_index=True)

    # Fill any missing contact_ids from sample_submission if necessary (not strictly needed if test_meta is complete)
    # But for this demo, we just save what we predicted.

    # Save Submission
    sub_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)

    print(f"Submission saved to {sub_path}")
    print(f"Submission shape: {submission.shape}")
    print(submission.head())

    # Validation assertions
    assert os.path.exists(sub_path), "Submission file was not created"
    assert "contact_id" in submission.columns, "contact_id column missing"
    assert "contact" in submission.columns, "contact column missing"
    assert submission["contact"].isin([0, 1]).all(), "Predictions must be binary"

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
