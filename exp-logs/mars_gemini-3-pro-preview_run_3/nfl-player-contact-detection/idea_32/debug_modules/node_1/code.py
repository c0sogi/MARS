import os
import sys
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.feature_engineering import FeatureEngineer
from library.model_trainer import ModelTrainer
from library.metric_optimizer import MetricOptimizer


def run_demo():
    print("=== Starting Pipeline Demo ===")

    # 1. Setup & Configuration Override for Speed
    print("\n[1] Configuring environment...")
    Config.setup()

    # Override XGBoost parameters to ensure quick execution
    # Reducing n_estimators and early_stopping_rounds significantly
    Config.XGB_PARAMS_STREAM_A["n_estimators"] = 10
    Config.XGB_PARAMS_STREAM_A["early_stopping_rounds"] = 2
    Config.XGB_PARAMS_STREAM_B["n_estimators"] = 10
    Config.XGB_PARAMS_STREAM_B["early_stopping_rounds"] = 2

    # Set random seed
    np.random.seed(Config.SEED)

    # 2. Feature Engineering (Debug Mode)
    print("\n[2] Running Feature Engineering (Debug Mode)...")
    fe = FeatureEngineer()

    # Process Train, Validation, and Test sets
    # debug=True limits processing to the first 5 plays in the metadata
    print("  -> Processing Train Data...")
    train_data = fe.process_data(mode="train", load_cached_data=False, debug=True)

    print("  -> Processing Validation Data...")
    val_data = fe.process_data(mode="validation", load_cached_data=False, debug=True)

    print("  -> Processing Test Data...")
    test_data = fe.process_data(mode="test", load_cached_data=False, debug=True)

    # Verification: Check Data Structure
    for stream in ["stream_a", "stream_b"]:
        assert stream in train_data, f"Missing {stream} in train_data"
        assert "X" in train_data[stream], f"Missing X in train_data[{stream}]"
        assert "y" in train_data[stream], f"Missing y in train_data[{stream}]"
        assert "ids" in train_data[stream], f"Missing ids in train_data[{stream}]"

        # Check that we have rows (unless the slice happened to be empty for a specific stream, which is unlikely for top 5 plays)
        if len(train_data[stream]["X"]) > 0:
            assert (
                train_data[stream]["X"].shape[0] == train_data[stream]["y"].shape[0]
            ), f"Mismatch in X and y shapes for {stream}"

    print("  -> Feature Engineering Output Verified.")

    # 3. Model Training
    print("\n[3] Training Models...")
    trainer = ModelTrainer()

    # Train using the debug datasets
    # force_retrain=True ensures we don't load old cached models
    models = trainer.train(train_data, val_data, force_retrain=True)

    # Verification: Check Models
    assert "stream_a" in models, "Stream A model missing"
    assert "stream_b" in models, "Stream B model missing"

    # It is possible for a model to be None if no training data was found in the debug slice,
    # but with 5 plays, we expect data.
    if models["stream_a"] is not None:
        print("  -> Stream A Model trained successfully.")
    else:
        print("  -> Warning: Stream A Model is None (insufficient debug data).")

    if models["stream_b"] is not None:
        print("  -> Stream B Model trained successfully.")
    else:
        print("  -> Warning: Stream B Model is None (insufficient debug data).")

    # 4. Threshold Optimization
    print("\n[4] Optimizing Thresholds...")
    optimizer = MetricOptimizer()

    # Optimize thresholds on validation data
    thresholds = optimizer.optimize_thresholds(models, val_data)

    print(f"  -> Optimized Thresholds: {thresholds}")
    assert "stream_a" in thresholds
    assert "stream_b" in thresholds
    assert 0.0 < thresholds["stream_a"] < 1.0, "Threshold A out of bounds"

    # 5. Prediction & Submission
    print("\n[5] Generating Submission...")

    # Generate raw probability predictions
    preds_df = trainer.predict(models, test_data)

    # Verify predictions dataframe
    assert not preds_df.empty, "Prediction DataFrame is empty"
    assert "contact_id" in preds_df.columns
    assert "score" in preds_df.columns

    # Apply thresholds and save submission
    optimizer.generate_submission(preds_df, thresholds)

    # Verify Submission File
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file not found"

    df_sub = pd.read_csv(submission_path)
    print(f"  -> Submission loaded. Shape: {df_sub.shape}")

    assert "contact_id" in df_sub.columns
    assert "contact" in df_sub.columns
    assert (
        df_sub["contact"].isin([0, 1]).all()
    ), "Contact column contains non-binary values"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
