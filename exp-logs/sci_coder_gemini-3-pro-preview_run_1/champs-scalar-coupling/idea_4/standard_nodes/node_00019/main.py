import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

# Import provided library modules
import library.config as config
from library.trainer import StratifiedTrainer
from library.utils import seed_everything, Timer


def run_pcse_pipeline():
    # 1. Setup & Configuration
    # Ensure reproducibility
    seed_everything(config.RANDOM_STATE)

    # 2. Training
    # Initialize the stratified trainer
    trainer = StratifiedTrainer()

    # Train the ensemble
    # We use load_cached_data=True to leverage any existing parquet files in ./working
    print("Starting Training Pipeline...")
    with Timer("Training Phase"):
        trainer.train(load_cached_data=True)

    # 3. Validation & Failure Analysis
    print("\n" + "=" * 30)
    print("VALIDATION & FAILURE ANALYSIS")
    print("=" * 30)

    # Load validation metadata
    val_meta = pd.read_csv(config.VAL_METADATA_PATH)

    # Generate features for validation set
    # The pipeline handles caching internally
    print("Generating validation features...")
    df_val = trainer.pipeline.generate_features(val_meta, "val", load_cached_data=True)

    log_maes = []
    validation_data_buffer = []  # To store data for failure analysis

    print("Evaluating Stratified Models...")
    for c_type in config.COUPLING_TYPES:
        # Prepare data for this specific coupling type
        X_val, y_val = trainer.pipeline.prepare_data_for_type(df_val, c_type)

        if X_val.empty:
            continue

        # Retrieve the trained model
        if c_type in trainer.models:
            model = trainer.models[c_type]
        else:
            # Load from disk if not in memory (e.g. if training was skipped due to cache)
            model_path = os.path.join(trainer.model_dir, f"xgb_{c_type}.json")
            if os.path.exists(model_path):
                model = xgb.XGBRegressor()
                model.load_model(model_path)
            else:
                print(f"Warning: No model found for {c_type}")
                continue

        # Inference
        # XGBoost handles device placement automatically based on config (cuda)
        preds = model.predict(X_val)

        # Calculate Metric Component: Log(MAE)
        mae = mean_absolute_error(y_val, preds)
        log_mae = np.log(mae)
        log_maes.append(log_mae)

        # Store for Failure Analysis
        # We attach errors to the feature set
        analysis_chunk = X_val.copy()
        analysis_chunk["abs_error"] = np.abs(y_val - preds)
        analysis_chunk["coupling_type"] = c_type
        validation_data_buffer.append(analysis_chunk)

    # Calculate Final Metric
    if log_maes:
        final_metric = np.mean(log_maes)
    else:
        final_metric = 0.0

    # REQUIRED OUTPUT: Print Final Validation Metric
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation
    print("\nFailure Analysis: Feature Correlations with Error Magnitude")
    if validation_data_buffer:
        # Combine all validation chunks
        # Note: Different types have different columns dropped. We align them (filling NaNs)
        full_val_analysis = pd.concat(validation_data_buffer, ignore_index=True)

        # Select numeric columns only
        numeric_cols = full_val_analysis.select_dtypes(include=[np.number]).columns

        # Calculate correlation with absolute error
        correlations = (
            full_val_analysis[numeric_cols]
            .corrwith(full_val_analysis["abs_error"])
            .abs()
        )

        # Sort and display top correlations
        print(correlations.sort_values(ascending=False).head(10))
    else:
        print("No validation data available for analysis.")

    # 4. Submission
    # Threshold defined in task
    THRESHOLD = -0.7386035268505905

    print("\n" + "=" * 30)
    print("SUBMISSION CHECK")
    print("=" * 30)

    if final_metric < THRESHOLD:
        print(f"Metric {final_metric} is better than threshold {THRESHOLD}.")
        print("Generating submission...")

        with Timer("Inference Phase"):
            trainer.predict(load_cached_data=True)

    else:
        print(f"Metric {final_metric} did not meet threshold {THRESHOLD}.")
        print("Skipping submission generation.")


if __name__ == "__main__":
    run_pcse_pipeline()
