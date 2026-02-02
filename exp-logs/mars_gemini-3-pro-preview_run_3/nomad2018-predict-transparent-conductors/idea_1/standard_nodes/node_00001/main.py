import os
import numpy as np
import pandas as pd
from library.config import Config
from library.data_manager import DataManager
from library.feature_engineer import FeaturePipeline
from library.model_trainer import RidgeRegressorWrapper
from library.utils import calculate_rmsle


def main():
    print("Starting End-to-End Pipeline...")

    # 1. Initialize Managers
    data_manager = DataManager()
    feature_pipeline = FeaturePipeline()
    model_wrapper = RidgeRegressorWrapper()

    # 2. Load Data
    # We use the full dataset (sample_size=None) for the final run.
    # load_cached_data=True allows using previously computed geometry features if available.
    print("\n--- Loading Data ---")
    df_train_raw = data_manager.load_train_data(sample_size=Config.DEBUG_SAMPLE_SIZE)
    df_val_raw = data_manager.load_val_data(sample_size=Config.DEBUG_SAMPLE_SIZE)
    df_test_raw = data_manager.load_test_data(sample_size=Config.DEBUG_SAMPLE_SIZE)

    # 3. Feature Engineering
    print("\n--- Processing Features ---")
    # Fit and transform on Train
    df_train_processed = feature_pipeline.process_and_cache(
        df_train_raw, Config.TRAIN_FEATURES_CACHE, is_training=True
    )

    # Transform Val and Test using the fitted pipeline
    df_val_processed = feature_pipeline.process_and_cache(
        df_val_raw, Config.VAL_FEATURES_CACHE, is_training=False
    )

    df_test_processed = feature_pipeline.process_and_cache(
        df_test_raw, Config.TEST_FEATURES_CACHE, is_training=False
    )

    # 4. Prepare Data for Training
    # Exclude ID and Targets from X
    feature_cols = [
        c
        for c in df_train_processed.columns
        if c not in Config.TARGET_COLS and c != Config.ID_COL
    ]

    X_train = df_train_processed[feature_cols].values
    y_train = df_train_processed[Config.TARGET_COLS].values

    X_val = df_val_processed[feature_cols].values
    y_val = df_val_processed[Config.TARGET_COLS].values

    X_test = df_test_processed[feature_cols].values
    ids_test = df_test_processed[Config.ID_COL].values

    # 5. Train Models
    print("\n--- Training Models ---")
    model_wrapper.train(X_train, y_train)

    # 6. Validation & Metric
    print("\n--- Validation ---")
    # The wrapper's evaluate method prints detailed metrics
    metrics = model_wrapper.evaluate(X_val, y_val)

    # Print the final metric in the required format
    print(f"Final Validation Metric: {metrics['rmsle_mean']}")

    # 7. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Get predictions for validation set
    y_val_pred = model_wrapper.predict(X_val)

    # Calculate error magnitude in log space (since metric is RMSLE)
    # error = |log(1+true) - log(1+pred)|
    log_true = np.log1p(y_val)
    log_pred = np.log1p(y_val_pred)
    error_matrix = np.abs(log_true - log_pred)

    # Create a DataFrame for correlation analysis
    # We correlate errors with the ORIGINAL raw numerical features to see physical dependencies
    analysis_df = df_val_raw[Config.NUM_COLS + Config.GEO_COLS].copy()

    # Handle NaNs in raw geometry features if any (simple fill for correlation check)
    analysis_df = analysis_df.fillna(0)

    # Add errors
    analysis_df["error_formation"] = error_matrix[:, 0]
    analysis_df["error_bandgap"] = error_matrix[:, 1]

    # Compute correlations
    corr = analysis_df.corr()

    print("Top correlations with Formation Energy Error:")
    print(
        corr["error_formation"]
        .drop(["error_formation", "error_bandgap"])
        .sort_values(key=abs, ascending=False)
        .head(5)
    )

    print("\nTop correlations with Bandgap Energy Error:")
    print(
        corr["error_bandgap"]
        .drop(["error_formation", "error_bandgap"])
        .sort_values(key=abs, ascending=False)
        .head(5)
    )

    # 8. Generate Submission
    print("\n--- Generating Submission ---")
    y_test_pred = model_wrapper.predict(X_test)

    submission_df = pd.DataFrame(
        {
            Config.ID_COL: ids_test,
            Config.TARGET_COLS[0]: y_test_pred[:, 0],
            Config.TARGET_COLS[1]: y_test_pred[:, 1],
        }
    )

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())


if __name__ == "__main__":
    main()
