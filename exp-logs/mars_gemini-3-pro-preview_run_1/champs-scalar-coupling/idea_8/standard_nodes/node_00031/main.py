import pandas as pd
import numpy as np
import os
import sys
import gc
import random
from sklearn.model_selection import train_test_split

# Import provided library modules
import library.config as config
from library.feature_engineering import get_data
from library.model_pipeline import StratifiedRegressor
from library.utils import calculate_log_mae, save_submission


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup and Configuration
    SEED = 42
    set_seed(SEED)

    print("Initializing Fast Baseline Run...")

    # Override XGBoost parameters for fast execution (Fast Baseline)
    # We reduce estimators and depth to ensure the run completes quickly within the task limits.
    # Note: In a full production run, we would use the high-capacity defaults in config.py.
    config.XGB_PARAMS["training"]["n_estimators"] = 1000
    config.XGB_PARAMS["common"]["max_depth"] = 8

    # 2. Load Data
    # We load cached data if available to save time on feature engineering
    print("Loading datasets...")
    X_train, y_train, X_val, y_val, X_test, ids_test = get_data(load_cached_data=True)

    # 3. Subsampling for Fast Baseline
    # We limit training data to 200,000 samples to verify the pipeline quickly.
    # We use stratified sampling to ensure all coupling types are represented.
    MAX_TRAIN_SAMPLES = 200000

    if len(X_train) > MAX_TRAIN_SAMPLES:
        print(
            f"Subsampling training set from {len(X_train)} to {MAX_TRAIN_SAMPLES} samples..."
        )
        X_train_sub, _, y_train_sub, _ = train_test_split(
            X_train,
            y_train,
            train_size=MAX_TRAIN_SAMPLES,
            stratify=X_train["type"],
            random_state=SEED,
        )
        # Force garbage collection to free memory
        del X_train, y_train
        gc.collect()

        X_train = X_train_sub.reset_index(drop=True)
        y_train = y_train_sub
    else:
        print("Training set size is within limits. Using full training set.")

    # 4. Model Training
    print("Starting Stratified Training...")
    regressor = StratifiedRegressor()
    regressor.fit(X_train, y_train, X_val, y_val)

    # 5. Validation
    print("Performing Validation Inference...")
    val_preds = regressor.predict(X_val)

    # Construct validation dataframe for metric calculation
    val_df = pd.DataFrame(
        {
            "type": X_val["type"].values,
            "scalar_coupling_constant": y_val,
            "prediction": val_preds,
        }
    )

    # Calculate and print the required metric
    # The function prints the breakdown and returns the final score
    metric = calculate_log_mae(val_df)
    print(f"Final Validation Metric: {metric}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate absolute error
    val_df["abs_error"] = (
        val_df["scalar_coupling_constant"] - val_df["prediction"]
    ).abs()

    # Correlate error with features
    # We select numeric features from X_val (excluding 'type' which is categorical)
    # We align X_val with val_df (indices should match as no shuffling occurred on X_val)
    analysis_cols = X_val.select_dtypes(include=[np.number]).columns.tolist()

    # Compute correlations efficiently
    # We create a temporary dataframe with just features and error
    corr_df = X_val[analysis_cols].copy()
    corr_df["abs_error"] = val_df["abs_error"].values

    # Calculate correlation with abs_error
    correlations = (
        corr_df.corr()["abs_error"].drop("abs_error").sort_values(ascending=False)
    )

    print("Top 5 Features correlated with Prediction Error:")
    print(correlations.head(5))

    # Clean up memory
    del corr_df, val_df
    gc.collect()

    # 7. Submission Generation
    # Only generate submission if the metric meets the strict threshold
    THRESHOLD = -1.1285111904144287

    if metric < THRESHOLD:
        print(
            f"\nMetric ({metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        test_preds = regressor.predict(X_test)
        save_submission(ids_test, test_preds)
    else:
        print(
            f"\nMetric ({metric}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
