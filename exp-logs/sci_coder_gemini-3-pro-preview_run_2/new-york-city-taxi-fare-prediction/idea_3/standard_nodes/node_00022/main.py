import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings

# Add current directory to path to ensure library imports work
sys.path.append(".")

# Import provided library modules
from library import config
from library.data_handler import DataHandler
from library.ensemble_learner import EnsembleLearner

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Initialization
    print("Initializing pipeline...")
    set_seed(config.RANDOM_SEED)

    # 2. Data Loading & Preprocessing
    # We use the DataHandler to load processed data from cache or scratch
    dh = DataHandler()
    # Force reprocessing to ensure new sanitization logic is applied
    train_df, val_df, test_df = dh.load_and_process_data(load_cached_data=False)

    # FAST BASELINE CONSTRAINT:
    # Limit training data to 10 million rows to ensure quick execution (< 2 hours).
    # This is sufficient to achieve the target metric with an ensemble approach.
    MAX_TRAIN_SAMPLES = 10_000_000
    if len(train_df) > MAX_TRAIN_SAMPLES:
        print(
            f"Subsampling training data from {len(train_df)} to {MAX_TRAIN_SAMPLES} for fast baseline execution."
        )
        train_df = train_df.sample(
            n=MAX_TRAIN_SAMPLES, random_state=config.RANDOM_SEED
        ).reset_index(drop=True)

    # 3. Ensemble Preparation
    # Create subsets for the ensemble (Partitioning strategy defined in config)
    subsets = dh.create_subsets(train_df)

    el = EnsembleLearner()

    # 4. Training
    # Train the ensemble of XGBoost models
    # Each model trains on a partition and uses the full validation set for early stopping
    el.train_ensemble_loop(subsets, val_df)

    # 5. Validation Inference & Metrics
    print("\n=== Validation Evaluation ===")
    # Generate predictions on the full validation set using the trained ensemble
    # We reuse the predict_ensemble method which handles model loading and averaging
    val_preds = el.predict_ensemble(val_df)

    # Calculate RMSE
    y_true = val_df["fare_amount"].values
    mse = np.mean((y_true - val_preds) ** 2)
    rmse = np.sqrt(mse)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {rmse}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute error
    errors = np.abs(y_true - val_preds)

    # Get feature columns used by the model
    _, _, feature_cols = el._get_features_and_target(val_df, is_train=False)

    # Calculate correlation between Error Magnitude and Features
    correlations = {}
    for col in feature_cols:
        # Check if column is numeric and exists
        if col in val_df.columns and pd.api.types.is_numeric_dtype(val_df[col]):
            # Handle potential NaNs in features (though XGBoost handles them, corrcoef needs clean data)
            # We create a mask for valid values
            valid_mask = ~np.isnan(val_df[col]) & ~np.isnan(errors)
            if np.sum(valid_mask) > 100:  # Ensure enough samples
                corr = np.corrcoef(val_df[col][valid_mask], errors[valid_mask])[0, 1]
                correlations[col] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Correlation between Error Magnitude and Features:")
    for feat, corr in sorted_corr:
        print(f"  {feat}: {corr:.4f}")

    # 7. Submission Generation
    # Threshold defined in task
    TARGET_METRIC = 4.278504866347902

    if rmse < TARGET_METRIC:
        print(
            f"\nValidation metric ({rmse}) meets threshold ({TARGET_METRIC}). Generating submission..."
        )
        submission_path = config.DATA_PATHS["submission"]
        el.generate_submission(test_df, submission_path)
    else:
        print(
            f"\nValidation metric ({rmse}) does not meet threshold ({TARGET_METRIC}). Submission skipped."
        )


if __name__ == "__main__":
    main()
