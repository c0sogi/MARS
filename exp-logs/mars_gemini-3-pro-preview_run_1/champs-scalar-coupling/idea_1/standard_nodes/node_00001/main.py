import os
import numpy as np
import pandas as pd
import warnings
from library.config import Config
from library.data_loader import get_train_val_data, get_test_data
from library.model import CouplingPredictor
from library.utils import calculate_log_mae, save_submission
from library.features import COUPLING_TYPES

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    print("Initializing pipeline...")
    set_seed(Config.RANDOM_SEED)

    # 2. Load Data
    print("Loading training and validation data...")
    # Using cached data if available for speed
    X_train, y_train, X_val, y_val = get_train_val_data(load_cached_data=True)

    # Fast Baseline Optimization: Limit training samples
    # The dataset is large (>3M rows). For a fast baseline, we subsample.
    MAX_TRAIN_SAMPLES = 500000
    if len(X_train) > MAX_TRAIN_SAMPLES:
        print(
            f"Subsampling training data from {len(X_train)} to {MAX_TRAIN_SAMPLES} for fast baseline..."
        )
        # Use numpy choice for random indices
        indices = np.random.choice(len(X_train), MAX_TRAIN_SAMPLES, replace=False)
        X_train_sub = X_train.iloc[indices].copy()
        y_train_sub = y_train.iloc[indices].copy()
    else:
        X_train_sub = X_train
        y_train_sub = y_train

    print(f"Training shape: {X_train_sub.shape}")
    print(f"Validation shape: {X_val.shape}")

    # 3. Model Training
    print("\nInitializing and training model...")
    predictor = CouplingPredictor()

    # Fit the model
    # Note: The predictor handles moving data to GPU if configured in Config.XGB_PARAMS
    predictor.fit(X_train_sub, y_train_sub, X_val, y_val)

    # 4. Validation Assessment
    print("\nPerforming final validation assessment...")
    # Predict on validation set
    val_preds = predictor.predict(X_val)

    # Prepare DataFrame for metric calculation
    # We need to reconstruct the original 'type' string column from 'type_enc'
    val_eval_df = pd.DataFrame({Config.TARGET_COL: y_val.values, "pred": val_preds})

    # Reconstruct 'type' column
    if Config.TYPE_ENC_COL in X_val.columns:
        type_decoder = {i: t for i, t in enumerate(COUPLING_TYPES)}
        val_eval_df[Config.TYPE_COL] = (
            X_val[Config.TYPE_ENC_COL].astype(int).map(type_decoder)
        )
    else:
        raise ValueError(
            "Cannot calculate metric: 'type_enc' column missing in validation features."
        )

    # Calculate metric
    # We set verbose=False here to avoid duplicate printing, as we need a specific format below
    metric_score = calculate_log_mae(
        val_eval_df,
        pred_col="pred",
        target_col=Config.TARGET_COL,
        type_col=Config.TYPE_COL,
        verbose=False,
    )

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {metric_score}")

    # 5. Failure Analysis
    print("\nPerforming failure analysis...")
    # Calculate absolute error
    val_eval_df["abs_error"] = (
        val_eval_df[Config.TARGET_COL] - val_eval_df["pred"]
    ).abs()

    # Combine error with features for correlation analysis
    # We use X_val indices to align
    analysis_df = X_val.copy()
    analysis_df["abs_error"] = val_eval_df["abs_error"].values

    # Calculate correlation between features and error
    correlations = analysis_df.corrwith(analysis_df["abs_error"]).sort_values(
        ascending=False
    )

    print("Correlation between Input Features and Absolute Error:")
    # Drop the self-correlation of abs_error
    print(correlations.drop("abs_error"))

    # 6. Submission Generation
    print("\nGenerating submission...")
    print("Loading test data...")
    X_test, ids = get_test_data(load_cached_data=True)

    print(f"Predicting on test set ({len(X_test)} samples)...")
    test_preds = predictor.predict(X_test)

    print("Saving submission file...")
    save_submission(ids.values, test_preds)

    print("Run complete.")


if __name__ == "__main__":
    main()
