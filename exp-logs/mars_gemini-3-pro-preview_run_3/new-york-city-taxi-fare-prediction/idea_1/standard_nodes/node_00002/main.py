import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Import provided library modules
from library import config
from library.trainer import train_model


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    # Suppress warnings for clean output
    warnings.filterwarnings("ignore")

    # Set seeds for reproducibility
    np.random.seed(config.SEED)

    # Define parameters
    # Scaling up to 10M samples (Cite solution_lesson_node_00001) for better performance
    TRAIN_SAMPLE_SIZE = 10_000_000
    MAX_ITER = 1000
    BASELINE_RMSE = 19.075552126316484

    # -------------------------------------------------------------------------
    # 2. Pipeline Execution (Train -> Predict -> Submit)
    # -------------------------------------------------------------------------
    # train_model orchestrates data processing and training.
    # It returns the trained regressor object and the processed test dataframe.
    regressor, test_df = train_model(
        load_cached_data=True, train_sample_size=TRAIN_SAMPLE_SIZE, max_iter=MAX_ITER
    )

    # -------------------------------------------------------------------------
    # 3. Validation Assessment
    # -------------------------------------------------------------------------
    # Load the processed validation set from the cache created by the data processor.
    # The processor saves files to the configured CACHE_DIR.
    val_cache_path = os.path.join(config.CACHE_DIR, "val_processed.parquet")

    if not os.path.exists(val_cache_path):
        raise FileNotFoundError(
            f"Validation cache not found at {val_cache_path}. Ensure data processing completed successfully."
        )

    # Load validation data
    val_df = pd.read_parquet(val_cache_path)

    # Generate predictions on the validation set
    # The regressor class handles feature selection internally based on self.feature_names
    val_preds = regressor.predict(val_df)
    y_val = val_df["fare_amount"]

    # Calculate RMSE
    mse = mean_squared_error(y_val, val_preds)
    rmse = np.sqrt(mse)

    # REQUIRED OUTPUT: Final Validation Metric
    print(f"Final Validation Metric: {rmse}")

    # Conditional Submission
    if rmse < BASELINE_RMSE:
        print(
            f"RMSE {rmse:.4f} is better than baseline {BASELINE_RMSE:.4f}. Generating submission..."
        )
        predictions = regressor.predict(test_df)
        regressor.save_submission(test_df, predictions)
    else:
        print(
            f"RMSE {rmse:.4f} did not improve baseline {BASELINE_RMSE:.4f}. Skipping submission."
        )

    # -------------------------------------------------------------------------
    # 4. Failure Analysis
    # -------------------------------------------------------------------------
    print("\nFailure Analysis (Correlation with Absolute Error):")

    # Calculate Absolute Error (Residuals)
    val_df["abs_error"] = np.abs(y_val - val_preds)

    # Analyze correlation with features used in the model
    features = regressor.feature_names
    correlations = {}

    for feat in features:
        if feat in val_df.columns:
            # Calculate Pearson correlation between the feature and the error magnitude
            corr = val_df[feat].corr(val_df["abs_error"])
            correlations[feat] = corr

    # Sort by absolute correlation strength to highlight most impactful features
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    for feat, corr in sorted_corrs:
        print(f"{feat}: {corr:.4f}")


if __name__ == "__main__":
    main()
