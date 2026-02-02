import os
import sys
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Import from library
from library.config import Config
from library.data import load_and_featurize_data, TargetTransformer
from library.model import DualTargetRegressor


def main():
    # 1. Setup and GPU detection
    print("--- Setting up ---")
    # Check for GPU
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("GPU detected. Configuring XGBoost for GPU.")
    else:
        print("No GPU detected. Using CPU.")

    # 2. Load Data
    # We use the provided function which handles caching
    # The feature extraction includes Physical, RDF, and Steinhardt features
    train_df, val_df, test_df = load_and_featurize_data(
        debug_sample=Config.DEBUG_SAMPLE_SIZE, load_cached_data=True
    )

    # 3. Initialize Model
    regressor = DualTargetRegressor()

    # Update params for GPU if available to optimize speed
    if use_gpu:
        # Set device to cuda for XGBoost
        regressor.params["device"] = "cuda"
        # Ensure using histogram method which is efficient on GPU
        regressor.params["tree_method"] = "hist"

    # 4. Train
    print("--- Training Models ---")
    # Fits XGBoost models for both formation energy and bandgap
    regressor.fit(train_df, val_df)

    # 5. Validation Assessment
    print("--- Validating ---")

    # Generate predictions on the validation set
    val_preds_df = regressor.predict(val_df)

    # Align predictions with ground truth using 'id'
    val_merged = pd.merge(
        val_df[["id"] + Config.TARGET_COLS],
        val_preds_df,
        on="id",
        suffixes=("_true", "_pred"),
    )

    rmsles = []
    transformer = TargetTransformer()

    print("\nValidation Metrics per Target:")
    for target in Config.TARGET_COLS:
        y_true = val_merged[f"{target}_true"].values
        y_pred = val_merged[f"{target}_pred"].values

        # Ensure non-negative predictions for log transform (physics constraint)
        y_pred = np.maximum(y_pred, 0)

        # Transform to log space: z = log(1 + y)
        z_true = transformer.transform(y_true)
        z_pred = transformer.transform(y_pred)

        # Calculate RMSE in log space, which is equivalent to RMSLE in original space
        mse = mean_squared_error(z_true, z_pred)
        rmsle = np.sqrt(mse)
        rmsles.append(rmsle)

        print(f"  {target}: RMSLE = {rmsle:.6f}")

    # The final metric is the column-wise mean of RMSLEs
    final_metric = np.mean(rmsles)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Identify which features correlate most with model error

    # Get the features used by the model
    feature_cols = regressor.feature_cols
    X_val = val_df[feature_cols].copy()

    for target in Config.TARGET_COLS:
        y_true = val_merged[f"{target}_true"].values
        y_pred = val_merged[f"{target}_pred"].values

        # Calculate absolute error in log space (since we optimize for RMSLE)
        z_true = transformer.transform(y_true)
        z_pred = transformer.transform(np.maximum(y_pred, 0))
        errors = np.abs(z_true - z_pred)

        # Add error to a temporary dataframe to compute correlations
        analysis_df = X_val.copy()
        analysis_df["error"] = errors

        # Compute correlation of features with the error
        # We drop the error column itself from the result
        corrs = (
            analysis_df.corrwith(analysis_df["error"])
            .abs()
            .sort_values(ascending=False)
        )

        print(f"\nTop 5 features correlated with error for {target}:")
        print(corrs.drop("error").head(5))

    # 7. Submission
    # Threshold defined in requirements
    THRESHOLD = 0.056919346405286564

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")

        # Generate predictions for test set
        test_preds_df = regressor.predict(test_df)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        # Save submission file
        test_preds_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric {final_metric} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
