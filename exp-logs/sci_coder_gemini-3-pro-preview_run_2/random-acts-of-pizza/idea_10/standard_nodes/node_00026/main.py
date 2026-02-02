import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import load_dataset
from library.features import HybridFeaturePipeline
from library.model import BaggedLREnsemble


def main():
    # 1. Setup
    set_seed(Config.SEED)
    logger = setup_logger("runfile")

    # 2. Load Data
    logger.info("Loading datasets...")
    df_train, df_val, df_test = load_dataset(load_cached_data=True)

    # 3. Feature Engineering
    logger.info("Initializing and fitting feature pipeline...")
    pipeline = HybridFeaturePipeline()

    # Fit on training data only to prevent leakage
    pipeline.fit(df_train)

    # Transform all splits
    logger.info("Transforming datasets...")
    X_train = pipeline.transform(df_train)
    y_train = df_train[Config.TARGET_COL].values.astype(int)

    X_val = pipeline.transform(df_val)
    y_val = df_val[Config.TARGET_COL].values.astype(int)

    X_test = pipeline.transform(df_test)
    test_ids = df_test[Config.ID_COL].values

    # 4. Model Training (Train Split)
    logger.info("Training model on training split...")
    model = BaggedLREnsemble()
    model.optimize_and_fit(X_train, y_train)

    # 5. Validation
    logger.info("Validating model...")
    val_probs = model.predict_proba(X_val)
    val_auc = roc_auc_score(y_val, val_probs)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {val_auc:.15f}")

    # 6. Failure Analysis
    logger.info("Performing failure analysis...")
    # Calculate absolute errors (residuals)
    # y_val is 0 or 1, val_probs is [0, 1]. Error is distance from truth.
    errors = np.abs(y_val - val_probs)

    # Compute correlation between each feature and the error vector
    # We use vectorized numpy operations for speed over the high-dimensional feature space

    # Center the data
    X_val_centered = X_val - X_val.mean(axis=0)
    errors_centered = errors - errors.mean()

    # Compute Covariance: E[(X - E[X])(Y - E[Y])]
    # Note: X_val is (N, D), errors is (N,)
    # Dot product sums over N
    covariance = np.dot(errors_centered, X_val_centered) / (len(errors) - 1)

    # Compute Standard Deviations
    X_val_std = X_val.std(axis=0)
    errors_std = errors.std()

    # Compute Correlation: Cov(X, Y) / (Std(X) * Std(Y))
    # Handle division by zero for constant features
    with np.errstate(divide="ignore", invalid="ignore"):
        correlations = covariance / (X_val_std * errors_std)

    # Replace NaNs (from 0 std dev) with 0 correlation
    correlations = np.nan_to_num(correlations)

    # Get indices of top 5 features most positively correlated with error
    # (Features that, when high, tend to co-occur with high error)
    top_indices = np.argsort(np.abs(correlations))[-5:][::-1]

    print("\n--- Failure Analysis: Top 5 Features Correlated with Prediction Error ---")
    for idx in top_indices:
        print(f"Feature Index {idx}: Correlation = {correlations[idx]:.4f}")
    print("-------------------------------------------------------------------------")

    # 7. Submission Logic
    threshold = 0.713561265524314

    if val_auc > threshold:
        logger.info(
            f"Validation AUC ({val_auc:.4f}) exceeds threshold ({threshold}). Proceeding to submission."
        )

        # Combine Train and Validation data for final training
        logger.info("Combining Train and Validation sets...")
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.hstack([y_train, y_val])

        # Retrain model on combined data
        logger.info("Retraining model on combined data...")
        final_model = BaggedLREnsemble()
        final_model.optimize_and_fit(X_combined, y_combined)

        # Generate predictions for Test set
        logger.info("Generating test predictions...")
        test_probs = final_model.predict_proba(X_test)

        # Create submission DataFrame
        submission_df = pd.DataFrame(
            {Config.ID_COL: test_ids, Config.TARGET_COL: test_probs}
        )

        # Save submission
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        logger.warning(
            f"Validation AUC ({val_auc:.4f}) did not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
