import pandas as pd
import numpy as np
import os
import sys

# Import from provided library files
from library.config import Config
from library.utils import set_seed, get_logger
from library.data_loader import DatasetManager
from library.model_xgb import XGBWrapper
from library.metrics import optimize_threshold, calculate_mcc


def main():
    # 1. Setup
    set_seed(Config.SEED)
    logger = get_logger("runfile")
    logger.info("Starting pipeline execution...")

    # 2. Data Loading
    # We use load_cached_data=True to leverage any existing preprocessed data
    dm = DatasetManager(load_cached_data=True)

    logger.info("Loading training data...")
    X_train, y_train, ids_train = dm.get_train_data()

    logger.info("Loading validation data...")
    X_val, y_val, ids_val = dm.get_validation_data()

    # Quick check on data shapes
    logger.info(f"Training Data Shape: {X_train.shape}")
    logger.info(f"Validation Data Shape: {X_val.shape}")

    # 3. Model Training
    # Initialize XGBoost wrapper
    model = XGBWrapper()

    # Train with early stopping
    # The wrapper handles the GPU usage via Config params (tree_method='gpu_hist')
    model.train(X_train, y_train, X_val, y_val)

    # 4. Validation & Threshold Optimization
    logger.info("Performing validation inference...")
    y_val_pred_proba = model.predict(X_val)

    # Find optimal threshold maximizing MCC
    best_threshold, best_mcc = optimize_threshold(y_val, y_val_pred_proba)

    # REQUIRED: Print Final Validation Metric
    print(f"Final Validation Metric: {best_mcc}")

    # 5. Failure Analysis
    logger.info("Performing failure analysis...")
    # Calculate absolute error
    errors = np.abs(y_val - y_val_pred_proba)

    # We calculate correlation between features and the error magnitude
    # This helps identify if specific values of features (e.g. high speed) lead to more errors
    # Using pandas corrwith for efficiency

    # Ensure X_val is a DataFrame for correlation computation
    if isinstance(X_val, np.ndarray):
        # If it was converted to numpy inside model or elsewhere, we might need column names
        # But FeatureEngineer returns DataFrame, so we should be good.
        pass

    try:
        # Create a series for errors with matching index
        error_series = pd.Series(errors, index=X_val.index)

        # Compute correlations
        correlations = X_val.corrwith(error_series).abs().sort_values(ascending=False)

        logger.info("Top 5 features correlated with prediction error:")
        print(correlations.head(5))
    except Exception as e:
        logger.warning(f"Failure analysis correlation computation failed: {e}")

    # 6. Submission Generation
    TARGET_METRIC = 0.637875144942081

    if best_mcc > TARGET_METRIC:
        logger.info(
            f"Validation metric ({best_mcc}) meets threshold ({TARGET_METRIC}). Generating submission..."
        )

        # Load Test Data
        logger.info("Loading test data...")
        X_test, y_test_placeholder, ids_test = dm.get_test_data()

        # Predict
        y_test_pred_proba = model.predict(X_test)

        # Apply Threshold
        y_test_pred_binary = (y_test_pred_proba >= best_threshold).astype(int)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {"contact_id": ids_test, "contact": y_test_pred_binary}
        )

        # Ensure output directory exists
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Save
        save_path = Config.SUBMISSION_PATH
        submission_df.to_csv(save_path, index=False)
        logger.info(f"Submission saved to {save_path}. Rows: {len(submission_df)}")

    else:
        logger.warning(
            f"Validation metric ({best_mcc}) is below threshold ({TARGET_METRIC}). Skipping submission generation."
        )

    logger.info("Pipeline execution completed.")


if __name__ == "__main__":
    main()
