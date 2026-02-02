import os
import numpy as np
import pandas as pd
import xgboost as xgb
from library.config import (
    XGB_PARAMS,
    EARLY_STOPPING_ROUNDS,
    VERBOSE_EVAL,
    CACHE_DIR,
    SUBMISSION_PATH,
)


def train_regressor(X_train, y_train, X_val, y_val, load_cached_model=True):
    """
    Trains the XGBoost Regressor using the provided training and validation data.
    Implements Early Stopping and Model Caching.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (np.ndarray): Training targets.
        X_val (pd.DataFrame): Validation features.
        y_val (np.ndarray): Validation targets.
        load_cached_model (bool): If True, attempts to load a pre-trained model from cache.

    Returns:
        xgb.XGBRegressor: The trained model.
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)
    model_path = os.path.join(CACHE_DIR, "xgb_model.json")

    # Initialize model with configuration parameters
    # Note: eval_metric is included in XGB_PARAMS
    reg = xgb.XGBRegressor(**XGB_PARAMS)

    # Attempt to load from cache
    if load_cached_model and os.path.exists(model_path):
        print(f"Loading cached XGBoost model from {model_path}...")
        try:
            reg.load_model(model_path)
            print("Model loaded successfully.")
            return reg
        except Exception as e:
            print(f"Failed to load cached model: {e}. Training from scratch...")

    print("Training XGBoost Regressor from scratch...")

    # Train the model
    # We pass eval_metric here explicitly or rely on constructor.
    # Passing in fit is standard for early stopping monitoring.
    reg.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=VERBOSE_EVAL,
    )

    # Log best performance
    if hasattr(reg, "best_score"):
        print(f"Best Iteration: {reg.best_iteration}")
        # Print full precision as requested
        print(f"Best Validation RMSE: {reg.best_score:.20f}")

    # Save the trained model to cache
    print(f"Saving model to {model_path}...")
    reg.save_model(model_path)

    return reg


def predict_fare(model, X_test):
    """
    Generates fare predictions for the test set using the trained model.
    Applies a post-processing floor to ensure no predictions are below the base fare.

    Args:
        model (xgb.XGBRegressor): The trained model.
        X_test (pd.DataFrame): Test features.

    Returns:
        np.ndarray: Predicted fare amounts.
    """
    print("Generating predictions on test set...")

    # Generate raw predictions
    preds = model.predict(X_test)

    # Apply Post-Processing: Floor at $2.50
    # The minimum fare for a NYC taxi is $2.50.
    preds = np.maximum(preds, 2.50)

    return preds


def generate_submission(predictions, test_keys, output_path=SUBMISSION_PATH):
    """
    Saves the predictions to a CSV file in the format required for submission.

    Args:
        predictions (np.ndarray): Array of predicted fare amounts.
        test_keys (np.ndarray): Array of key strings corresponding to the test set.
        output_path (str): Path to save the submission CSV.
    """
    print(f"Saving submission file to {output_path}...")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Construct DataFrame
    submission_df = pd.DataFrame({"key": test_keys, "fare_amount": predictions})

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved successfully. Shape: {submission_df.shape}")
