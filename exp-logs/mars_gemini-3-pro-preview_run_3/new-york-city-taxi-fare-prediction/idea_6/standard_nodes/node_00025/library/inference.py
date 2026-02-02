import os
import gc
import joblib
import numpy as np
import pandas as pd
from library.config import Config
from library.data_processing import process_data


def generate_ensemble_predictions(load_cached_data=True, debug_sample_size=None):
    """
    Generates predictions for the test set using the ensemble of trained models.

    Args:
        load_cached_data (bool): Whether to attempt loading processed data from cache.
        debug_sample_size (int, optional): Size of data to use for debugging/testing flow.
    """
    print("Starting inference process...")

    # 1. Load Test Data
    # Optimization: Try to load directly from processed cache to avoid loading full train set
    test_df = None
    if load_cached_data and os.path.exists(Config.TEST_PROCESSED_PATH):
        print(
            f"Loading processed test data directly from {Config.TEST_PROCESSED_PATH}..."
        )
        try:
            test_df = pd.read_parquet(Config.TEST_PROCESSED_PATH)
            # Apply sampling if debugging is requested on cached data
            if debug_sample_size is not None and len(test_df) > debug_sample_size:
                test_df = test_df.iloc[:debug_sample_size]
        except Exception as e:
            print(f"Failed to load cached test data directly: {e}")

    if test_df is None:
        # Fallback to full processing pipeline
        print("Loading/Processing data via pipeline...")
        # We discard train and val to save memory
        _, _, test_df = process_data(
            load_cached_data=load_cached_data, debug_sample_size=debug_sample_size
        )
        # Force cleanup of unused dataframes
        del _
        gc.collect()

    # 2. Prepare Features
    # Identify feature columns (exclude key and target if present)
    # Note: processed test_df usually doesn't have fare_amount, but might have key
    exclude_cols = ["key", "fare_amount", "pickup_datetime"]
    feature_cols = [c for c in test_df.columns if c not in exclude_cols]
    print(f"Inference features: {feature_cols}")

    X_test = test_df[feature_cols]

    # Initialize final predictions
    final_preds = np.zeros(len(X_test))
    total_weight = 0.0
    models_found = False

    # 3. Load and Predict with XGBoost
    xgb_path = os.path.join(Config.WORKING_DIR, "xgboost_model.joblib")
    xgb_weight = Config.ENSEMBLE_WEIGHTS.get("xgb", 0.0)

    if xgb_weight > 0 and os.path.exists(xgb_path):
        print(f"Loading XGBoost model from {xgb_path}...")
        try:
            xgb_model = joblib.load(xgb_path)
            print(f"Predicting with XGBoost (Weight: {xgb_weight})...")
            xgb_preds = xgb_model.predict(X_test)
            final_preds += xgb_preds * xgb_weight
            total_weight += xgb_weight
            models_found = True

            # Cleanup
            del xgb_model, xgb_preds
            gc.collect()
        except Exception as e:
            print(f"Error using XGBoost model: {e}")
    elif xgb_weight > 0:
        print(
            f"Warning: XGBoost weight is {xgb_weight} but model file not found at {xgb_path}"
        )

    # 4. Load and Predict with LightGBM
    lgbm_path = os.path.join(Config.WORKING_DIR, "lgbm_model.joblib")
    lgbm_weight = Config.ENSEMBLE_WEIGHTS.get("lgbm", 0.0)

    if lgbm_weight > 0 and os.path.exists(lgbm_path):
        print(f"Loading LightGBM model from {lgbm_path}...")
        try:
            lgbm_model = joblib.load(lgbm_path)
            print(f"Predicting with LightGBM (Weight: {lgbm_weight})...")
            lgbm_preds = lgbm_model.predict(X_test)
            final_preds += lgbm_preds * lgbm_weight
            total_weight += lgbm_weight
            models_found = True

            # Cleanup
            del lgbm_model, lgbm_preds
            gc.collect()
        except Exception as e:
            print(f"Error using LightGBM model: {e}")
    elif lgbm_weight > 0:
        print(
            f"Warning: LightGBM weight is {lgbm_weight} but model file not found at {lgbm_path}"
        )

    if not models_found:
        raise RuntimeError(
            "No trained models found to generate predictions. Check working directory."
        )

    # Normalize if weights don't sum to 1 (though usually they should)
    if total_weight > 0:
        final_preds /= total_weight

    # 5. Create Submission
    submission = pd.DataFrame({"key": test_df["key"], "fare_amount": final_preds})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE_PATH), exist_ok=True)

    print(f"Saving submission to {Config.SUBMISSION_FILE_PATH}...")
    submission.to_csv(Config.SUBMISSION_FILE_PATH, index=False)
    print("Submission generation complete.")
