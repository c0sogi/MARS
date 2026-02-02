import pandas as pd
import numpy as np
import gc
import os
from library.config import Config
from library.feature_engineering import FeatureProcessor
from library.models import LGBMWrapper, XGBWrapper


def predict_and_submit(threshold=0.5, load_cached_data=True):
    """
    Generates predictions for the test set using the trained ensemble models
    and creates the submission file.

    Args:
        threshold (float): The probability threshold for binary classification.
                           Defaults to 0.5 if not optimized.
        load_cached_data (bool): Whether to attempt loading test features from cache.
                                 If False or cache missing, features are regenerated.

    Returns:
        pd.DataFrame: The submission dataframe.
    """
    print("Initializing Feature Processor for Inference...")
    processor = FeatureProcessor()

    # ---------------------------------------------------------
    # 1. Load and Process Test Data
    # ---------------------------------------------------------
    print("Loading Test Data...")
    # process_split handles caching, loading metadata/tracking, merging, and feature engineering
    df_test = processor.process_split("test", load_cached_data=load_cached_data)

    # ---------------------------------------------------------
    # 2. Prepare Features
    # ---------------------------------------------------------
    # Define columns to exclude to isolate feature columns
    # Must match the exclusion list used in train.py
    exclude_cols = [
        "contact_id",
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "contact",
        "datetime",
        "p1_id",
        "p2_id",
    ]

    # Identify feature columns dynamically
    feature_cols = [c for c in df_test.columns if c not in exclude_cols]
    print(f"Inference with {len(feature_cols)} features.")

    # Extract features for prediction
    X_test = df_test[feature_cols]

    # Extract contact_ids for the submission file
    # Ensure we copy to avoid issues when deleting df_test
    contact_ids = df_test["contact_id"].values.copy()

    # Clean up the large dataframe to free memory
    del df_test
    gc.collect()

    # ---------------------------------------------------------
    # 3. Load Models
    # ---------------------------------------------------------
    print("Loading LightGBM model...")
    lgbm_model = LGBMWrapper()
    # Model wrapper handles path joining with Config.WORKING_DIR
    lgbm_model.load("lgbm_model.joblib")

    print("Loading XGBoost model...")
    xgb_model = XGBWrapper()
    xgb_model.load("xgb_model.joblib")

    # ---------------------------------------------------------
    # 4. Generate Predictions (Ensemble)
    # ---------------------------------------------------------
    print("Generating predictions...")

    # Get probabilities from LightGBM
    probs_lgbm = lgbm_model.predict_proba(X_test)

    # Get probabilities from XGBoost
    probs_xgb = xgb_model.predict_proba(X_test)

    # Calculate Ensemble Average (Unweighted)
    ensemble_probs = (probs_lgbm + probs_xgb) / 2.0

    # Apply Threshold to get Binary Predictions
    predictions = (ensemble_probs >= threshold).astype(int)

    # ---------------------------------------------------------
    # 5. Create and Save Submission
    # ---------------------------------------------------------
    submission = pd.DataFrame({"contact_id": contact_ids, "contact": predictions})

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save to CSV
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    return submission
