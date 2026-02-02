import os
import joblib
import numpy as np
import pandas as pd

from library.config import TEST_PATH, WORKING_DIR, SUBMISSION_PATH, SEED
from library.data_loader import clean_test_data
from library.feature_engineering import FeatureEngineer


def generate_submission(
    load_cached_data=True, model_suffix="", debug=False, debug_size=100000
):
    """
    Manages the prediction workflow for the test set.
    Loads data, performs feature engineering, loads trained models,
    generates predictions, and saves the submission file.

    Args:
        load_cached_data (bool): Whether to use cached feature engineering results.
        model_suffix (str): Suffix for model files (e.g., "_debug") to load specific versions.
        debug (bool): If True, subsamples the test set for rapid debugging.
        debug_size (int): Number of rows to use if debug is True.

    Returns:
        pd.DataFrame: The generated submission DataFrame.
    """

    # 1. Load Data
    # We load directly from the metadata parquet file to avoid loading the large training set
    print(f"Loading test data from {TEST_PATH}...")
    test_df = pd.read_parquet(TEST_PATH)

    # Handle Debugging
    # Determine the cache name based on suffix and debug status
    cache_name = f"test{model_suffix}"

    if debug:
        print(f"DEBUG MODE: Sampling {debug_size} rows from test set...")
        test_df = test_df.sample(min(len(test_df), debug_size), random_state=SEED)
        # Use a distinct cache name for inference debug to avoid collisions with full test processing
        cache_name = f"test_inference_debug_{debug_size}"

    # 2. Clean Data
    # Apply bounding box clipping
    print("Cleaning test data...")
    test_df = clean_test_data(test_df)

    # 3. Feature Engineering
    # Initialize FeatureEngineer with caching preference
    fe = FeatureEngineer(load_cached_data=load_cached_data)
    # Process the data (adds time features, rotated coords, physics features, airport flags)
    test_df = fe.process(test_df, name=cache_name)

    # 4. Prepare Feature Matrix
    # Identify feature columns by excluding metadata/target columns
    ignore_cols = ["key", "fare_amount", "pickup_datetime"]
    features = [c for c in test_df.columns if c not in ignore_cols]
    X_test = test_df[features]

    print(f"Test feature matrix shape: {X_test.shape}")

    # 5. Load Models
    # Construct paths for the stacked ensemble components
    xgb_path = os.path.join(WORKING_DIR, f"xgb_model{model_suffix}.joblib")
    lgbm_path = os.path.join(WORKING_DIR, f"lgbm_model{model_suffix}.joblib")
    meta_path = os.path.join(WORKING_DIR, f"meta_model{model_suffix}.joblib")

    print(f"Loading models from {WORKING_DIR} with suffix '{model_suffix}'...")

    # Verify model existence
    for path in [xgb_path, lgbm_path, meta_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

    # Load models using joblib
    xgb_model = joblib.load(xgb_path)
    lgbm_model = joblib.load(lgbm_path)
    meta_model = joblib.load(meta_path)

    # 6. Generate Predictions
    print("Generating base learner predictions...")
    xgb_pred = xgb_model.predict(X_test)
    lgbm_pred = lgbm_model.predict(X_test)

    print("Generating meta learner predictions...")
    # Stack base predictions: [xgb, lgbm] to form Level 1 input
    X_stack = np.column_stack((xgb_pred, lgbm_pred))

    # Predict using the Ridge Meta-Learner
    final_pred = meta_model.predict(X_stack)

    # 7. Create and Save Submission
    submission = pd.DataFrame({"key": test_df["key"], "fare_amount": final_pred})

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")

    return submission
