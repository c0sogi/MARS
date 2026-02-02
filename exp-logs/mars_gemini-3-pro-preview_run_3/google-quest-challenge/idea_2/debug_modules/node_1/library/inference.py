import os
import numpy as np
import pandas as pd
import torch
import joblib

from library.config import (
    DEVICE,
    MODEL_STATE_DICT_PATH,
    TEST_FEATURES_PATH,
    RIDGE_MODEL_PATH,
    TEST_METADATA_PATH,
    SUBMISSION_PATH,
    TARGET_COLS,
    SUBMISSION_DIR,
)
from library.dataset import get_dataloaders
from library.model import SiameseNetwork
from library.feature_caching import extract_and_save


def predict_and_submit(load_cached_data=True, debug=False):
    """
    Generates the final submission file.

    1. Extracts features from the test set using the fine-tuned SiameseNetwork.
       (Uses cached features if available and requested).
    2. Loads the trained Ridge Regression model.
    3. Predicts target probabilities.
    4. Saves the submission CSV.

    Args:
        load_cached_data (bool): If True, attempts to use cached test features (.npy).
        debug (bool): If True, runs on a subset of data.
    """
    print("Starting Inference and Submission Generation...")

    # Ensure submission directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Feature Extraction (Backbone)
    # -------------------------------------------------------------------------
    # Check if we need to compute features
    features_exist = os.path.exists(TEST_FEATURES_PATH)

    if load_cached_data and features_exist:
        print(f"Loading cached test features from {TEST_FEATURES_PATH}")
    else:
        print("Computing test features using fine-tuned backbone...")

        # Load Model Architecture
        model = SiameseNetwork().to(DEVICE)

        # Load Weights
        if not os.path.exists(MODEL_STATE_DICT_PATH):
            raise FileNotFoundError(
                f"Model weights not found at {MODEL_STATE_DICT_PATH}. "
                "Cannot perform inference without fine-tuned backbone."
            )

        print(f"Loading model weights from {MODEL_STATE_DICT_PATH}")
        model.load_state_dict(torch.load(MODEL_STATE_DICT_PATH, map_location=DEVICE))

        # Get Test DataLoader
        # We only need the test loader (index 2)
        # load_cached_data=load_cached_data here refers to the parquet text cache
        _, _, test_loader = get_dataloaders(
            load_cached_data=load_cached_data, debug=debug
        )

        # Extract and Save
        extract_and_save(
            model, test_loader, TEST_FEATURES_PATH, target_path=None, desc="Test"
        )

        # Free up memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Load the features (Now guaranteed to exist)
    if not os.path.exists(TEST_FEATURES_PATH):
        raise RuntimeError("Test features file missing after extraction step.")

    X_test = np.load(TEST_FEATURES_PATH)
    print(f"Test features shape: {X_test.shape}")

    # -------------------------------------------------------------------------
    # 2. Prediction (Ridge Head)
    # -------------------------------------------------------------------------
    if not os.path.exists(RIDGE_MODEL_PATH):
        raise FileNotFoundError(
            f"Ridge model not found at {RIDGE_MODEL_PATH}. "
            "Please run Stage 2 training (refinement) first."
        )

    print(f"Loading Ridge model from {RIDGE_MODEL_PATH}")
    ridge_model = joblib.load(RIDGE_MODEL_PATH)

    print("Predicting targets...")
    test_preds = ridge_model.predict(X_test)

    # Clip predictions to [0, 1] as required
    test_preds = np.clip(test_preds, 0, 1)

    # -------------------------------------------------------------------------
    # 3. Submission Generation
    # -------------------------------------------------------------------------
    print("Generating submission file...")

    # Load Test Metadata to get QA_IDs
    if not os.path.exists(TEST_METADATA_PATH):
        raise FileNotFoundError(f"Test metadata not found at {TEST_METADATA_PATH}")

    df_test_meta = pd.read_csv(TEST_METADATA_PATH)

    # Handle debug mode mismatch
    # If debug=True, features are subsetted (e.g. 100 rows), but metadata is full.
    if debug:
        print("Debug mode: Subsetting metadata to match predictions.")
        df_test_meta = df_test_meta.head(len(test_preds))

    # Verify alignment
    if len(df_test_meta) != len(test_preds):
        raise ValueError(
            f"Mismatch between test metadata rows ({len(df_test_meta)}) "
            f"and predictions ({len(test_preds)})."
        )

    # Construct DataFrame
    submission_df = pd.DataFrame(test_preds, columns=TARGET_COLS)
    submission_df.insert(0, "qa_id", df_test_meta["qa_id"])

    # Save
    print(f"Saving submission to {SUBMISSION_PATH}")
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    print("Inference completed successfully.")
