import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything
from library.data_loader import load_data
from library.features import extract_features
from library.model import ScoreRegressor


def generate_submission(load_cached_data: bool = True, nrows: int = None):
    """
    Orchestrates the inference pipeline:
    1. Loads test data.
    2. Extracts TF-IDF features using the saved vectorizer.
    3. Loads the trained Ridge Regression model.
    4. Generates predictions.
    5. Post-processes predictions (rounding to integers).
    6. Saves the submission file.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
        nrows (int, optional): Number of rows to load for debugging.
    """
    # 1. Setup
    seed_everything()
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print("--- Starting Inference Pipeline ---")

    # 2. Load Data
    print("Loading Test Data...")
    # load_data handles caching logic internally
    df_test = load_data(split="test", load_cached_data=load_cached_data, nrows=nrows)

    # 3. Feature Extraction
    print("Extracting Features (Test)...")
    # extract_features automatically loads the vectorizer from Config.VECTORIZER_PATH
    # when split is not 'train'.
    try:
        X_test = extract_features(df_test, split="test")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Feature extraction failed: {e}. Ensure training has been run to generate the vectorizer."
        )

    print(f"Test Features Shape: {X_test.shape}")

    # 4. Load Model
    print(f"Loading Model from {Config.MODEL_PATH}...")
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_PATH}. Ensure training has been run."
        )

    model = ScoreRegressor.load(Config.MODEL_PATH)

    # 5. Prediction
    print("Generating Predictions...")
    # model.predict handles clipping to [SCORE_MIN, SCORE_MAX]
    preds = model.predict(X_test)

    # Post-processing: Round to nearest integer for QWK metric requirements
    # The regression outputs floats, but the submission requires integers 1-6
    preds_int = np.round(preds).astype(int)

    # 6. Create Submission File
    print("Creating Submission DataFrame...")
    submission = pd.DataFrame(
        {Config.ID_COL: df_test[Config.ID_COL], Config.TARGET_COL: preds_int}
    )

    # Verify submission format
    print(f"Submission Shape: {submission.shape}")
    print(f"Submission Head:\n{submission.head()}")

    # Save
    print(f"Saving submission to {Config.SUBMISSION_FILE}...")
    submission.to_csv(Config.SUBMISSION_FILE, index=False)

    print("--- Inference Pipeline Completed ---")
