import os
import numpy as np
import pandas as pd
import joblib
from library.config import (
    N_FOLDS,
    CACHE_DIR,
    SUBMISSION_PATH,
)
from library.utils import setup_logger
from library.data_loader import (
    load_test_data,
    extract_text_data,
    extract_numeric_data,
)
from library.feature_engineering import generate_embeddings, prepare_design_matrix

# Setup logger for inference
logger = setup_logger(os.path.join(CACHE_DIR, "inference.log"))


def run_inference(load_cached_data=True):
    """
    Executes the inference pipeline:
    1. Loads test data.
    2. Generates features (Text Embeddings + Numeric Metadata).
    3. Loads trained fold models.
    4. Aggregates predictions (Soft Voting across folds).
    5. Saves submission file.

    Args:
        load_cached_data (bool): Whether to use cached data/embeddings. Defaults to True.
    """
    logger.info("Starting inference process...")

    # ---------------------------------------------------------
    # 1. Load and Prepare Test Data
    # ---------------------------------------------------------
    logger.info("Loading test data...")
    df_test = load_test_data(load_cached_data=load_cached_data)

    # Text Features
    logger.info("Processing text features for test set...")
    text_data = extract_text_data(df_test)
    # We use "test" as cache_name to distinguish from training embeddings
    embeddings = generate_embeddings(
        text_data, "test", load_cached_data=load_cached_data
    )

    # Numeric Features
    logger.info("Processing numeric features for test set...")
    numeric_data = extract_numeric_data(df_test)

    # Combine into Design Matrix
    # metadata_start_idx is returned here, ensuring we know where embeddings end and metadata begins
    X_test, metadata_start_idx = prepare_design_matrix(embeddings, numeric_data)

    logger.info(f"Test Design Matrix Shape: {X_test.shape}")
    logger.info(f"Metadata starts at index: {metadata_start_idx}")

    # ---------------------------------------------------------
    # 2. Model Loading and Prediction
    # ---------------------------------------------------------
    test_preds_sum = np.zeros(len(df_test))
    models_found = 0

    logger.info(f"Aggregating predictions across {N_FOLDS} folds...")

    for fold in range(N_FOLDS):
        model_path = os.path.join(CACHE_DIR, f"model_fold_{fold}.joblib")

        if not os.path.exists(model_path):
            logger.warning(
                f"Model for fold {fold} not found at {model_path}. Skipping."
            )
            continue

        try:
            logger.info(f"Loading model for fold {fold}...")
            pipeline = joblib.load(model_path)

            # Predict probabilities (class 1: received pizza)
            fold_preds = pipeline.predict_proba(X_test)[:, 1]

            test_preds_sum += fold_preds
            models_found += 1

        except Exception as e:
            logger.error(f"Error processing fold {fold}: {e}")

    if models_found == 0:
        raise RuntimeError("No trained models found. Cannot generate submission.")

    # Average predictions (Ensemble of Ensembles / CV-Bagging)
    avg_preds = test_preds_sum / models_found
    logger.info(f"Successfully aggregated predictions from {models_found} models.")

    # ---------------------------------------------------------
    # 3. Generate Submission File
    # ---------------------------------------------------------
    submission = pd.DataFrame(
        {"request_id": df_test["request_id"], "requester_received_pizza": avg_preds}
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    submission.to_csv(SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {SUBMISSION_PATH}")

    # Print sample for verification
    logger.info("Sample predictions:")
    logger.info(submission.head().to_string())
